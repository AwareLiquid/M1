"""Cross-session snapshot/restore fidelity (CPU, tiny random llama).

The claim under test: snapshot_adapter_streams -> torch.save -> (fresh model
object, as if a new process) -> torch.load -> restore_adapter_streams
reproduces the fast-weight read EXACTLY — a segment-B continuation from a
restored snapshot equals the continuation from the live state it was taken
from. This is the round-trip that turns the volatile (F, z) into durable
cross-session memory; if it is lossy, every downstream consolidation claim
is built on sand.

Control: restoring nothing (zeros) must DIVERGE — proving the state actually
carries the segment-A information and the test has discriminating power.
"""

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mt_lnn.llama_adapter import (
    _iter_all_adapters,
    reset_adapter_streams,
    restore_adapter_streams,
    set_adapter_streaming,
    snapshot_adapter_streams,
)
from mt_lnn.mt_lnn_v2 import attach_mt_v2_adapters


def build(seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128,
    )
    m = LlamaForCausalLM(cfg)
    attach_mt_v2_adapters(m, every=4, n_protofilaments=4, d_proto=16,
                          n_time_scales=3, proj_rank=8, fast_weight_dim=8)
    # Boost residual gates so the fast-weight state visibly moves the logits.
    for a in _iter_all_adapters(m):
        a.scale.data.fill_(0.5)
        if getattr(a, "fw_scale", None) is not None:
            a.fw_scale.data.fill_(0.5)
    m.eval()
    return m


def test_snapshot_restore_fidelity_across_fresh_model():
    seg_a = torch.randint(0, 128, (1, 24), generator=torch.Generator().manual_seed(1))
    seg_b = torch.randint(0, 128, (1, 10), generator=torch.Generator().manual_seed(2))

    # --- Live model: write segment A, snapshot, then continue with segment B.
    live = build()
    sd = live.state_dict()                      # identical weights for the twin
    set_adapter_streaming(live, True)
    reset_adapter_streams(live)
    with torch.no_grad():
        live(input_ids=seg_a, use_cache=False)  # writes (F, z)
    snap = snapshot_adapter_streams(live)

    # snapshot must be CPU float32 (dtype/device round-trip mitigation)
    for k, e in snap.items():
        if k == "_schema" or e["fw"] is None:
            continue
        for t in e["fw"]:
            assert t.device.type == "cpu" and t.dtype == torch.float32

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "session.pt"
        torch.save(snap, path)                  # durable: survives process death
        loaded = torch.load(path, weights_only=False)

        with torch.no_grad():
            logits_ref = live(input_ids=seg_b, use_cache=False).logits

        # --- Fresh model (new object == new process), identical weights.
        twin = build()
        twin.load_state_dict(sd)
        twin.eval()

        # Control: NO restore -> zeros -> must diverge from the live continuation.
        set_adapter_streaming(twin, True)
        reset_adapter_streams(twin)
        with torch.no_grad():
            logits_ctrl = twin(input_ids=seg_b, use_cache=False).logits
        assert not torch.allclose(logits_ref, logits_ctrl, rtol=1e-3, atol=1e-4), (
            "no-restore control matches the live continuation — either the "
            "fast-weight is inert or the test lost its power"
        )

        # Restore the persisted snapshot -> must reproduce the live continuation.
        n = restore_adapter_streams(twin, loaded)
        assert n > 0, "no adapters restored"
        with torch.no_grad():
            logits_restored = twin(input_ids=seg_b, use_cache=False).logits
        diff = (logits_ref - logits_restored).abs().max().item()
        assert torch.allclose(logits_ref, logits_restored, rtol=1e-4, atol=1e-5), (
            f"restored continuation diverges from live: max|diff|={diff:.2e}"
        )

    print(f"[fidelity] snapshot->save->load->restore is lossless "
          f"(max|diff| {diff:.1e}); no-restore control diverges  OK")


def test_arr_mixer_roundtrip():
    """The O-series (ARR, attention-free) carries the same (F,z) state on
    MTRecurrentMixer, not on an adapter. snapshot/restore must cover it — the
    review found the adapter-only iterator silently no-op'd ARR persistence."""
    from mt_lnn.arr import convert_to_arr

    from transformers import LlamaConfig, LlamaForCausalLM

    def build_arr(seed=0):
        torch.manual_seed(seed)
        cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                          num_hidden_layers=6, num_attention_heads=4,
                          num_key_value_heads=4, max_position_embeddings=128)
        m = LlamaForCausalLM(cfg)
        convert_to_arr(m, n_protofilaments=4, d_proto=16, n_time_scales=3,
                       proj_rank=8, fast_weight_dim=8)
        m.eval()
        return m

    live = build_arr()
    sd = live.state_dict()
    n = set_adapter_streaming(live, True)          # must now touch mixers
    assert n > 0, "set_adapter_streaming found no ARR mixers — iterator too narrow"
    seg_a = torch.randint(0, 128, (1, 20), generator=torch.Generator().manual_seed(3))
    seg_b = torch.randint(0, 128, (1, 8), generator=torch.Generator().manual_seed(4))
    reset_adapter_streams(live)
    with torch.no_grad():
        live(input_ids=seg_a, use_cache=False)
    snap = snapshot_adapter_streams(live)
    assert any(k != "_schema" for k in snap), "snapshot captured no mixer state"

    with torch.no_grad():
        logits_ref = live(input_ids=seg_b, use_cache=False).logits

    twin = build_arr(); twin.load_state_dict(sd); twin.eval()
    set_adapter_streaming(twin, True); reset_adapter_streams(twin)
    with torch.no_grad():
        logits_ctrl = twin(input_ids=seg_b, use_cache=False).logits
    assert not torch.allclose(logits_ref, logits_ctrl, rtol=1e-3, atol=1e-4), \
        "ARR no-restore control matches — mixer state inert or test powerless"

    assert restore_adapter_streams(twin, snap) > 0, "no ARR mixers restored"
    with torch.no_grad():
        logits_restored = twin(input_ids=seg_b, use_cache=False).logits
    diff = (logits_ref - logits_restored).abs().max().item()
    assert torch.allclose(logits_ref, logits_restored, rtol=1e-4, atol=1e-5), \
        f"ARR mixer restore diverges: max|diff|={diff:.2e}"
    print(f"[arr] mixer snapshot/restore lossless (max|diff| {diff:.1e})  OK")


def test_batch_guard_drops_mismatch():
    """restore_adapter_streams(batch=B) must DROP entries whose snapshot batch
    != B (and not count them), the guard the docstring promises."""
    m = build()
    set_adapter_streaming(m, True); reset_adapter_streams(m)
    with torch.no_grad():
        m(input_ids=torch.randint(0, 128, (1, 12)))   # B=1 snapshot
    snap = snapshot_adapter_streams(m)
    twin = build(); twin.load_state_dict(m.state_dict())
    assert restore_adapter_streams(twin, snap, batch=4) == 0, \
        "B=1 snapshot into batch=4 should drop all entries"
    assert restore_adapter_streams(twin, snap, batch=1) > 0, \
        "B=1 snapshot into batch=1 should restore"
    print("[batch-guard] mismatched-batch entries dropped, matched restored  OK")


def test_empty_stream_roundtrips():
    m = build()
    set_adapter_streaming(m, True)
    reset_adapter_streams(m)                     # never written
    snap = snapshot_adapter_streams(m)
    assert all(e["fw"] is None for k, e in snap.items() if k != "_schema")
    twin = build()
    twin.load_state_dict(m.state_dict())
    assert restore_adapter_streams(twin, snap) > 0   # restores None cleanly
    print("[empty] never-written stream snapshots/restores as None  OK")


if __name__ == "__main__":
    test_snapshot_restore_fidelity_across_fresh_model()
    test_arr_mixer_roundtrip()
    test_batch_guard_drops_mismatch()
    test_empty_stream_roundtrips()
    print("all cross-session snapshot tests passed")
