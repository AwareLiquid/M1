"""Streaming-state parity tests (CPU, tiny random llama).

The claim under test: with set_adapter_streaming(model, True), feeding a
sequence in arbitrary chunks through the KV-cached forward produces the SAME
logits as one full-sequence forward — i.e. the adapter's recurrence now
carries across calls exactly like the training-time full-sequence scan.

And the control: with streaming OFF (the old behaviour), chunked decode does
NOT match the full forward — proving the train/serve mismatch this fixes was
real and the test has discriminating power.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mt_lnn.llama_adapter import (
    attach_mt_adapters,
    adapter_streaming_paused,
    reset_adapter_streams,
    set_adapter_streaming,
    _iter_all_adapters,
)
from mt_lnn.mt_lnn_v2 import attach_mt_v2_adapters

CHUNKS = [7, 9, 15, 1]          # deliberately ragged, ends with a T=1 decode step
T_TOTAL = sum(CHUNKS)


def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128,
    )
    return LlamaForCausalLM(cfg)


def boost_scales(model):
    """Raise residual gates from 1e-3 so state effects are numerically visible."""
    for a in _iter_all_adapters(model):
        a.scale.data.fill_(0.5)
        if getattr(a, "fw_scale", None) is not None:
            a.fw_scale.data.fill_(0.5)


@torch.no_grad()
def chunked_logits(model, ids):
    past, outs, pos = None, [], 0
    for c in CHUNKS:
        chunk = ids[:, pos: pos + c]
        out = model(input_ids=chunk, past_key_values=past, use_cache=True)
        past = out.past_key_values
        outs.append(out.logits)
        pos += c
    return torch.cat(outs, dim=1)


def run_parity(name, attach_fn):
    model = tiny_model()
    attach_fn(model)
    boost_scales(model)
    model.eval()
    ids = torch.randint(0, 128, (1, T_TOTAL), generator=torch.Generator().manual_seed(1))

    with torch.no_grad():
        full = model(input_ids=ids).logits

    # Control: streaming OFF -> chunked must NOT match (the old broken path)
    stale = chunked_logits(model, ids)
    assert not torch.allclose(full, stale, rtol=1e-3, atol=1e-4), (
        f"[{name}] chunked-without-state unexpectedly matches full forward — "
        f"either the adapter is inert or the test lost its power"
    )

    # Streaming ON -> exact parity with the full-sequence forward
    n = set_adapter_streaming(model, True)
    assert n > 0
    streamed = chunked_logits(model, ids)
    diff = (full - streamed).abs().max().item()
    assert torch.allclose(full, streamed, rtol=1e-4, atol=1e-5), (
        f"[{name}] streaming parity failed: max|diff|={diff:.2e}"
    )

    # Reset semantics: a fresh sequence after reset reproduces a fresh run
    reset_adapter_streams(model)
    streamed2 = chunked_logits(model, ids)
    assert torch.allclose(full, streamed2, rtol=1e-4, atol=1e-5), (
        f"[{name}] parity broken on second sequence after reset"
    )

    # Paused probes must not perturb the stream
    reset_adapter_streams(model)
    past, outs, pos = None, [], 0
    with torch.no_grad():
        for c in CHUNKS:
            chunk = ids[:, pos: pos + c]
            out = model(input_ids=chunk, past_key_values=past, use_cache=True)
            past = out.past_key_values
            outs.append(out.logits)
            pos += c
            with adapter_streaming_paused(model):   # mid-stream side forward
                model(input_ids=ids[:, :5])
    streamed3 = torch.cat(outs, dim=1)
    assert torch.allclose(full, streamed3, rtol=1e-4, atol=1e-5), (
        f"[{name}] paused side-forward perturbed the stream"
    )

    set_adapter_streaming(model, False)
    print(f"[{name}] parity + reset + pause  OK  (control diff was real, "
          f"streamed max|diff| {diff:.1e})")


def test_v1_streaming_parity():
    run_parity("v1", lambda m: attach_mt_adapters(
        m, every=4, n_protofilaments=4, n_time_scales=3, map_hidden_dim=16))


def test_v1_fastweight_streaming_parity():
    run_parity("v1+fw", lambda m: attach_mt_adapters(
        m, every=4, n_protofilaments=4, n_time_scales=3, map_hidden_dim=16,
        use_fast_weight=True, fast_weight_dim=8))


def test_v2_streaming_parity():
    run_parity("v2", lambda m: attach_mt_v2_adapters(
        m, every=4, n_protofilaments=4, d_proto=16, n_time_scales=3,
        proj_rank=8, fast_weight_dim=8))


def test_v2_selective_streaming_parity():
    run_parity("v2s", lambda m: attach_mt_v2_adapters(
        m, every=4, n_protofilaments=4, d_proto=16, n_time_scales=3,
        proj_rank=8, fast_weight_dim=8, selective_decay=True))


def test_train_through_state_keeps_graph():
    """Cross-window training depends on carried state STAYING in the autograd
    graph (stream_in_training + stream_detach=False): segment A's compute must
    be reachable from a loss on segment B."""
    model = tiny_model()
    attach_mt_v2_adapters(model, every=4, n_protofilaments=4, d_proto=16,
                          n_time_scales=3, proj_rank=8, fast_weight_dim=8)
    adapters = list(_iter_all_adapters(model))
    for a in adapters:
        a.stream_enabled = True
        a.stream_in_training = True
        a.stream_detach = False
        a.reset_stream()
    model.train()

    ids_a = torch.randint(0, 128, (2, 8))
    ids_b = torch.randint(0, 128, (2, 8))
    model(input_ids=ids_a, use_cache=False)
    for a in adapters:
        assert a._stream_h is not None and a._stream_h.grad_fn is not None, \
            "carried MT state was detached — segment A unreachable from B's loss"
        assert a._stream_fw is not None and a._stream_fw[0].grad_fn is not None, \
            "carried fast-weight state was detached"
    out_b = model(input_ids=ids_b, use_cache=False)
    out_b.logits.sum().backward()   # must not raise; graph spans both segments

    # And the default (inference) mode still detaches:
    for a in adapters:
        a.stream_in_training = False
        a.stream_detach = True
        a.reset_stream()
    model.eval()
    with torch.no_grad():
        model(input_ids=ids_a, use_cache=False)
    for a in adapters:
        assert a._stream_h.grad_fn is None
    print("[train-through] carried state stays in graph; inference detaches  OK")


if __name__ == "__main__":
    test_v1_streaming_parity()
    test_v1_fastweight_streaming_parity()
    test_v2_streaming_parity()
    test_v2_selective_streaming_parity()
    test_train_through_state_keeps_graph()
    print("all streaming-state tests passed")
