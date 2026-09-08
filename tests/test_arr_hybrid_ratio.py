"""O-series hybrid-ratio: layer-subset conversion, the bit-equivalence of the
historical default, the analytic state ledger, and a one-step distillation
smoke of the whole ratio sweep.

Budgets: everything here runs on CPU with tiny configs; the whole module is
designed to finish in well under 2 minutes (Phase B gate runs it together
with the rest of the suite).
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mt_lnn.arr import (ARRDecoderLayer, MTRecurrentMixer, convert_to_arr,
                        iter_mixer_parameters, measured_state_bytes,
                        normalize_layer_indices, parse_ratio,
                        plan_hybrid_layers, recurrent_state_elems,
                        select_attention_layers)

N_LAYERS = 6
KEEP = {1, 3}
MIXER_KW = {"d_proto": 16, "proj_rank": 16, "fast_weight_dim": 8}


def test_layer_subset_conversion_leaves_the_rest_bit_identical():
    """The converted layers become mixers; every other layer is untouched.

    "Untouched" is asserted bitwise on the attention projections, not by
    type — a wrapper that forwards to the same weights would pass a weaker
    check while still changing the KV path.
    """
    pristine = _llama()
    model = _llama()
    converted = convert_to_arr(model, layer_indices=KEEP, **MIXER_KW)
    assert converted == sorted(KEEP)
    layers, base = _decoder_layers(model), _decoder_layers(pristine)
    for idx in range(N_LAYERS):
        if idx in KEEP:
            assert isinstance(layers[idx], ARRDecoderLayer), idx
            assert isinstance(layers[idx].mixer, MTRecurrentMixer), idx
            continue
        assert not isinstance(layers[idx], ARRDecoderLayer), idx
        new, old = layers[idx].self_attn, base[idx].self_attn
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert torch.equal(getattr(new, name).weight,
                               getattr(old, name).weight), (idx, name)
    # A hybrid still contains attention, so it still needs a KV cache — this
    # is the line that separates it from the O(1) O-series.
    assert model.config.use_cache is True
    mixers = [m for m in model.modules() if isinstance(m, MTRecurrentMixer)]
    assert len(mixers) == len(KEEP)


def test_full_conversion_is_bit_equivalent_to_the_historical_default():
    """None must behave EXACTLY like before the subset parameter existed.

    Pinned by comparing the initialised mixer tensors of three conversions of
    identically seeded models: the implicit default, an explicit ordered list,
    and an unordered set. Same RNG draw order => bitwise-equal mixers.
    """
    variants = {"none": None, "list": list(range(N_LAYERS)),
                "set": {5, 0, 3, 1, 4, 2}}
    params, caches = {}, {}
    for name, indices in variants.items():
        model = _llama()
        returned = convert_to_arr(model, layer_indices=indices, **MIXER_KW)
        assert returned == list(range(N_LAYERS)), name
        params[name] = _mixer_params(model)
        caches[name] = model.config.use_cache
        mixer_ids = {id(p) for p in iter_mixer_parameters(model)}
        assert all(not p.requires_grad for p in model.parameters()
                   if id(p) not in mixer_ids), name
    for name in ("list", "set"):
        assert len(params[name]) == len(params["none"])
        assert all(torch.equal(a, b) for a, b
                   in zip(params["none"], params[name])), name
    assert set(caches.values()) == {False}      # full conversion: no KV cache


def test_analytic_state_bytes_match_the_live_streaming_state():
    """The memory ledger is analytic; this pins it to the real tensors.

    The ledger has to be computable without a GPU, but a formula nobody has
    checked against a forward is a claim, not a ledger.
    """
    mixer = MTRecurrentMixer(hidden_size=32, n_protofilaments=5, d_proto=8,
                             proj_rank=8, n_time_scales=3, fast_weight_dim=6,
                             fast_weight_heads=2).eval()
    assert measured_state_bytes(mixer) == 0      # nothing carried yet
    mixer.stream_enabled = True
    mixer(torch.randn(1, 7, 32))
    h, (fmat, z) = mixer._stream_h, mixer._stream_fw
    assert h.shape == (1, 5, 3, 8)
    assert fmat.shape == (1, 2, 6, 6) and z.shape == (1, 2, 6)
    elems = recurrent_state_elems(5, 8, 3, 6, 2)
    assert elems == 5 * 3 * 8 + 2 * 6 * 6 + 2 * 6
    assert sum(t[0].numel() for t in (h, fmat, z)) == elems
    assert measured_state_bytes(mixer, elem_bytes=4) == elems * 4


def test_ratio_plan_is_monotone_and_covers_every_layer():
    """1:8 < 1:4 <= 1:2 attention layers, spread over the depth, no overlaps."""
    assert (parse_ratio("0"), parse_ratio("1:8")) == (0.0, 0.125)
    assert (parse_ratio("1:4"), parse_ratio("1:2")) == (0.25, 0.5)
    assert plan_hybrid_layers(8, 0.0) == ([], list(range(8)))
    counts = {}
    for text in ("0", "1:8", "1:4", "1:2"):
        keep, converted = plan_hybrid_layers(22, parse_ratio(text))
        assert sorted(keep + converted) == list(range(22))
        assert not set(keep) & set(converted)
        counts[text] = len(keep)
    assert counts["0"] == 0
    assert counts["1:8"] < counts["1:4"] <= counts["1:2"]
    spread = select_attention_layers(22, 0.25)
    assert spread[0] < 22 // 4 and spread[-1] > 22 // 2   # not front-loaded
    try:
        normalize_layer_indices({9}, N_LAYERS)
        raise AssertionError("out-of-range layer index was accepted")
    except ValueError:
        pass


def test_smoke_sweep_covers_all_four_ratios():
    """One KD step per ratio on a random teacher: wiring, not quality.

    The real check here is the memory account's SHAPE: ratio 0 must be the
    only row with an empty KV column, state must shrink and KV must grow as
    attention is back-filled, and 2 seeds must not be promotable.
    """
    import benchmarks.arr_ratio_sweep as sweep

    args = _smoke_args()
    runs = sweep._smoke_runs(args)
    assert [r["ratio"] for r in runs] == list(sweep.RATIOS)
    assert all(math.isfinite(r["val_ppl"]) and r["val_ppl"] > 0 for r in runs)
    memory = {r["ratio"]: r["memory"] for r in runs}
    for bits in ("16", "2"):
        for context in ("512", "1048576"):
            assert memory["0"]["kv_bytes"][bits][context] == 0
    states = [memory[r]["recurrent_state_bytes"] for r in sweep.RATIOS]
    assert states[0] > 0 and states == sorted(states, reverse=True)
    kvs = [memory[r]["kv_bytes"]["16"]["8192"] for r in sweep.RATIOS]
    assert kvs[0] == 0 and kvs == sorted(kvs)
    ledger = sweep._ledger(runs, args)
    assert [r["ratio"] for r in ledger["rows"]] == list(sweep.RATIOS)
    assert "control" in ledger["rows"][0]["verdict"]
    assert ledger["promotion"]["seeds_required"] == 3
    assert not ledger["promotion"]["promotable"]   # 2 seeds = exploratory only


def _smoke_args(**overrides):
    """Parse the sweep's real defaults — no duplicated defaults to drift."""
    import benchmarks.arr_ratio_sweep as sweep

    argv = sys.argv
    sys.argv = ["arr_ratio_sweep", "--smoke"]
    try:
        args = sweep._parse_args()
    finally:
        sys.argv = argv
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=N_LAYERS, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=128)
    return LlamaForCausalLM(cfg)


def _decoder_layers(model):
    from mt_lnn.llama_adapter import find_decoder_layers
    return find_decoder_layers(model)


def _mixer_params(model):
    return [p.detach().clone() for p in iter_mixer_parameters(model)]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[arr-hybrid-ratio] {name} OK")
    print("all arr hybrid-ratio tests passed")


def test_subset_ledger_without_ratio_0_does_not_crash():
    """子集运行（--ratios 不含 "0"）收尾写盘不得崩 — 2026-08-31 A100 事故回归点。

    事故: _ratio_row 无条件算 control-mean，子集无对照行时 control=None →
    TypeError，已完成的臂全部丢失规范文件（commit 1ed4625 的 provenance）。
    修复后: gain/gap 如实记 None，_verdict 打 "no ratio-0 control" 标，
    _promotion 记不可判 —— 账本照常落盘。
    """
    import argparse
    import benchmarks.arr_ratio_sweep as sweep

    def run(ratio, seed, ppl):
        return {"ratio": ratio, "seed": seed, "model": "tiny",
                "val_ppl": ppl, "teacher_ppl": 11.8,
                "ppl_after_stage_a": None, "attention_layers": [],
                "n_layers": 6, "n_recurrent_layers": 6, "d_head": 8,
                "n_kv_heads": 8, "mixer_params": 0,
                "memory": sweep._memory_row(6, 0, 8, 8, 16,
                                            sweep.CONTEXTS)}

    runs = [run("1:4", 0, 180.0), run("1:4", 1, 170.0)]
    args = argparse.Namespace(ratios=["1:4"], seeds=[0, 1], model="tiny",
                              smoke=False, contexts=list(sweep.CONTEXTS))
    ledger = sweep._ledger(runs, args)  # 修复前: TypeError (control=None)
    row = ledger["rows"][0]
    assert row["ppl_gain_vs_control"] is None
    assert row["gap_closed_vs_teacher"] is None
    assert "no ratio-0 control" in row["verdict"]
    assert ledger["promotion"]["promotable"] is False


def test_stage_b_warmup_scale_is_wellformed():
    """stage-B warmup 旋钮: off 位恒 1.0(现行协议不变); 开位线性升至 1。"""
    import benchmarks.distill_arr as distill
    assert distill._warmup_scale(0, 0) == 1.0
    assert distill._warmup_scale(500, 0) == 1.0
    assert abs(distill._warmup_scale(0, 100) - 0.01) < 1e-9
    assert abs(distill._warmup_scale(98, 100) - 0.99) < 1e-9
    assert distill._warmup_scale(100, 100) == 1.0
    assert distill._warmup_scale(500, 100) == 1.0
