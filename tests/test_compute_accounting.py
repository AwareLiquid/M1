"""compute_accounting.py 的手算值锁定 + 与真实模型参数量对账。

锁定三类不变量：
  1. 手算值（transformer 块稠密 123,136 MACs/token、LNN 子层 49,569）——
     防止公式被无意改动（Task 3 的横轴全部依赖这两个数）。
  2. 解析权重计数 == 真实构建模型（probe 形状）的非偏置权重参数量——
     防止账本枚举漏模块（coherence/gwtb 这类容易漏）。
  3. 结构不变量：stack ≥ core 同深度；CoT 随 N 单调；N=0 = 直接作答。

Run:  python -m pytest tests/test_compute_accounting.py -v  (全 CPU, <10s)
"""

import sys

import pytest

sys.path.insert(0, ".")

from benchmarks.compute_accounting import (
    PROBE, account_mtlnn, account_tfm, mtlnn_block_macs, mtlnn_top_macs,
    tfm_block_macs, _mtlnn_weight_counts,
)

# 与构建模型对账时排除的名字（偏置 / 不在默认 forward 路径的模块）。
_EXCLUDE = ("bias", "b_in", "log_tau", "blend_weights", "kappa_gate",
            "embedding", "target_head", "target_queries",
            "attn_norm", "lnn_norm", "compress_norm", "workspace_norm",
            "layer_norm", "final_norm", "target_norm")


def test_transformer_block_dense_hand_value():
    """4·d² + 3·d·d_ff = 4·10816 + 3·26624 = 123,136 MACs/token/layer。"""
    assert tfm_block_macs(PROBE)["dense"] == 4 * 104**2 + 3 * 104 * 256
    assert tfm_block_macs(PROBE)["dense"] == 123136


def test_mtlnn_lnn_dense_hand_value():
    """LNN 子层逐项手算：10816+4160+1352+1664+3328+2704+13312+832+10816+65。"""
    hand = (10816 + 4160 + 1352 + 1664 + 3328 + 2704 + 13312 + 832
            + 10816 + 65)
    blk = mtlnn_block_macs(PROBE)
    assert blk["lnn_dense"] == hand == 49049
    assert blk["attn_dense"] == 32448
    assert blk["scan"] == PROBE["P"] * PROBE["S"] * PROBE["D"]


def test_weight_counts_match_built_model():
    """解析权重计数必须等于真实模型（probe 形状）的非偏置权重元素数。"""
    torch = pytest.importorskip("torch")
    from benchmarks.reasoning_depth import build_mtlnn, build_transformer

    sh = dict(PROBE, vocab=10 + PROBE["n_values"])
    m = build_mtlnn(sh["vocab"], sh["T"], max_depth=1, seed=0)
    actual = sum(p.numel() for n, p in m.named_parameters()
                 if p.ndim >= 2 and not any(e in n for e in _EXCLUDE))
    w = _mtlnn_weight_counts(sh)
    top = mtlnn_top_macs(sh)
    # lm_head 与 embedding 绑定：真实模型只计一次（在 embedding 名下，已被
    # 排除），账本公式单独保留 head 行；对账时从公式侧减掉。
    formula = sh["n_layers"] * w["block"] + w["top"] - top["lm_head"]
    assert formula == actual, f"ledger {formula} != model {actual}"
    assert top["lm_head"] == 104 * sh["vocab"]

    t = build_transformer(sh["vocab"], sh["T"], seed=0)
    t_actual = sum(p.numel() for n, p in t.named_parameters()
                   if p.ndim >= 2 and "embedding" not in n)
    blk = tfm_block_macs(sh)
    assert sh["n_layers"] * blk["dense"] == t_actual


def test_structural_invariants():
    sh = dict(PROBE, vocab=26)
    for n in (1, 2, 4, 8):
        core = account_mtlnn(sh, n, "core")
        stack = account_mtlnn(sh, n, "stack")
        assert stack["flops"] >= core["flops"]
    cots = [account_tfm(sh, n, "cot")["flops"] for n in (0, 1, 2, 4, 8)]
    assert cots == sorted(cots) and len(set(cots)) == len(cots)


def test_kv_peak_and_cot_direct_equivalence():
    """KV 峰值闭式 L·2·d·(T+N+1)·4；cot N=0 = prefill + 1 答案步。"""
    sh = dict(PROBE, vocab=26)
    row = account_tfm(sh, 8, "cot")
    assert row["kv_peak_bytes"] == 2 * 2 * 104 * (54 + 9) * 4
    direct = account_tfm(sh, 0, "cot")
    T, L, d, V = 54, 2, 104, 26
    blk = tfm_block_macs(sh)
    expect = ((T + 1) * L * blk["dense"]
              + L * blk["score"] * (T * (T + 1) / 2 + (T + 1))
              + 1 * d * V)
    assert direct["macs"] == expect


def test_invalid_depth_rejected():
    sh = dict(PROBE, vocab=26)
    with pytest.raises(ValueError):
        account_mtlnn(sh, 0, "core")
    with pytest.raises(ValueError):
        account_tfm(sh, -1, "cot")
