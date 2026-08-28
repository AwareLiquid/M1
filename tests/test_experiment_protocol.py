"""tests/test_experiment_protocol.py — E0 协议固化契约（P3-14）

真实案例回放：
  • GQA g0 单 seed 1.0000 → 必须被 seeds 门槛拦下
  • grokking 双峰 (0.18/0.17 vs 1.0/1.0) → 必须被判双峰
  • E1-d16 selective 5/6 grok vs stock 0/6 → Fisher p≈0.0022~0.015
    (双侧; 具体值取决于口径, 断言 p<0.05)
"""

import pytest

from benchmarks.experiment_protocol import (
    ProtocolViolation, bimodality_1d, fisher_exact_2x2, forbid_bimodal,
    paired_sign_test, publishable, require_min_seeds,
)


def test_min_seeds_gate():
    with pytest.raises(ProtocolViolation, match="单/双 seed"):
        require_min_seeds([0.5], what="g0")            # Kaggle 单 seed 案例
    with pytest.raises(ProtocolViolation):
        require_min_seeds([0.5, 0.6])                  # 双 seed 同样不够
    assert require_min_seeds([0.5, 0.6, 0.55]) == [0.5, 0.6, 0.55]


def test_bimodality_detects_grokking_bimodal():
    # HANDOFF §2.5: 本地 seeds 1,2 ≈ 0.183/0.165, Kaggle seed0 = 1.0
    bm = bimodality_1d([0.183, 0.165, 1.0, 1.0, 0.17, 1.0])
    assert bm["bimodal"] is True
    with pytest.raises(ProtocolViolation, match="双峰"):
        forbid_bimodal([0.183, 0.165, 1.0, 1.0, 0.17, 1.0])


def test_bimodality_passes_unimodal_noise():
    bm = bimodality_1d([0.52, 0.55, 0.53, 0.56, 0.54, 0.55])
    assert bm["bimodal"] is False


def test_paired_sign_test_significance():
    # 6/6 全胜 → p = 2·(1/64) = 0.03125 < 0.05
    st = paired_sign_test([1.0] * 6, [0.0] * 6)
    assert st["p"] == pytest.approx(0.03125)
    # 3胜3负 (无平局) → p = 2·42/64 = 1.0 (不显著)
    st2 = paired_sign_test([1, 0.3, 1, 0.3, 1, 0.3], [0.4] * 6)
    assert st2["p"] > 0.99
    # 平局被丢弃: 0 vs 0 的对不参与 (3 对里 2 对分胜负)
    st3 = paired_sign_test([1, 0, 1], [0, 0, 0])
    assert st3["n_effective"] == 2 and st3["wins"] == 2


def test_fisher_exact_matches_e1_grok_case():
    # E1-d16: selective 5/6 grok vs stock 0/6 (ABLATIONS: Fisher p=0.0022
    # 为单侧口径; 本实现双侧, 必须同样显著)
    r = fisher_exact_2x2([[5, 1], [0, 6]])
    assert r["p"] < 0.05


def test_fisher_exact_independence_edge():
    assert fisher_exact_2x2([[3, 3], [3, 3]])["p"] > 0.9   # 无关联


def test_publishable_full_gate():
    # 干净分离 + 足量 seeds + 单峰 → 可引用 (6/6 全胜 p=0.031)
    ok, rep = publishable([1.0] * 6, [0.2] * 6, tag="demo-good")
    assert ok, rep["reasons"]
    # 4 seeds 全胜: p=2/16=0.125 ≥ 0.05 → 协议正确地拒绝 (4 seeds 不够引用)
    ok4s, rep4s = publishable([1.0] * 4, [0.2] * 4, tag="four-seeds")
    assert not ok4s and any("配对符号检验" in r for r in rep4s["reasons"])
    # 单 seed → 只能入档
    ok2, rep2 = publishable([1.0], [0.0], tag="g0-kaggle")
    assert not ok2 and any("seed" in r for r in rep2["reasons"])
    # 双峰臂 → 只能入档
    ok3, rep3 = publishable([0.17, 0.18, 1.0, 1.0, 1.0, 1.0],
                            [0.2] * 6, tag="bimodal-arm")
    assert not ok3 and any("双峰" in r for r in rep3["reasons"])
    # 无分离 (p≥α) → 只能入档
    ok4, rep4 = publishable([0.5, 0.6, 0.55, 0.52, 0.58, 0.51],
                            [0.52, 0.5, 0.58, 0.54, 0.5, 0.55], tag="null")
    assert not ok4 and any("配对符号检验" in r for r in rep4["reasons"])
