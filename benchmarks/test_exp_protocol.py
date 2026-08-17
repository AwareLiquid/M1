"""E0 实验协议统计模块的单元测试。

运行：py -3.11 -m pytest benchmarks/test_exp_protocol.py -v
"""
import math

from benchmarks.exp_protocol import (
    fisher_exact_2x2,
    paired_sign_test,
    grok_rate,
    bimodality_check,
    protocol_report,
    aggregate_by_tag,
    load_jsonl_rows,
)


def test_fisher_known_strong_separation():
    # 6 sel 全 grok vs 6 stock 全 chance → p 极小
    p = fisher_exact_2x2(6, 0, 0, 6)
    assert p < 0.01, f"expected strong separation, got p={p}"


def test_fisher_known_null():
    # 2/6 vs 1/6 抛硬币差异 → p 接近 1.0
    p = fisher_exact_2x2(2, 4, 1, 5)
    assert p > 0.5, f"expected null, got p={p}"


def test_sign_test_symmetric_delta():
    p, n_pos, n_neg = paired_sign_test([0.1, -0.1, 0.2, -0.2])
    assert n_pos == 2 and n_neg == 2
    assert p > 0.9, f"symmetric deltas should be null, got p={p}"


def test_sign_test_all_positive():
    p, n_pos, n_neg = paired_sign_test([0.3, 0.4, 0.5, 0.2, 0.35, 0.45])
    assert (n_pos, n_neg) == (6, 0)
    assert p < 0.05, f"6/6 positive should be significant, got p={p}"


def test_sign_test_three_positive_not_significant():
    # 3/3 正差的二项双侧 p = 2 * 0.5^3 = 0.25，不显著
    p, n_pos, n_neg = paired_sign_test([0.3, 0.4, 0.5])
    assert (n_pos, n_neg) == (3, 0)
    assert abs(p - 0.25) < 1e-9, f"3/3 two-sided binom p should be 0.25, got {p}"


def test_grok_rate():
    assert grok_rate([1.0, 0.5, 0.99, 0.49]) == 0.5


def test_bimodality_clean():
    r = bimodality_check([0.50, 0.51, 0.49])
    assert r["bimodal"] is False
    assert r["verdict"] == "clean"


def test_bimodality_mixed_zone():
    # 典型 grokking 双峰：chance 峰 + grok 峰并存
    r = bimodality_check([0.50, 0.51, 0.49, 1.0, 1.0, 0.99])
    assert r["bimodal"] is True
    assert r["n_chance"] == 3 and r["n_grok"] == 3


def test_bimodality_mid_not_bimodal():
    # 中间灰区占主导 = 训练中途，不是双峰
    r = bimodality_check([0.50, 0.70, 0.99])
    assert r["bimodal"] is False


def test_protocol_report_separated():
    # sel 6 个全 grok，stock 6 个全 chance → fisher p≈0.002
    rep = protocol_report([1.0] * 6, [0.50, 0.51, 0.49, 0.50, 0.52, 0.48])
    assert rep["verdict"] == "SEPARATED"
    assert rep["decision_gate_passed"] is True
    assert rep["fisher_p"] < 0.05


def test_protocol_report_bimodal_real_pvd2():
    # 真实 pv-d2 数据：sel 2/6 grok vs stock 1/6 → 双峰区不可信 + fisher≈1.0
    sel = [1.0, 1.0, 0.513, 0.513, 0.511, 0.487]
    stock = [1.0, 0.513, 0.513, 0.503, 0.497, 0.487]
    rep = protocol_report(sel, stock)
    assert rep["verdict"] == "BIMODAL_ZONE_UNRELIABLE"
    assert rep["decision_gate_passed"] is False
    assert rep["fisher_p"] > 0.5


def test_aggregate_by_tag_nested_dict():
    rows = [
        {"tag": "sel", "seed": 0, "mtlnn_acc_by_depth": {"1": {"1": 1.0, "2": 1.0}}},
        {"tag": "sel", "seed": 1, "mtlnn_acc_by_depth": {"1": {"1": 0.5, "2": 0.5}}},
        {"tag": "stock", "seed": 0, "mtlnn_acc_by_depth": {"1": 0.5}},
        {"tag": "stock", "seed": 1, "mtlnn_acc_by_depth": {"1": 0.6}},
    ]
    sel, stock = aggregate_by_tag(rows, "sel", "stock")
    assert sel == [1.0, 0.5]
    assert stock == [0.5, 0.6]


def test_aggregate_by_tag_flat_dict():
    rows = [
        {"tag": "sel", "seed": 0, "mtlnn_acc_by_depth": {"1": 0.99}},
        {"tag": "stock", "seed": 0, "mtlnn_acc_by_depth": {"1": 0.50}},
    ]
    sel, stock = aggregate_by_tag(rows, "sel", "stock")
    assert sel == [0.99]
    assert stock == [0.50]
