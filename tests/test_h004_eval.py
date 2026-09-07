"""tests/test_h004_eval.py — 评测与记账脚手架测试（CPU，零主张）"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmarks.h004_eval as he  # noqa: E402
import pytest  # noqa: E402


# ── CostLedger ──────────────────────────────────────────────────────────────

def test_ledger_math():
    led = he.CostLedger()
    led.add("A", 7, 70, "POC")
    led.add("B", 21.5, 70, "main")
    t = led.total()
    assert t["gpu_hours"] == 28.5
    assert t["cny"] == pytest.approx((7 + 21.5) * 70)
    assert t["n_entries"] == 2


def test_ledger_rejects_negative():
    led = he.CostLedger()
    with pytest.raises(ValueError):
        led.add("x", -1, 70)
    with pytest.raises(ValueError):
        led.add("x", 1, -70)


def test_ledger_report_structure(tmp_path):
    led = he.CostLedger()
    led.add("A", 1, 70)
    path = tmp_path / "ledger.json"
    r = led.dump(str(path))
    assert r["schema"] == he.SCHEMA_LEDGER
    doc = json.loads(path.read_text())
    assert doc["total"]["cny"] == 70
    assert "git_hash" in doc and "timestamp_utc" in doc


# ── CSE 权重 ────────────────────────────────────────────────────────────────

def test_cse_uniform_stays_uniform():
    w = he.cse_weights({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25})
    assert all(v == pytest.approx(0.25) for v in w.values())


def test_cse_inverse_relation():
    w = he.cse_weights({"common": 0.9, "rare": 0.1})
    assert w["rare"] > w["common"], "过采样桶必须被降权"
    assert sum(w.values()) == pytest.approx(1.0)


def test_cse_zero_freq_excluded_and_allzero_raises():
    w = he.cse_weights({"a": 0.5, "ghost": 0.0})
    assert "ghost" not in w
    with pytest.raises(ValueError):
        he.cse_weights({"a": 0.0, "b": 0.0})


# ── 回放任务生成器 ──────────────────────────────────────────────────────────

def test_replay_deterministic():
    t1 = he.build_replay(42, n_sessions=3)
    t2 = he.build_replay(42, n_sessions=3)
    assert t1["sessions"] == t2["sessions"]
    assert t1["probes"] == t2["probes"]
    t3 = he.build_replay(43, n_sessions=3)
    assert t3["probes"] != t1["probes"] or t3["sessions"] != t1["sessions"]


def test_replay_integrity_ground_truth():
    task = he.build_replay(7, n_sessions=4, n_probes=8)
    chk = he.check_replay_integrity(task)
    assert chk["ground_truth_ok"], f"bad probes: {chk['bad_probes']}"
    assert chk["all_probes_cross_session"], "探针必须跨会话（构造保证）"
    assert chk["probes_total"] == 8


def test_replay_requires_two_sessions():
    with pytest.raises(ValueError):
        he.build_replay(0, n_sessions=1)


def test_replay_shapes():
    task = he.build_replay(1, n_sessions=3, facts_per_session=5, n_probes=6)
    assert len(task["sessions"]) == 3
    for s in task["sessions"]:
        assert len(s["facts"]) == 5
        assert len(s["fillers"]) == 20
    assert len(task["probes"]) == 6


# ── 预注册落档 ──────────────────────────────────────────────────────────────

def test_prereg_structure(tmp_path):
    rule = {"c1_ruler_ratio": 0.9, "c2_state_half_of_kv64k": True,
            "c4_min_citable_cells": 2}
    path = tmp_path / "h004_prereg.json"
    doc = he.write_prereg(str(path), rule)
    assert doc["schema"] == he.SCHEMA_PREREG
    assert doc["pre_registered_rule"] == rule
    on_disk = json.loads(path.read_text())
    assert on_disk["pre_registered_rule"] == rule, "落档后可复核（判据写死）"
    assert "git_hash" in on_disk and "timestamp_utc" in on_disk
