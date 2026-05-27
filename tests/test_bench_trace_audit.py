import json
from pathlib import Path

from scripts.bench_trace_audit import audit_one


def _make_trace(tmp_path: Path, rows):
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def test_audit_self_sufficiency_all_local(tmp_path):
    rows = [
        {"event": "token", "session_id": "s", "step": i, "route": "local", "entropy": 1.2}
        for i in range(10)
    ]
    p = _make_trace(tmp_path, rows)
    r = audit_one(p)
    assert r["total_tokens"] == 10
    assert r["self_sufficiency"] == 1.0
    assert r["routes"]["local"] == 10
    assert r["cloud_injects"] == 0


def test_audit_self_sufficiency_with_cloud(tmp_path):
    rows = [{"event": "token", "session_id": "s", "step": i, "route": "local", "entropy": 1.0}
            for i in range(8)]
    rows += [{"event": "token", "session_id": "s", "step": 9, "route": "cloud", "entropy": 5.0},
             {"event": "token", "session_id": "s", "step": 10, "route": "cloud", "entropy": 5.0}]
    rows += [{"event": "cloud_inject", "session_id": "s", "step": 9, "source": "mock", "query": "q", "fact_len": 100}]
    p = _make_trace(tmp_path, rows)
    r = audit_one(p)
    assert r["total_tokens"] == 10
    assert r["routes"]["cloud"] == 2
    assert r["self_sufficiency"] == 0.8
    assert r["cloud_injects"] == 1
    assert r["absorbed_bytes"] == 100
    assert r["estimated_cost_usd"]["net_savings"] > 0


def test_audit_handles_phi(tmp_path):
    rows = [
        {"event": "token", "session_id": "s", "step": 1, "route": "local", "entropy": 1.0, "phi": 0.1},
        {"event": "token", "session_id": "s", "step": 2, "route": "local", "entropy": 1.0, "phi": 0.3},
        {"event": "token", "session_id": "s", "step": 3, "route": "local", "entropy": 1.0},
    ]
    p = _make_trace(tmp_path, rows)
    r = audit_one(p)
    assert r["phi_hat"]["samples"] == 2
    assert r["phi_hat"]["mean"] == 0.2


def test_audit_empty_trace(tmp_path):
    p = _make_trace(tmp_path, [])
    r = audit_one(p)
    assert "error" in r
