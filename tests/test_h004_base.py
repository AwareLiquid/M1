"""tests/test_h004_base.py — 基座核销层结构测试（CPU，离线，零主张）"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmarks.h004_base as hb  # noqa: E402


def test_registry_integrity():
    assert "ultralong-1m" in hb.BASES, "主基座缺失（2026-09-06 修订二）"
    roles = {s.role for s in hb.BASES.values()}
    assert roles <= {"primary", "fallback"}, f"未知 role: {roles}"
    for s in hb.BASES.values():
        if s.role == "primary":
            assert s.min_ctx_tokens is not None, "primary 基座必须设上下文门槛"
            assert s.min_ctx_tokens >= 1_000_000, "H004 主张是 1M 流式"
        assert s.license_allow, "license 白名单不能为空"


def test_verify_offline_report_structure():
    r = hb.verify_base(hb.BASES["ultralong-1m"], config=None, online=False)
    assert r["schema"] == hb.SCHEMA
    assert r["base_key"] == "ultralong-1m"
    # 离线无 config：允许 unknown，但不允许 fail
    assert r["all_pass"] is True, f"离线静态核销不应 fail: {r['failed_checks']}"
    assert set(r["failed_checks"]) == set()
    assert len(r["git_hash"]) in (40, 7) or r["git_hash"] == "unknown"
    assert "timestamp_utc" in r
    for name in ("license", "ctx"):
        assert name in r["checks"]
        assert r["checks"][name]["status"] in ("pass", "unknown", "fail")


def test_verify_ctx_gate_fails():
    cfg = {"max_position_embeddings": 131072}  # 远低于 1M 门槛
    r = hb.verify_base(hb.BASES["ultralong-1m"], config=cfg, online=False)
    assert r["checks"]["ctx"]["status"] == "fail"
    assert r["all_pass"] is False
    assert "ctx" in r["failed_checks"]


def test_verify_ctx_gate_passes():
    cfg = {"max_position_embeddings": 1_000_000}
    r = hb.verify_base(hb.BASES["ultralong-1m"], config=cfg, online=False)
    assert r["checks"]["ctx"]["status"] == "pass"


def test_verify_license_gate_fails():
    cfg = {"license": "gpl-3.0"}  # 不在白名单
    r = hb.verify_base(hb.BASES["ultralong-1m"], config=cfg, online=False)
    assert r["checks"]["license"]["status"] == "fail"
    assert r["all_pass"] is False


def test_verify_license_passes():
    cfg = {"license": "apache-2.0"}
    r = hb.verify_base(hb.BASES["ultralong-1m"], config=cfg, online=False)
    assert r["checks"]["license"]["status"] == "pass"


def test_fallback_ctx_is_unknown_not_fail():
    cfg = {"max_position_embeddings": 32768}
    r = hb.verify_base(hb.BASES["qwen3-4b-longrope"], config=cfg, online=False)
    assert r["checks"]["ctx"]["status"] == "unknown", "fallback 基座 ctx 由外挂扩展达成，不判 fail"
    assert r["all_pass"] is True


def test_verify_all_shape():
    r = hb.verify_all(None, online=False)
    assert set(r["reports"].keys()) == set(hb.BASES.keys())
    assert r["all_pass"] == all(x["all_pass"] for x in r["reports"].values())
