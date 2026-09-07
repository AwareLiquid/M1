#!/usr/bin/env python3
"""benchmarks/h004_base.py — H004 Bet A 基座核销层（机制轨，零主张）

Bet A（docs/RESEARCH_PLAN.md §2.1）的基座接入在 GPU 会话**之前**完成纸面核销：
候选基座 registry、license 白名单、上下文长度门槛、核验报告落档（config 回显 +
git 哈希 + 时间戳，仓内结果纪律）。本模块不下载权重、不做训练、不出实验数字。

- 主基座: nvidia/Llama-3.1-Nemotron-UltraLong（1M 档开源权重，跳过自扩；
  2026-09-06 修订二采纳，见 docs/RESEARCH_PLAN.md §4）
- 备选:   Qwen3-4B-Base + LongRoPE（主基座许可/适配不顺时切换）
- 在线核销（--online）只请求 HF API 元数据（存在性/license），绝不下载权重；
  默认离线（CI）只做静态检查——HF id 的最终确认在开题日完成并回填本文件。

Usage:
    python benchmarks/h004_base.py --list
    python benchmarks/h004_base.py --verify ultralong-1m [--config-json cfg.json] [--online]
    python benchmarks/h004_base.py --verify all
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

SCHEMA = "h004_base_check/1"
RESULTS_DIR = "benchmarks/results"


@dataclasses.dataclass(frozen=True)
class BaseSpec:
    key: str
    hf_id: str
    role: str                     # primary | fallback
    min_ctx_tokens: int | None    # 上下文长度门槛；None = 需外挂扩展（fallback 路线）
    license_allow: tuple[str, ...]
    notes: str = ""


BASES: dict[str, BaseSpec] = {
    # hf_id 以开题日 --online 核销为准；此处 id 来自 2026-09-06 调研
    # （docs/RESEARCH_PLAN.md §0，arXiv:2504.06214 / ultralong.github.io）。
    "ultralong-1m": BaseSpec(
        key="ultralong-1m",
        hf_id="nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Base",
        role="primary",
        min_ctx_tokens=1_000_000,
        license_allow=("other", "apache-2.0", "llama3.1", "llama3.2", "nvidia-open-model-license"),
        notes="主基座：开源 1M 权重，跳过自扩；两阶段 CPT+iT 配方可追溯（ACL 2026）。",
    ),
    "ultralong-1m-instruct": BaseSpec(
        key="ultralong-1m-instruct",
        hf_id="nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct",
        role="primary",
        min_ctx_tokens=1_000_000,
        license_allow=("other", "apache-2.0", "llama3.1", "llama3.2", "nvidia-open-model-license"),
        notes="主基座 instruct 变体；若走 API 式问答评测优先核销此档。",
    ),
    "qwen3-4b-longrope": BaseSpec(
        key="qwen3-4b-longrope",
        hf_id="Qwen/Qwen3-4B-Base",
        role="fallback",
        min_ctx_tokens=None,  # 需 LongRoPE 外挂扩展后达 1M
        license_allow=("apache-2.0",),
        notes="备选：主基座许可/适配不顺时切换；扩展成本 ≈ +1-2 A100 日。",
    ),
}


def git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def check_online(spec: BaseSpec, timeout: int = 15) -> dict:
    """HF API 元数据核销（存在性 + license tag）。离线/失败 → status=unknown。"""
    url = f"https://huggingface.co/api/models/{spec.hf_id}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            meta = json.load(resp)
        tags = [t.split(":", 1)[-1] for t in meta.get("cardData", {}).get("license", [])]
        return {"status": "pass", "exists": True, "license_tags": tags}
    except Exception as exc:  # noqa: BLE001 — 离线/网络/限流一律 unknown，不猜
        return {"status": "unknown", "exists": None, "error": f"{type(exc).__name__}: {exc}"}


def verify_base(spec: BaseSpec, config: dict | None = None, online: bool = False) -> dict:
    """核销一个基座。config 为 HF config 的关键字段（可由 --config-json 提供），
    缺失的检查记 unknown（不失败，但报告里可见）。all_pass = 无 fail。"""
    checks: dict[str, dict] = {}

    checks["registered"] = {"status": "pass", "detail": f"role={spec.role}"}
    if spec.role == "primary" and spec.min_ctx_tokens is None:
        checks["registered"] = {"status": "fail", "detail": "primary 基座必须设 min_ctx_tokens"}

    if online:
        checks["hf_online"] = check_online(spec)
        lic_tags = checks["hf_online"].get("license_tags") or []
        if lic_tags:
            ok = any(t.lower() in spec.license_allow for t in lic_tags)
            checks["license"] = {"status": "pass" if ok else "fail", "tags": lic_tags}
        else:
            checks["license"] = {"status": "unknown", "detail": "HF 元数据未含 license tag"}
    else:
        cfg_lic = (config or {}).get("license")
        if cfg_lic:
            ok = str(cfg_lic).lower() in spec.license_allow
            checks["license"] = {"status": "pass" if ok else "fail", "tags": [cfg_lic]}
        else:
            checks["license"] = {"status": "unknown", "detail": "离线且未提供 --config-json"}

    cfg_ctx = (config or {}).get("max_position_embeddings")
    if cfg_ctx is not None:
        if spec.min_ctx_tokens is None:
            checks["ctx"] = {"status": "unknown", "detail": "fallback 基座：ctx 由外挂扩展达成"}
        else:
            ok = int(cfg_ctx) >= spec.min_ctx_tokens
            checks["ctx"] = {"status": "pass" if ok else "fail",
                             "ctx": int(cfg_ctx), "min": spec.min_ctx_tokens}
    else:
        checks["ctx"] = {"status": "unknown", "detail": "未提供 config"}

    failed = [k for k, v in checks.items() if v.get("status") == "fail"]
    report = {
        "schema": SCHEMA,
        "base_key": spec.key,
        "hf_id": spec.hf_id,
        "role": spec.role,
        "online": online,
        "checks": checks,
        "all_pass": not failed,
        "failed_checks": failed,
        "config_echo": config or {},
        "git_hash": git_hash(),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    return report


def verify_all(config_by_key: dict[str, dict] | None = None, online: bool = False) -> dict:
    reports = {k: verify_base(s, (config_by_key or {}).get(k), online)
               for k, s in BASES.items()}
    return {
        "schema": SCHEMA, "kind": "verify_all", "reports": reports,
        "all_pass": all(r["all_pass"] for r in reports.values()),
        "git_hash": git_hash(),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="H004 基座核销（零主张，机制轨）")
    p.add_argument("--list", action="store_true")
    p.add_argument("--verify", metavar="KEY|all")
    p.add_argument("--config-json", help="HF config 关键字段 JSON（license/max_position_embeddings）")
    p.add_argument("--online", action="store_true", help="请求 HF API 元数据（不下载权重）")
    p.add_argument("--out", default=os.path.join(RESULTS_DIR, "h004_base_check.json"))
    args = p.parse_args(argv)

    if args.list:
        for k, s in BASES.items():
            print(f"{k:24s} role={s.role:8s} {s.hf_id}  min_ctx={s.min_ctx_tokens}")
        return 0

    if not args.verify:
        p.print_help()
        return 2

    config = None
    if args.config_json:
        with open(args.config_json) as f:
            config = json.load(f)

    if args.verify == "all":
        report = verify_all(None, args.online)
        if config:  # 单一 config 无法对号多个基座：写进报告备注，静态检查照常
            report["note"] = "config-json 提供时请用 --verify KEY 单基座核销"
    else:
        if args.verify not in BASES:
            print(f"unknown base key: {args.verify}", file=sys.stderr)
            return 2
        report = verify_base(BASES[args.verify], config, args.online)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
