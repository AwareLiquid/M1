#!/usr/bin/env python3
"""
scripts/validate_results.py — 基准结果 JSON 结构校验

每个已合入 PR 的基准结果 JSON 都应有可断言的结构（种子数、运行数、字段完整
性）。本脚本在 CI 的 audit-results job 中自动执行：结构不符 → 退出码非 0 →
CI 红灯 → 阻止合入。

只校验结构完整性，不硬编码实验数字（如 nRMSE 值），因为具体数字会随实验变化。
结构（如"每 config × 3 seeds"、"240 runs = 2×3×4×10"）是契约，不应变化。

Usage:
    python scripts/validate_results.py              # 校验全部
    python scripts/validate_results.py -v           # 详细输出
    python scripts/validate_results.py --file X.json  # 只校验单个
"""

import json
import os
import sys

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "results")
RESULTS_DIR = os.path.normpath(RESULTS_DIR)

# ── 校验器：每个 = 一个 (文件名, 校验函数) ────────────────────────────────

_validators = []


def validator(name):
    """Decorator to register a validator."""

    def deco(fn):
        _validators.append((name, fn))
        return fn
    return deco


def load(filename):
    """Load a JSON file from the results directory; exit on failure."""
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"  FAIL  {filename}: file not found")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


# ── 事件流 θ 扫描 ────────────────────────────────────────────────────────

@validator("event_theta_sweep.json")
def check_event_theta_sweep():
    d = load("event_theta_sweep.json")
    runs = d.get("runs", {})
    n = len(runs)
    # 解析键结构: task|theta|arch|seed
    tasks, thetas, archs, seeds = set(), set(), set(), set()
    for k in runs:
        parts = k.split("|")
        if len(parts) != 4:
            print(f"  FAIL  bad key format: {k}")
            return False
        tasks.add(parts[0])
        thetas.add(parts[1])
        archs.add(parts[2])
        seeds.add(parts[3])
    expected = len(tasks) * len(thetas) * len(archs) * len(seeds)
    if n != expected:
        print(f"  FAIL  expected {expected} runs ({len(tasks)}t×{len(thetas)}θ"
              f"×{len(archs)}a×{len(seeds)}s), got {n}")
        return False
    if len(seeds) < 10:
        print(f"  FAIL  expected ≥10 seeds, got {len(seeds)}")
        return False
    print(f"  PASS  {n} runs = {len(tasks)}t×{len(thetas)}θ×{len(archs)}a"
          f"×{len(seeds)}s, config={d.get('config', {})}")
    return True


# ── 事件流 Δt 跨度 ────────────────────────────────────────────────────────

@validator("event_dt_span.json")
def check_event_dt_span():
    d = load("event_dt_span.json")
    runs = d.get("runs", {})
    n = len(runs)
    tasks, spans, archs, seeds = set(), set(), set(), set()
    for k in runs:
        parts = k.split("|")
        if len(parts) != 4:
            print(f"  FAIL  bad key format: {k}")
            return False
        tasks.add(parts[0])
        spans.add(parts[1])
        archs.add(parts[2])
        seeds.add(parts[3])
    expected = len(tasks) * len(spans) * len(archs) * len(seeds)
    if n != expected:
        print(f"  FAIL  expected {expected} runs, got {n}")
        return False
    if len(seeds) < 10:
        print(f"  FAIL  expected ≥10 seeds, got {len(seeds)}")
        return False
    print(f"  PASS  {n} runs = {len(tasks)}t×{len(spans)}s×{len(archs)}a"
          f"×{len(seeds)}s")
    return True


# ── N-Caltech101 ──────────────────────────────────────────────────────────

@validator("event_ncaltech101.json")
def check_event_ncaltech101():
    d = load("event_ncaltech101.json")
    status = d.get("status", "")
    if status != "OK":
        print(f"  FAIL  expected status=OK, got {status!r}")
        return False
    # 应含 gru_d pending-merge 标记
    gd = d.get("gru_d", "")
    if "pending merge" not in gd and "pending" not in gd:
        print(f"  FAIL  expected gru_d pending-merge marker, got {gd!r}")
        return False
    print(f"  PASS  status=OK, gru_d={gd!r}")
    return True


# ── Parametric memory summary ─────────────────────────────────────────────

@validator("parametric_memory_summary.json")
def check_parametric_memory_summary():
    d = load("parametric_memory_summary.json")
    lms = d.get("lm_configs", {})
    if not lms:
        print("  FAIL  no lm_configs key")
        return False
    for cfg_name, cfg in lms.items():
        for dim in ("accurate_retrieval", "test_time_learning",
                     "conflict_resolution", "cross_session_long_range"):
            if dim not in cfg:
                print(f"  FAIL  {cfg_name} missing dimension {dim}")
                return False
            stats = cfg[dim]
            if stats.get("n", 0) < 3:
                print(f"  FAIL  {cfg_name}/{dim}: expected ≥3 seeds, "
                      f"got {stats.get('n')}")
                return False
            if "mean" not in stats or "std" not in stats:
                print(f"  FAIL  {cfg_name}/{dim}: missing mean/std")
                return False
    print(f"  PASS  {len(lms)} configs × ≥3 seeds × 4 dimensions")
    return True


# ── KV frontier ledger ────────────────────────────────────────────────────

@validator("kv_frontier_ledger.json")
def check_kv_frontier_ledger():
    d = load("kv_frontier_ledger.json")
    if "ledger" not in d or "frontier" not in d:
        print("  FAIL  missing ledger or frontier key")
        return False
    if not d.get("ledger"):
        print("  FAIL  empty ledger")
        return False
    print(f"  PASS  ledger ({len(d['ledger'])} entries) + frontier present")
    return True


# ── KV measured ───────────────────────────────────────────────────────────

@validator("kv_measured.json")
def check_kv_measured():
    d = load("kv_measured.json")
    models = d.get("models", [])
    if not models:
        print("  FAIL  no models in kv_measured.json")
        return False
    # 检查每行有 fp16_dev_pct（0.0% 偏差校验）
    for m in models:
        for row in m.get("rows", []):
            dev = row.get("fp16_dev_pct")
            if dev is None:
                print(f"  FAIL  missing fp16_dev_pct in {m.get('model')}")
                return False
    print(f"  PASS  {len(models)} model(s) × {sum(len(m.get('rows',[])) for m in models)} rows, fp16_dev_pct present")
    return True


# ── Pscan bench ───────────────────────────────────────────────────────────

@validator("pscan_bench_const.json")
def check_pscan_bench_const():
    d = load("pscan_bench_const.json")
    meta = d.get("meta", {})
    results = d.get("results", [])
    if not meta or not results:
        print("  FAIL  missing meta or results")
        return False
    impls = set(r.get("impl") for r in results if r.get("mode") == "fwd_bwd")
    expected_impls = {"pscan_const", "chunk_const"}
    if not expected_impls.issubset(impls):
        print(f"  FAIL  expected impls {expected_impls}, got {impls}")
        return False
    print(f"  PASS  {len(results)} rows, impls={impls}")
    return True


# ── Pscan bench general ──────────────────────────────────────────────────

@validator("pscan_bench.json")
def check_pscan_bench_general():
    path = os.path.join(RESULTS_DIR, "pscan_bench.json")
    if not os.path.exists(path):
        # 通用版可能按需生成，不强制
        print("  SKIP  pscan_bench.json not found (optional)")
        return True
    d = load("pscan_bench.json")
    meta = d.get("meta", {})
    results = d.get("results", [])
    if not meta or not results:
        print("  FAIL  missing meta or results")
        return False
    print(f"  PASS  {len(results)} rows")
    return True


# ── 主入口 ────────────────────────────────────────────────────────────────

def main():
    verbose = "-v" in sys.argv or "--verbose" in sys.argv

    # 可选：指定单个文件
    single = None
    for i, arg in enumerate(sys.argv[1:]):
        if arg.startswith("--file="):
            single = arg.split("=", 1)[1]
        elif arg == "--file" and i + 2 < len(sys.argv):
            single = sys.argv[i + 2]

    if single:
        matched = [(n, fn) for n, fn in _validators if n == single]
        if not matched:
            print(f"Unknown file: {single}")
            sys.exit(1)
        _validators[:] = matched

    passed = 0
    failed = 0
    skipped = 0

    print(f"Validating benchmark results in {RESULTS_DIR}")
    print()

    for name, fn in _validators:
        if verbose:
            print(f"  [{name}]")
        try:
            result = fn()
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            result = False
        if result is True:
            passed += 1
        elif result is None:
            skipped += 1
        else:
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    if failed:
        print("FAIL: 1 or more structural validations failed")
        sys.exit(1)
    print("PASS: all structural validations passed")


if __name__ == "__main__":
    main()