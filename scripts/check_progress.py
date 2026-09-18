#!/usr/bin/env python3
"""check_progress.py — Kaggle kernel 进度追踪

拉取指定 kernel 的 output，统计已完成配置数。
用法: python scripts/check_progress.py <kernel_slug> [expected_total]
输出: JSON 格式进度报告
"""
import json, sys, os, subprocess, glob, tempfile, shutil

KERNEL_SLUG = sys.argv[1] if len(sys.argv) > 1 else "aricredemption/m1-b2-tau-ladder-cpu"
EXPECTED = int(sys.argv[2]) if len(sys.argv) > 2 else 18
KAGGLE = os.path.join(os.path.dirname(__file__), "..", ".venv", "bin", "kaggle")

# 拉取到临时目录
tmp = tempfile.mkdtemp(prefix="pmb_progress_")
r = subprocess.run([KAGGLE, "kernels", "output", KERNEL_SLUG, "-p", tmp, "--force"],
                   capture_output=True, text=True, timeout=300)

probes = sorted(glob.glob(os.path.join(tmp, "results", "tau_probe_*.json")))
done = []
for f in probes:
    d = json.load(open(f))
    done.append({"ladder": d.get("ladder"), "depth": d.get("depth"),
                 "seed": d.get("seed"), "mean_acc": d.get("mean_acc")})

verdict_path = os.path.join(tmp, "results", "tau_ladder_verdict.json")
verdict = json.load(open(verdict_path)) if os.path.exists(verdict_path) else None

# logs
depth_logs = sorted(glob.glob(os.path.join(tmp, "tau_depth*.log")))
log_summary = []
for lf in depth_logs:
    lines = open(lf).read().strip().split("\n")
    done_runs = [l for l in lines if "done mean_acc" in l]
    log_summary.append({"log": os.path.basename(lf), "completed": len(done_runs)})

report = {
    "kernel": KERNEL_SLUG,
    "status": "UNKNOWN (run kaggle kernels status separately)",
    "configs_done": len(done),
    "configs_expected": EXPECTED,
    "progress_pct": round(len(done) / EXPECTED * 100, 1) if EXPECTED else 0,
    "verdict_available": verdict is not None,
    "verdict_h_supported": verdict.get("h_supported") if verdict else None,
    "configs": done,
    "session_logs": log_summary,
}
print(json.dumps(report, indent=1, ensure_ascii=False))

# 清理
shutil.rmtree(tmp, ignore_errors=True)
