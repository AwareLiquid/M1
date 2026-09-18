#!/usr/bin/env python3
"""B-2 tau-ladder probe on Kaggle CPU notebook — V15: full 4-core utilisation.

V14 -> V15 (B-2 提速诊断, backlog 2026-09-16):
- OMP/MKL threads pinned to 1 via environment BEFORE spawning workers:
  torch.set_num_threads(1) in the parent does NOT propagate to subprocesses,
  so V14 ran 3 children x 4 default OMP threads = 12 threads on 4 cores.
- 4 lanes instead of 3: the session is a hard 12h window, so lane count is
  the throughput lever (9 (depth,seed) pairs dealt round-robin = 3/2/2/2;
  V14's 3 lanes left the 4th core idle AND a lane-clock unused).
- Provenance: V14's launcher source was never committed (only lived as
  Kaggle version 14); this file was reconstructed from `kaggle kernels pull`
  of the exact V14 source + the changes above.

Bundle: private dataset aricredemption/m1-b1-text-ab-bundle (v6+: mt_lnn full
closure + benchmarks/{tau_ladder_probe,reasoning_depth,atomic_io,baselines,
experiment_protocol,reasoning_tasks}.py + wikitext2).
Output: /kaggle/working/results/ (tau_probe_*.json + tau_ladder_verdict.json),
resume-safe per config — truncated sessions keep their completed configs.
"""
import os, sys, glob, time, subprocess, shutil

# 0. thread pinning MUST precede worker spawn: children inherit the
#    environment, and 4 single-thread workers == 4 cores, no OMP thrash.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# 1. bundle 定位：kagglehub 优先，挂载扫描兜底
DS = None
try:
    import kagglehub
    DS = kagglehub.dataset_download("aricredemption/m1-b1-text-ab-bundle")
except Exception as e:
    print(f"[diag] kagglehub 不可用 ({type(e).__name__}), 回退挂载扫描", flush=True)
if not DS or not os.path.exists(os.path.join(DS or "", "benchmarks", "tau_ladder_probe.py")):
    for root, dirs, files in os.walk("/kaggle/input"):
        if os.path.exists(os.path.join(root, "benchmarks", "tau_ladder_probe.py")):
            DS = root; break
assert DS and os.path.exists(os.path.join(DS, "benchmarks", "tau_ladder_probe.py")), \
    f"bundle not found: {DS}"
print(f"[setup] bundle at {DS}", flush=True)

sys.path.insert(0, DS)
os.environ["PYTHONPATH"] = DS
import torch
torch.set_num_threads(1)   # parent does not train; belt-and-braces
os.chdir(DS)

# 2. 4 lanes over the (depth, seed) pair queue, dealt round-robin.
#    Each pair = one probe invocation (2 ladder modes x 1 seed x 1 depth).
#    Lanes are self-skipping (the probe resumes over completed configs),
#    so a truncated session only loses its in-flight pair.
WORK = "/kaggle/working"
os.makedirs(f"{WORK}/results", exist_ok=True)
# 2.5 多主机状态恢复：metadata.dataset_sources 挂载的 checkpoint 数据集里
#     若有历史配置 JSON，先并入 results —— probe 的 resume 跳过即跨会话/主机生效。
import glob as _glob
n_ckpt = 0
for root, _, files in os.walk("/kaggle/input"):
    for f in files:
        if f.startswith("tau_probe_") and f.endswith(".json"):
            dst = os.path.join(WORK, "results", f)
            if not os.path.exists(dst):
                shutil.copy(os.path.join(root, f), dst)
                n_ckpt += 1
print(f"[ckpt-restore] 并入 {n_ckpt} 个历史配置 JSON", flush=True)
pairs = [(d, s) for d in (1, 2, 4) for s in (0, 1, 2)]   # 9 pairs = 18 configs
LANES = 4
lanes = [pairs[i::LANES] for i in range(LANES)]           # 3/2/2/2
t0 = time.time()
procs = []
for i, lane in enumerate(lanes):
    if not lane:
        continue
    log = open(f"{WORK}/tau_lane{i+1}.log", "w")
    script = " && ".join(
        " ".join(
            "'%s'" % c if (" " in c or c.startswith("-")) else c for c in cmd)
        for cmd in (
            [sys.executable, "benchmarks/tau_ladder_probe.py",
             "--steps", "30000", "--gate-step", "10000",
             "--seeds", str(s), "--depths", str(d),
             "--device", "cpu",
             "--out-dir", f"{WORK}/results"]
            for d, s in lane))
    p = subprocess.Popen(["bash", "-c", script],
                         stdout=log, stderr=subprocess.STDOUT)
    procs.append((lane, p))
    print(f"[launch] lane{i+1} pairs={lane} pid={p.pid}", flush=True)

# 3. 等待 + 汇总
fails = []
for lane, p in procs:
    rc = p.wait()
    el = (time.time() - t0) / 60
    print(f"[finish] lane_pairs={lane} rc={rc} @{el:.0f}min", flush=True)
    if rc != 0:
        fails.append((lane, rc))

merged = f"{WORK}/tau_all_results"
os.makedirs(merged, exist_ok=True)
n = 0
for f in glob.glob(f"{WORK}/results/tau_probe_*.json") + \
         glob.glob(f"{WORK}/results/tau_ladder_verdict.json"):
    shutil.copy(f, os.path.join(merged, os.path.basename(f)))
    n += 1
print(f"[merge] {n} result files -> {merged}", flush=True)
print(f"[done] fails={fails} total {(time.time()-t0)/60:.0f}min", flush=True)
