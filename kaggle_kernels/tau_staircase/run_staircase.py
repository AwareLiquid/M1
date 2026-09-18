#!/usr/bin/env python3
"""B-2' staircase 校准 kernel（T4 GPU）— 定位 parity k_bits 的深度敏感带.

扫描：k ∈ {32, 64, 128} × depth ∈ {1, 4}（ladder off+on 都跑，seed 0）。
每个配置训练 20K 步，10K 步 gate 读数 = staircase 的"学不会/学得会"判据。
进度日志约定（人看友好）：
  [cfg i/N] START  k=.. d=.. —— 配置开始
  step/loss 行                          —— 训练循环（探针自带，每 2500 步）
  [cfg i/6] GATE ... / DONE ... wall=..  —— 探针自带 + 汇总行
  [eta] 已完成 x/6，均 wall ..min，预计剩余 ..min
Resume-safe：结果 JSON 按已存在跳过（文件名含 k 后缀，不同 k 不冲突）。
"""
import glob, json, os, shutil, subprocess, sys, time

KS = [32, 64, 128]
DEPTHS = [1, 4]
SEED = 0
STEPS = 20000
GATE = 10000
LANE_CFGS = [(k, d) for k in KS for d in DEPTHS]          # 6 个 (k, d) 对

WORK = "/kaggle/working"
t0 = time.time()

def ts():
    return time.strftime("%H:%M:%S")

def stamp():
    el = (time.time() - t0) / 60
    return f"[{ts()} @{el:.0f}min]"

# bundle 定位（kagglehub 优先，挂载扫描兜底 —— 与 B-2 kernel 同款）
DS = None
try:
    import kagglehub
    DS = kagglehub.dataset_download("aricredemption/m1-b1-text-ab-bundle")
except Exception as e:
    print(f"[diag] kagglehub 不可用 ({type(e).__name__})", flush=True)
if not DS or not os.path.exists(os.path.join(DS or "", "benchmarks", "tau_ladder_probe.py")):
    for root, dirs, files in os.walk("/kaggle/input"):
        if os.path.exists(os.path.join(root, "benchmarks", "tau_ladder_probe.py")):
            DS = root; break
assert DS and os.path.exists(os.path.join(DS, "benchmarks", "tau_ladder_probe.py")), \
    f"bundle not found: {DS}"
print(f"{stamp()} [setup] bundle at {DS}", flush=True)
sys.path.insert(0, DS)
os.environ["PYTHONPATH"] = DS
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.chdir(DS)

import torch
print(f"{stamp()} [env] torch={torch.__version__} cuda={torch.cuda.is_available()} "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)

# pending 判定：results_k{k}/ 下同名 JSON 存在即跳过（L010 resume-safe）
def done(k, d):
    return os.path.exists(f"{WORK}/results_k{k}/tau_probe_ladderon_d{d}_s{SEED}.json") and \
           os.path.exists(f"{WORK}/results_k{k}/tau_probe_ladderoff_d{d}_s{SEED}.json")

todo = [(k, d) for (k, d) in LANE_CFGS if not done(k, d)]
print(f"{stamp()} [plan] 待跑 {len(todo)}/{len(LANE_CFGS)} 个 (k,d) 对: {todo}", flush=True)

walls = []
for i, (k, d) in enumerate(todo, 1):
    out_dir = f"{WORK}/results_k{k}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"{stamp()} [cfg {i}/{len(todo)}] START k={k} d={d} "
          f"(2 ladder 模式 × {STEPS} 步, seed={SEED})", flush=True)
    tc = time.time()
    p = subprocess.run(
        [sys.executable, "benchmarks/tau_ladder_probe.py",
         "--steps", str(STEPS), "--gate-step", str(GATE),
         "--seeds", str(SEED), "--depths", str(d),
         "--difficulty", str(k),
         "--device", "cuda",
         "--out-dir", out_dir],
        capture_output=False)
    wall = (time.time() - tc) / 60
    walls.append(wall)
    ok = p.returncode == 0 and done(k, d)
    print(f"{stamp()} [cfg {i}/{len(todo)}] {'DONE' if ok else 'FAIL'} "
          f"k={k} d={d} rc={p.returncode} wall={wall:.1f}min", flush=True)
    # eta：用已完成均值估算剩余
    if walls:
        avg = sum(walls) / len(walls)
        remain = (len(todo) - i) * avg
        print(f"{stamp()} [eta] 完成 {i}/{len(todo)}，均 wall {avg:.1f}min，"
              f"预计剩余 {remain:.0f}min", flush=True)

# 汇总：全部 JSON 复制到统一目录（k 前缀防冲突）
merged = f"{WORK}/tau_staircase_results"
os.makedirs(merged, exist_ok=True)
n = 0
for k in KS:
    for f in glob.glob(f"{WORK}/results_k{k}/tau_probe_*.json") + \
             glob.glob(f"{WORK}/results_k{k}/tau_ladder_verdict.json"):
        base = os.path.basename(f)
        shutil.copy(f, os.path.join(merged, f"k{k}_{base}"))
        n += 1
print(f"{stamp()} [merge] {n} 个文件（k 前缀防冲突）-> {merged}", flush=True)

# 判读摘要：每个 (k, d) 的 gate_acc 与 mean_acc 一览
print(f"{stamp()} [summary] k/d 一览（ladderon gate_acc@10K / mean_acc@20K）:", flush=True)
for k in KS:
    for d in DEPTHS:
        f = f"{WORK}/results_k{k}/tau_probe_ladderon_d{d}_s{SEED}.json"
        if os.path.exists(f):
            r = json.load(open(f))
            print(f"  k={k:4} d={d}: gate_acc={r.get('gate_acc_k1')} "
                  f"mean_acc={r.get('mean_acc')}", flush=True)
        else:
            print(f"  k={k:4} d={d}: (未完成)", flush=True)
print(f"{stamp()} [done] 总 {(time.time()-t0)/60:.0f}min", flush=True)
