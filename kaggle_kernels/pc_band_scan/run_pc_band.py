#!/usr/bin/env python3
"""B-2'' pointer_chase 深度敏感带探雷 kernel（T4 GPU，medium 规模）。

预注册：notes/overnight/backlog.md B-2'' 条目（2026-09-18，用户拍板走 Kaggle）。
扫描：k_hops ∈ {2,4,8}（n_nodes=16）× depth {1,4} × ladder {off,on}；
d1 只跑 off 单臂（阶梯在 d1 结构性 no-op，2026-09-18 JSON 逐字节对比确认）
= 9 配置 × 10K 步，batch 128，lr 3e-4，seed 0，d_model=1040（16.8M ≈ medium）。
gate 读数 = train 10K 步后 evaluate_per_k 的 mean_k(acc)（chance=1/16≈0.0625）。

进度日志约定（同 staircase）：[cfg i/N] START/DONE/wall + [eta] 均值外推
+ [summary] 一览表 + [verdict] 带命中/全墙/全天花板 三分支。
Resume-safe：pc_scan_k{h}_d{d}_{ladder}.json 存在即跳过（文件名含全部三轴）。
本地冒烟：PC_DEVICE=mps PC_STEPS=200 python run_pc_band.py
"""
import json
import os
import sys
import time

KS = [2, 4, 8]            # k_hops 难度轴
N_NODES = 16
D_MODEL = 1040            # medium：16.8M 参数（实测 get_num_params）
STEPS = int(os.environ.get("PC_STEPS", "10000"))
SEED = 0
GATE_LEARN = 0.55         # 预注册判据："学得会"线
DEVICE = os.environ.get("PC_DEVICE", "cuda")

# 便宜的 d1 臂先跑 —— 首配置实测校准 ETA（L011）
CFGS = [(k, 1, "off") for k in KS] + [(k, 4, lad) for k in KS for lad in ("off", "on")]

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
OUT = os.path.join(WORK, "pc_band_results")
os.makedirs(OUT, exist_ok=True)
t0 = time.time()


def stamp():
    return f"[{time.strftime('%H:%M:%S')} @{(time.time()-t0)/60:.0f}min]"


def row_path(k, d, ladder):
    return os.path.join(OUT, f"pc_scan_k{k}_d{d}_{ladder}.json")


# --- bundle 定位（kagglehub 优先，挂载扫描兜底）---
DS = None
try:
    import kagglehub
    DS = kagglehub.dataset_download("aricredemption/m1-b1-text-ab-bundle")
except Exception as e:
    print(f"{stamp()} [diag] kagglehub 不可用 ({type(e).__name__})", flush=True)
if not DS or not os.path.exists(os.path.join(DS or "", "benchmarks", "reasoning_depth.py")):
    for root, dirs, files in os.walk("/kaggle/input"):
        if os.path.exists(os.path.join(root, "benchmarks", "reasoning_depth.py")):
            DS = root
            break
if not DS or not os.path.exists(os.path.join(DS, "benchmarks", "reasoning_depth.py")):
    # 本地冒烟兜底：repo root（kaggle_kernels/pc_band_scan/ 的上两级）
    cand = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if os.path.exists(os.path.join(cand, "benchmarks", "reasoning_depth.py")):
        DS = cand
assert DS and os.path.exists(os.path.join(DS, "benchmarks", "reasoning_depth.py")), \
    f"bundle not found: {DS}"
print(f"{stamp()} [setup] bundle at {DS}", flush=True)
sys.path.insert(0, DS)
os.chdir(DS)

import numpy as np
import torch
from benchmarks.reasoning_depth import (build_mtlnn, evaluate_per_k,
                                        make_mix_generator, train_model)
from benchmarks.reasoning_tasks import make_generator

print(f"{stamp()} [env] torch={torch.__version__} cuda={torch.cuda.is_available()} "
      f"device={DEVICE} steps={STEPS} d_model={D_MODEL}", flush=True)

_gen, VOCAB, _ = make_generator("pointer_chase", KS[0], N_NODES, seed=0)
SEQ = _gen(1, np.random.default_rng(0)).tokens.shape[1]
print(f"{stamp()} [setup] pointer_chase vocab={VOCAB} seq_len={SEQ} "
      f"n_nodes={N_NODES}", flush=True)


def run_cfg(k, d, ladder):
    torch.manual_seed(SEED)
    np.random.default_rng(SEED)
    mix = make_mix_generator("pointer_chase", k, N_NODES)
    model = build_mtlnn(VOCAB, SEQ, max_depth=d, seed=SEED, d_model=D_MODEL)
    if ladder == "on":
        # 零参数注入（tau_ladder_probe 同款）：config 事后设置无效，逐 block 设
        for block in model.blocks:
            block.lnn.resonance.ladder = True
            block.lnn.resonance.ladder_kappa = 1.0
    model = model.to(DEVICE)
    n_params = model.get_num_params()

    train_model(model, mix, DEVICE, STEPS, 128, 3e-4, SEED,
                depth_choices=[d], depth_setter="core",
                log_every=max(1, STEPS // 4))
    acc = evaluate_per_k(model, "pointer_chase", k, N_NODES,
                         np.random.default_rng(10_000 + SEED), DEVICE,
                         batches=8, batch=256)
    mean_acc = float(np.mean(list(acc.values())))
    row = {"task": "pointer_chase", "protocol": "pc_band_scan_b2pp",
           "difficulty_k_hops": k, "n_nodes": N_NODES, "depth": d,
           "ladder": ladder, "seed": SEED, "steps": STEPS,
           "d_model": D_MODEL, "params": n_params,
           "mean_acc_gate": round(mean_acc, 4),
           "acc_by_k": {str(x): round(v, 4) for x, v in acc.items()},
           "wall_s": None, "device": DEVICE}
    tmp = row_path(k, d, ladder) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=1)
    os.replace(tmp, row_path(k, d, ladder))
    return mean_acc


def load(k, d, ladder):
    try:
        with open(row_path(k, d, ladder), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def done(k, d, ladder):
    r = load(k, d, ladder)
    return r is not None and r.get("steps") == STEPS and r.get("mean_acc_gate") is not None


todo = [c for c in CFGS if not done(*c)]
print(f"{stamp()} [plan] 待跑 {len(todo)}/{len(CFGS)}: {todo}", flush=True)

walls = []
for i, (k, d, ladder) in enumerate(todo, 1):
    print(f"{stamp()} [cfg {i}/{len(todo)}] START k={k} d={d} ladder={ladder} "
          f"({STEPS} 步)", flush=True)
    tc = time.time()
    try:
        acc = run_cfg(k, d, ladder)
        ok = True
    except Exception as e:
        print(f"{stamp()} [cfg {i}/{len(todo)}] ERROR {type(e).__name__}: {e}",
              flush=True)
        ok = False
    wall = (time.time() - tc) / 60
    if ok:
        walls.append(wall)
        r = load(k, d, ladder)
        r["wall_s"] = round(wall * 60, 1)
        with open(row_path(k, d, ladder), "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=1)
        print(f"{stamp()} [cfg {i}/{len(todo)}] DONE k={k} d={d} "
              f"ladder={ladder} mean_acc={acc:.4f} wall={wall:.1f}min", flush=True)
    else:
        print(f"{stamp()} [cfg {i}/{len(todo)}] FAIL k={k} d={d} "
              f"ladder={ladder}", flush=True)
    if walls:
        avg = sum(walls) / len(walls)
        print(f"{stamp()} [eta] 完成 {i}/{len(todo)}，均 wall {avg:.1f}min，"
              f"预计剩余 {(len(todo)-i)*avg:.0f}min", flush=True)

# --- 判读（预注册三分支，写死）---
print(f"{stamp()} [summary] k_hops × depth × ladder mean_acc@{STEPS}:", flush=True)
band, all_wall, all_ceil = False, True, True
for k in KS:
    r1 = load(k, 1, "off")
    r4o = load(k, 4, "off")
    r4n = load(k, 4, "on")
    a1 = r1["mean_acc_gate"] if r1 else None
    a4 = max((r["mean_acc_gate"] for r in (r4o, r4n) if r), default=None)
    print(f"  k={k:2}: d1(off)={a1}  d4(off)={r4o and r4o['mean_acc_gate']}  "
          f"d4(on)={r4n and r4n['mean_acc_gate']}", flush=True)
    if a1 is not None and a4 is not None:
        if a1 < GATE_LEARN <= a4:
            band = True
        if a4 >= GATE_LEARN:
            all_wall = False
        if a1 < GATE_LEARN:
            all_ceil = False
n_done = len([c for c in CFGS if done(*c)])
if n_done < len(CFGS):
    v = f"INCOMPLETE ({n_done}/{len(CFGS)} 配置完成)"
elif band:
    v = "BAND_HIT — 存在深度敏感带，转正式跑预注册（另行拍板）"
elif all_wall:
    v = "ALL_WALL — pointer_chase medium 不可学，B 线整体结案判负"
elif all_ceil:
    v = "ALL_CEILING — 难度不足，允许加扫 k_hops {16,32} 一轮（预注册授权）"
else:
    v = "MIXED — 无明确带（部分 k 墙/部分 k 天花板但无 d1<0.55≤d4），登记边界"
print(f"{stamp()} [verdict] {v}", flush=True)
print(f"{stamp()} [done] 总 {(time.time()-t0)/60:.0f}min", flush=True)
