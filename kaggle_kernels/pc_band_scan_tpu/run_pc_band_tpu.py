#!/usr/bin/env python3
"""B-2'' pointer_chase 探雷 TPU 接力腿（v5 lite，medium 规模）。

背景：T4 腿实测 d1 配置 86.7min/个(0.52s/步)，d4 要 ×3-5 → 全量 9 配置
赶不上 GPU 周配额重置(9/19 00:00Z)。TPU 探针实测 medium d4 = 0.25s/步
(≈13× T4 d4 估算)，用 kernel_sources 挂载 T4 已完成的 JSON 做 resume-skip，
T4 被掐后本腿只补剩余 d4 配置。可行性拍板依据 TPU core audit。

与 T4 腿的等价性纪律（L009 跨臂可比是判读前提）：
  train_xla 是 bundle train_model 的逐行镜像(同 AdamW beta2=0.95/wd=0.01,
  同 CosineAnnealingLR(T_max=steps), 同 rng=seed 批流, 同 clip=1.0, 同 step
  顺序), 只在 sched.step() 后多一句 xm.mark_step()——纯设备冲刷不改数值;
  本地已做 CPU 逐 bit 对账(见 tests 记录)。d_model=1040/batch128/lr3e-4/
  steps10000/seed0 与 T4 腿逐字段一致。

进度日志：同 T4 腿约定 [cfg i/N]/[eta]/[summary]/[verdict]; 步行每 2500。
Resume-safe：pc_scan_k{h}_d{d}_{ladder}.json 存在即跳过; 启动时先把
/kaggle/input 里挂载的 T4 输出复制进 OUT。
本地冒烟：PC_DEVICE=cpu PC_STEPS=200 python run_pc_band_tpu.py
"""
import glob
import json
import os
import shutil
import sys
import time

KS = [2, 4, 8]
N_NODES = 16
D_MODEL = int(os.environ.get("PC_DMODEL", "1040"))   # medium：16.8M 参数
STEPS = int(os.environ.get("PC_STEPS", "10000"))
SEED = 0
GATE_LEARN = 0.55
DEVICE = os.environ.get("PC_DEVICE", "xla:0")
MARK_EVERY = 1            # 探针实测 0.25s/步就是 mark_step/步的口径

CFGS = [(k, 1, "off") for k in KS] + [(k, 4, lad) for k in KS for lad in ("off", "on")]

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
OUT = os.path.join(WORK, "pc_band_results")
os.makedirs(OUT, exist_ok=True)
t0 = time.time()


def stamp():
    return f"[{time.strftime('%H:%M:%S')} @{(time.time()-t0)/60:.0f}min]"


def row_path(k, d, ladder):
    return os.path.join(OUT, f"pc_scan_k{k}_d{d}_{ladder}.json")


# --- bundle 定位 ---
DS = None
try:
    import kagglehub
    DS = kagglehub.dataset_download("aricredemption/m1-b1-text-ab-bundle")
except Exception:
    pass
if not DS or not os.path.exists(os.path.join(DS or "", "benchmarks", "reasoning_depth.py")):
    for root, dirs, files in os.walk("/kaggle/input"):
        if os.path.exists(os.path.join(root, "benchmarks", "reasoning_depth.py")):
            DS = root
            break
if not DS:
    cand = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if os.path.exists(os.path.join(cand, "benchmarks", "reasoning_depth.py")):
        DS = cand
assert DS, "bundle not found"
sys.path.insert(0, DS)
os.chdir(DS)
print(f"{stamp()} [setup] bundle at {DS}", flush=True)

import numpy as np
import torch
from benchmarks.reasoning_depth import (build_mtlnn, evaluate_per_k,
                                        make_lm_batch, make_mix_generator)
from benchmarks.reasoning_tasks import make_generator

XM = None
if DEVICE.startswith("xla"):
    import torch_xla  # noqa: F401
    import torch_xla.core.xla_model as xm
    XM = xm
    # 探针实测口径：编译=步1+步2 各 ~35s，之后 0.25s/步
    print(f"{stamp()} [env] torch_xla={torch_xla.__version__} dev={DEVICE} "
          f"cores={torch_xla._XLAC._xla_get_num_devices()}", flush=True)
print(f"{stamp()} [env] torch={torch.__version__} device={DEVICE} steps={STEPS} "
      f"d_model={D_MODEL}", flush=True)

_gen, VOCAB, _ = make_generator("pointer_chase", KS[0], N_NODES, seed=0)
SEQ = _gen(1, np.random.default_rng(0)).tokens.shape[1]


def row(k, d, ladder):
    return {"task": "pointer_chase", "protocol": "pc_band_scan_b2pp",
            "difficulty_k_hops": k, "n_nodes": N_NODES, "depth": d,
            "ladder": ladder, "seed": SEED, "steps": STEPS,
            "d_model": D_MODEL, "params": None,
            "mean_acc_gate": None, "acc_by_k": {},
            "wall_s": None, "device": DEVICE}


def train_xla(model, gen, device, steps, batch, lr, seed, depth, log_every):
    """train_model 的 XLA 镜像：逐行同序，仅多 mark_step；见 docstring 纪律。"""
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    rng = np.random.default_rng(seed)
    model.set_core_iterations(depth)          # 固定深度：等价于每步重复 set
    for step in range(steps):
        ids, labels, _ = make_lm_batch(gen, batch, rng, device)
        out = model(ids, labels=labels)
        loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if XM and (step + 1) % MARK_EVERY == 0:
            XM.mark_step()
        if step % log_every == 0 or step == steps - 1:
            print(f"    step {step:5d}  loss {loss.item():.4f}", flush=True)
    return model


def run_cfg(k, d, ladder):
    torch.manual_seed(SEED)
    np.random.default_rng(SEED)
    mix = make_mix_generator("pointer_chase", k, N_NODES)
    model = build_mtlnn(VOCAB, SEQ, max_depth=d, seed=SEED, d_model=D_MODEL)
    if ladder == "on":
        for block in model.blocks:
            block.lnn.resonance.ladder = True
            block.lnn.resonance.ladder_kappa = 1.0
    model = model.to(DEVICE)
    n_params = model.get_num_params()
    train_xla(model, mix, DEVICE, STEPS, 128, 3e-4, SEED, d,
              log_every=max(1, STEPS // 4))
    acc = evaluate_per_k(model, "pointer_chase", k, N_NODES,
                         np.random.default_rng(10_000 + SEED), DEVICE,
                         batches=8, batch=256)
    mean_acc = float(np.mean(list(acc.values())))
    r = row(k, d, ladder)
    r.update(params=n_params, mean_acc_gate=round(mean_acc, 4),
             acc_by_k={str(x): round(v, 4) for x, v in acc.items()})
    tmp = row_path(k, d, ladder) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=1)
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


def seed_from_mounted_t4():
    """把挂载的 T4 腿输出(/kaggle/input/**/pc_scan_*.json)复制进 OUT → resume-skip。"""
    if not os.path.isdir("/kaggle/input"):
        return []
    copied = []
    for p in glob.glob("/kaggle/input/**/*.json", recursive=True):
        base = os.path.basename(p)
        if base.startswith("pc_scan_k") and not os.path.exists(os.path.join(OUT, base)):
            shutil.copy(p, os.path.join(OUT, base))
            copied.append(base)
    return copied


def main():
    copied = seed_from_mounted_t4()
    print(f"{stamp()} [resume] 从挂载 T4 输出继承 {len(copied)} 个配置: {copied}",
          flush=True)
    todo = [c for c in CFGS if not done(*c)]
    print(f"{stamp()} [plan] 待跑 {len(todo)}/{len(CFGS)}: {todo}", flush=True)
    walls = []
    for i, (k, d, ladder) in enumerate(todo, 1):
        print(f"{stamp()} [cfg {i}/{len(todo)}] START k={k} d={d} "
              f"ladder={ladder} ({STEPS} 步)", flush=True)
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

    print(f"{stamp()} [summary] k_hops × depth × ladder mean_acc@{STEPS}:", flush=True)
    band, all_wall, all_ceil = False, True, True
    for k in KS:
        r1, r4o, r4n = load(k, 1, "off"), load(k, 4, "off"), load(k, 4, "on")
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
        v = "MIXED — 无明确带（部分墙/部分天花板但无 d1<0.55≤d4），登记边界"
    print(f"{stamp()} [verdict] {v}", flush=True)
    print(f"{stamp()} [done] 总 {(time.time()-t0)/60:.0f}min", flush=True)


if __name__ == "__main__":
    main()
