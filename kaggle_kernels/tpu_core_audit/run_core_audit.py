#!/usr/bin/env python3
"""M1 TPU 核心利用率审计 — 可行性探针 PASS 之后的必答问题 (2026-09-18 用户拍板)。

背景: 可行性探针 B 阶段钉在 xla:0 单核训练, jax 侧却看到 8x TPU v5 lite。
      "8 核 ≈ 8 seeds 并行"目前只是推断, 未实测。Kaggle TPU 配额按会话
      wall-time 计, 单核跑 = 7/8 算力白烧。

三问三答(全部实测, 失败也落 JSON):
  A_view     torch_xla 侧到底可见几核 (_xla_get_devices / xr.local_device_count)?
  B_serial   逐核投工作量(matmul 流), 每核是否可寻址、可执行、速度多少?
  C_concurrent 8 核同时投工作量, 聚合吞吐是单核的几倍? (1x=假并行, ~8x=真并行)
输出 /kaggle/working/tpu_core_audit.json。预计墙钟 <15min, TPU 配额 <0.3h。
"""
import json
import os
import sys
import time

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
OUT = os.path.join(WORK, "tpu_core_audit.json")
t0 = time.time()
R = {"stages": {}, "verdict": None}


def stamp():
    return f"[{time.strftime('%H:%M:%S')} @{(time.time()-t0)/60:.1f}min]"


def save():
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=1)
    os.replace(tmp, OUT)


def stage(name, fn):
    print(f"{stamp()} [{name}] --- start", flush=True)
    try:
        fn()
        print(f"{stamp()} [{name}] ok", flush=True)
    except Exception as e:
        import traceback
        R["stages"][name] = {"error": f"{type(e).__name__}: {e}"}
        print(f"{stamp()} [{name}] ERROR {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
    save()


# ---- bundle ----
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
assert DS, "bundle not found"
sys.path.insert(0, DS)
os.chdir(DS)
print(f"{stamp()} [setup] bundle at {DS}", flush=True)

import torch  # noqa: E402
import torch_xla  # noqa: E402
import torch_xla.core.xla_model as xm  # noqa: E402

DEV_STRINGS = []


def a_view():
    import torch_xla.runtime as xr
    env = {"torch": torch.__version__, "torch_xla": torch_xla.__version__}
    try:
        env["xr_local_device_count"] = xr.local_device_count()
    except Exception as e:
        env["xr_local_device_count"] = f"ERR ({type(e).__name__})"
    try:
        import torch_xla._XLAC as X
        DEV_STRINGS.extend(X._xla_get_devices())
        env["xlac_devices"] = list(DEV_STRINGS)
    except Exception as e:
        env["xlac_devices"] = f"ERR ({type(e).__name__})"
    env["xm.xla_device()"] = str(xm.xla_device())
    R["stages"]["A_view"] = env
    print(f"{stamp()} [A] {json.dumps(env, ensure_ascii=False)}", flush=True)


def _matmul_wall(dev, n=10, size=2048):
    """单核 n 次 size^3 matmul 的墙钟 (秒)。XLA 异步, 用 .item() 强制同步。"""
    a = torch.randn(size, size, device=dev)
    b = torch.randn(size, size, device=dev)
    c = a @ b
    float(c[0, 0].item())  # warm 编译
    t = time.time()
    for _ in range(n):
        c = a @ b
        float(c[0, 0].item())
    return (time.time() - t) / n, c


def b_serial():
    if not DEV_STRINGS:
        raise RuntimeError("A 阶段没拿到设备列表")
    per = {}
    for i, ds in enumerate(DEV_STRINGS):
        t, _ = _matmul_wall(ds)
        gflops = (2 * 2048 ** 3) / t / 1e9
        per[ds] = round(t, 4)
        print(f"{stamp()} [B] {i+1}/{len(DEV_STRINGS)} {ds}: "
              f"{t*1000:.0f}ms/matmul ({gflops:.0f} GFLOP/s)", flush=True)
    R["stages"]["B_serial"] = {"per_device_s": per,
                               "note": "串行逐核, 证明可寻址+可执行"}


def c_concurrent():
    if len(DEV_STRINGS) < 2:
        raise RuntimeError(f"只有 {len(DEV_STRINGS)} 核可见, 无从并发")
    n = 10
    _, _ = _matmul_wall(DEV_STRINGS[0], n=1)  # 全局预热编译
    # 交错提交: 每轮在全部核上各排一个 matmul, XLA 异步 → 核间天然并发
    held = {d: (torch.randn(2048, 2048, device=d),
                torch.randn(2048, 2048, device=d)) for d in DEV_STRINGS}
    outs = {d: None for d in DEV_STRINGS}
    t = time.time()
    for _ in range(n):
        for d, (a, b) in held.items():
            outs[d] = a @ b
    for d in DEV_STRINGS:  # 统一收口同步
        float(outs[d][0, 0].item())
    wall = time.time() - t
    total_ops = n * len(DEV_STRINGS)
    single, _ = _matmul_wall(DEV_STRINGS[0], n=3)
    speedup = (total_ops * single) / wall
    R["stages"]["C_concurrent"] = {
        "world": len(DEV_STRINGS), "ops_total": total_ops, "wall_s": round(wall, 2),
        "s_per_op": round(wall / total_ops, 4), "single_core_s_per_op": round(single, 4),
        "effective_parallelism_x": round(speedup, 2),
        "interpret": "≈1x=单核排队(假并行); 接近 world=真并发",
    }
    print(f"{stamp()} [C] {total_ops} ops over {len(DEV_STRINGS)} devs in "
          f"{wall:.1f}s → 有效并行度 {speedup:.1f}x", flush=True)


stage("A_view", a_view)
stage("B_serial", b_serial)
stage("C_concurrent", c_concurrent)

c = R["stages"].get("C_concurrent", {})
a = R["stages"].get("A_view", {})
n_view = len(DEV_STRINGS) or a.get("xr_local_device_count") or 0
if isinstance(c.get("effective_parallelism_x"), float) and n_view:
    if c["effective_parallelism_x"] >= 0.7 * n_view:
        R["verdict"] = f"TRUE_PARALLEL — {n_view} 核可并发, 正式跑可吃满 TPU 配额"
    elif c["effective_parallelism_x"] < 1.5:
        R["verdict"] = (f"SINGLE_CORE_ONLY — torch 侧有效并行 {c['effective_parallelism_x']}x, "
                        f"8 核吃满需要 SPMD/多进程工程 (未验证前禁止把它当多 seed 车道)")
    else:
        R["verdict"] = f"PARTIAL — 有效并行 {c['effective_parallelism_x']}x / {n_view} 核"
else:
    R["verdict"] = "INCONCLUSIVE — 看 stages 错误详情"
print(f"{stamp()} [verdict] {R['verdict']}", flush=True)
save()
print(f"{stamp()} [done] -> {OUT}", flush=True)
