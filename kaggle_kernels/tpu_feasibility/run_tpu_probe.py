#!/usr/bin/env python3
"""M1 TPU 训练可行性探针 — mt_lnn × torch_xla on Kaggle TPU (v2-8)。

目的（2026-09-18 用户裁定：不求并行，只求把 TPU 训练能力立起来）：
  阶段A 环境盘点：torch/torch_xla/jax 版本、TPU 是否可达；
  阶段B 真训练：pointer_chase medium(d_model=1040,d=4) 在 XLA 上 20 步，
        首步=编译时间 单独记录，稳态 s/step 与 T4(31步/s)、本机 MPS(0.2-0.3步/s) 对照；
  阶段C 数值对账：同 seed CPU 20 步，比较首步/末步 loss 相对偏差（bf16/f32）。
  [verdict] PASS / FAIL + 原因。输出 tpu_probe_result.json 供收割。
失败也合格：任何阶段炸 → 记录 traceback 继续下一诊断，最终 JSON 落盘。
"""
import json
import os
import sys
import time

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
OUT = os.path.join(WORK, "tpu_probe_result.json")
PROBE_STEPS = 30   # B/C 同步：数值对账按同一步数比 loss_first/loss_last
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

MODEL = {}


def a_env():
    import torch
    env = {"torch": torch.__version__}
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        env["torch_xla"] = torch_xla.__version__
        env["xla_device"] = str(xm.xla_device())
        env["device_count"] = xm.xla_device_count() if hasattr(xm, "xla_device_count") else None
    except Exception as e:
        env["torch_xla"] = f"MISSING ({type(e).__name__}) — 尝试 pip 兜底"
        print(f"{stamp()} [A] torch_xla 缺失，pip install 兜底中…", flush=True)
        import subprocess
        r = subprocess.run([sys.executable, "-m", "pip", "install", "torch_xla"],
                           capture_output=True, text=True, timeout=900)
        env["pip_rc"] = r.returncode
        if r.returncode == 0:
            try:
                import torch_xla
                import torch_xla.core.xla_model as xm
                env["torch_xla"] = torch_xla.__version__ + " (pip)"
                env["xla_device"] = str(xm.xla_device())
            except Exception as e2:
                env["torch_xla"] = f"PIP-THEN-FAIL ({type(e2).__name__}: {e2})"
        else:
            env["pip_tail"] = r.stderr[-400:]
    try:
        import jax
        env["jax"] = jax.__version__
        env["jax_tpus"] = [d.device_kind for d in jax.devices("tpu")] if hasattr(jax, "devices") else "?"
    except Exception as e:
        env["jax"] = f"MISSING ({type(e).__name__})"
    env["colab_tpu_env"] = os.environ.get("TPU_NAME") or os.environ.get("KAGGLE_TPUS") or "n/a"
    R["stages"]["A_env"] = env
    print(f"{stamp()} [A] {json.dumps(env, ensure_ascii=False)}", flush=True)
    import torch
    MODEL["torch"] = torch


def b_train():
    import numpy as np
    torch = MODEL["torch"]
    import torch_xla
    import torch_xla.core.xla_model as xm
    from benchmarks.reasoning_depth import build_mtlnn, make_lm_batch, make_mix_generator
    from benchmarks.reasoning_tasks import make_generator

    device = str(xm.xla_device())
    gen, vocab, _ = make_generator("pointer_chase", 8, 16, seed=0)
    seq = gen(1, np.random.default_rng(0)).tokens.shape[1]
    mix = make_mix_generator("pointer_chase", 8, 16)
    torch.manual_seed(0)
    model = build_mtlnn(vocab, seq, max_depth=4, seed=0, d_model=1040)
    n = model.get_num_params()
    model.set_core_iterations(4)
    model = model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95),
                            weight_decay=0.01)
    rng = np.random.default_rng(0)

    losses, walls = [], []
    print(f"{stamp()} [B] 训练开始：{PROBE_STEPS} 步 × d_model=1040 d=4 on {device}"
          f"（params={n:,}）—— 首步含 XLA 编译，可能数分钟", flush=True)
    for i in range(PROBE_STEPS):
        t1 = time.time()
        ids, labels, _ = make_lm_batch(mix, 128, rng, device)
        out = model(ids, labels=labels)
        loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        xm.mark_step()
        dt = time.time() - t1
        losses.append(float(loss.item()))
        walls.append(dt)
        if i == 0 and dt > 30:
            print(f"{stamp()} [B] 编译完成（首步 {dt:.0f}s），进入稳态", flush=True)
        print(f"{stamp()} [B] step {i+1}/{PROBE_STEPS} loss {losses[-1]:.4f} "
              f"wall {dt:.2f}s", flush=True)
    R["stages"]["B_train"] = {
        "params": n, "device": device, "steps": PROBE_STEPS,
        "compile_s_step0": round(walls[0], 1),
        "steady_sps": round(sum(walls[1:]) / len(walls[1:]), 3),
        "loss_first": losses[0], "loss_last": losses[-1],
        "ref_t4_parity": "31 steps/s", "ref_mps_medium_d4": "0.2-0.3 steps/s",
    }


def c_numeric():
    import numpy as np
    torch = MODEL["torch"]
    from benchmarks.reasoning_depth import build_mtlnn, make_lm_batch, make_mix_generator
    from benchmarks.reasoning_tasks import make_generator

    gen, vocab, _ = make_generator("pointer_chase", 8, 16, seed=0)
    seq = gen(1, np.random.default_rng(0)).tokens.shape[1]
    mix = make_mix_generator("pointer_chase", 8, 16)
    torch.manual_seed(0)
    model = build_mtlnn(vocab, seq, max_depth=4, seed=0, d_model=1040)
    model.set_core_iterations(4)
    model = model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95),
                            weight_decay=0.01)
    rng = np.random.default_rng(0)
    losses = []
    print(f"{stamp()} [C] CPU 数值基准：{PROBE_STEPS} 步（同 seed，比对 B 的 loss）", flush=True)
    for i in range(PROBE_STEPS):
        ids, labels, _ = make_lm_batch(mix, 128, rng, "cpu")
        out = model(ids, labels=labels)
        loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))
        if (i + 1) % 10 == 0 or i == PROBE_STEPS - 1:
            print(f"{stamp()} [C] step {i+1}/{PROBE_STEPS} loss {losses[-1]:.4f}",
                  flush=True)
    R["stages"]["C_cpu_ref"] = {"steps": PROBE_STEPS,
                                "loss_first": losses[0], "loss_last": losses[-1]}
    xb = R["stages"].get("B_train", {})
    if "loss_first" in xb and losses[0] > 0:
        rel0 = abs(xb["loss_first"] - losses[0]) / losses[0]
        rel1 = abs(xb["loss_last"] - losses[-1]) / max(losses[-1], 1e-6)
        R["stages"]["C_numeric_check"] = {"rel_dev_first": round(rel0, 4),
                                           "rel_dev_last": round(rel1, 4),
                                           "tol": 0.05}


stage("A_env", a_env)
stage("B_xla_train", b_train)
stage("C_cpu_numeric", c_numeric)

b = R["stages"].get("B_train") or R["stages"].get("B_xla_train") or {}
if isinstance(b, dict) and "steady_sps" in b and b["steady_sps"] < 60:
    R["verdict"] = (f"PASS — XLA 可训练；稳态 {b['steady_sps']}s/step"
                    f"（对照 T4≈0.032s、本机MPS medium d4≈3.4s；"
                    f"bf16 降精度可再压，见 C 偏差）")
else:
    R["verdict"] = f"FAIL — 见 stages 诊断（B={b.get('error', b if b else 'not-run')}）"
print(f"{stamp()} [verdict] {R['verdict']}", flush=True)
save()
print(f"{stamp()} [done] -> {OUT}", flush=True)
