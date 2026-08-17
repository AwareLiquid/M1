#!/usr/bin/env python3
"""AwareLiquid-Adapter -- MT-LNN residual adapters on top of TinyLlama-1.1B.

What this kernel does:
  1. Clones M1 repo to /tmp/M1 (keeps /kaggle/working clean).
  2. Downloads TinyLlama-1.1B-Chat-v1.0 weights from HuggingFace.
  3. Attaches MT-LNN residual adapters (0.17% trainable params) via
     attach_mt_adapters() from mt_lnn.llama_adapter.
  4. Fine-tunes the adapters on WikiText-2 for ~2000 steps (~45 min on T4).
  5. Saves the slim adapter checkpoint to /kaggle/working/checkpoints/
     (config + adapter state_dict only, ~15 MB).

The adapter checkpoint is a drop-in for serve/server_hf.py:
  ADAPTER_CKPT=checkpoints/adapter_002000.pt BASE_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0 ...

Cross-session resume: attach the previous output as a dataset; the script
auto-resumes from the latest adapter_*.pt under /kaggle/input.
"""
import os
import subprocess
import sys
import time

t_start = time.time()
print("=" * 70)
print("AwareLiquid-Adapter -- TinyLlama + MT-LNN adapter training")
print("=" * 70)

REPO = "https://github.com/AwareLiquid/M1.git"
DIR = "/tmp/M1"
if not os.path.exists(DIR):
    subprocess.run(["git", "clone", "--depth", "1", REPO, DIR], check=True)
os.chdir(DIR)
sys.path.insert(0, DIR)

# Pascal (sm_60) compat: same fix as the Tiny kernel.
import torch as _pretorch
_cap = _pretorch.cuda.get_device_capability(0) if _pretorch.cuda.is_available() else None
_dev = _pretorch.cuda.get_device_name(0) if _pretorch.cuda.is_available() else "cpu"
print(f"[env] torch {_pretorch.__version__} | device={_dev} | cap={_cap}")
if _cap is not None and _cap[0] < 7:
    print(f"[env] {_dev} sm_{_cap[0]}{_cap[1]} unsupported -- installing torch 2.6.0+cu124")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "torch==2.6.0", "torchvision==0.21.0",
         "--index-url", "https://download.pytorch.org/whl/cu124"],
        check=True,
    )

_torch_ver = subprocess.check_output(
    [sys.executable, "-c", "import torch;print(torch.__version__.split('+')[0])"]
).decode().strip()
_constraints = "/tmp/pip-constraints.txt"
with open(_constraints, "w") as _f:
    _f.write(f"torch=={_torch_ver}\n")
print(f"[env] pinning torch=={_torch_ver}")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-c", _constraints,
     "transformers>=4.40.0", "datasets", "tokenizers", "tqdm", "einops",
     "accelerate"],
    check=True,
)

WORK = "/kaggle/working"
CKPT_DIR = os.path.join(WORK, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

resume_args = []

STEPS     = int(os.environ.get("A_STEPS", "2000"))
BATCH     = int(os.environ.get("A_BATCH", "4"))
GRAD_ACCUM = int(os.environ.get("A_GRAD_ACCUM", "4"))
BASE_MODEL = os.environ.get("A_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

cmd = [
    sys.executable, "train_llama_mt_adapter.py",
    "--model", BASE_MODEL,
    "--dataset", "wikitext", "--dataset_config", "wikitext-2-raw-v1",
    "--steps", str(STEPS),
    "--batch", str(BATCH),
    "--grad_accum", str(GRAD_ACCUM),
    "--seq_len", "512",
    "--lr", "3e-4",
    "--log_every", "50",
    "--save_every", "500",
    "--out_dir", CKPT_DIR,
    "--mt_every", "2",
    "--mt_proto", "13",
    "--mt_scales", "5",
] + resume_args

print("\n[train] " + " ".join(cmd) + "\n")
subprocess.run(cmd, check=True)

# --- VERDICT: did the re-arm actually let the MT residual gate train? -------
import glob
import torch
import torch.nn.functional as F
cks = sorted(glob.glob(os.path.join(CKPT_DIR, "llama_mt_adapter_*.pt")))
print("\n" + "=" * 70)
print("VERDICT (re-arm fix)")
print("=" * 70)
if cks:
    sd = torch.load(cks[-1], map_location="cpu", weights_only=False)["state_dict"]
    scales = {k: float(v) for k, v in sd.items() if k.endswith("mt_adapter.scale")}
    print(f"checkpoint: {os.path.basename(cks[-1])}")
    print("per-layer scale (init was 1e-3 -- ANY movement => fix works):")
    for k in sorted(scales):
        print(f"  {k} = {scales[k]:.6f}")
    moved = any(abs(v - 1e-3) > 1e-6 for v in scales.values())
    print(f"=> scale moved off 1e-3 init: {moved}")
    # tau = softplus(log_tau) + tau_min (tau_min default 0.01); NOT exp(log_tau).
    taus = {k: v for k, v in sd.items() if k.endswith("resonance.log_tau")}
    for k in sorted(taus):
        t = F.softplus(taus[k].float()) + 0.01
        print(f"   {k}: tau min={t.min():.3g} max={t.max():.3g} std={t.std():.3g}")
    print("\nPASS" if moved else "\nFAIL: scale still frozen -- fix did NOT take effect")
else:
    print("no checkpoint found to inspect")

print(f"\nDone in {(time.time()-t_start)/60:.1f} min.  Outputs: {CKPT_DIR}/")
