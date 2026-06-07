#!/usr/bin/env python3
"""AwareLiquid-M2 — 125M-class MT-LNN from-scratch pretraining with v2.0 modules.

This kernel:
  1. Clones the M1 repo (everest-an/M1) at HEAD.
  2. Tokenises WikiText-103 (gpt2 tokenizer) to data/{train,validation}.bin.
  3. Pretrains the MT-LNN backbone with all four v2.0 bio-inspired modules ON
     (Phase A competitive GWT, Phase C predictive world model, Phase D Hebbian).
  4. Streams v2 module health metrics to /kaggle/working/metrics.jsonl
     (bounded scalars: competition entropy, surprise, etc. — the "eyes").

Cross-session resume: attach the previous run's output as a dataset and the
script auto-resumes from checkpoints/last.pt if present.

This first launch is a STABILITY-VALIDATION run (short): enough steps to clear
EMA warmup and confirm no representational collapse / routing collapse before
committing to the multi-day full run.
"""
import os
import subprocess
import sys
import time

t_start = time.time()

# ---------------------------------------------------------------------------
# 1. Environment
# ---------------------------------------------------------------------------
print("=" * 70)
print("AwareLiquid-M2 pretraining — setup")
print("=" * 70)

REPO = "https://github.com/everest-an/M1.git"
DIR = "/kaggle/working/M1"
if not os.path.exists(DIR):
    subprocess.run(["git", "clone", "--depth", "1", REPO, DIR], check=True)
os.chdir(DIR)

# CRITICAL: Kaggle ships a torch build matched to the assigned GPU (e.g. the
# P100 is compute-capability sm_60, which recent torch wheels have *dropped*).
# Installing datasets/transformers can silently upgrade torch to a build with
# no sm_60 kernels → "CUDA error: no kernel image is available for execution on
# the device" on the first forward pass. Pin the pre-installed torch via a pip
# constraints file so the extras resolve their deps but can NEVER replace torch.
import torch as _pretorch  # noqa: E402  (Kaggle pre-installs a GPU-matched torch)
_torch_public = _pretorch.__version__.split("+")[0]   # strip +cuXXX local tag
_constraints = "/tmp/pip-constraints.txt"
with open(_constraints, "w") as _f:
    # `==X.Y.Z` (no local segment) matches the installed `X.Y.Z+cuNNN` build,
    # so this pins without forcing a reinstall.
    _f.write(f"torch=={_torch_public}\n")
print(f"[env] pinning torch=={_torch_public} (pre-installed, GPU-matched) "
      f"so pip cannot upgrade it")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-c", _constraints,
     "datasets", "transformers", "tokenizers", "tqdm", "einops"],
    check=True,
)

import torch  # noqa: E402  (after pip install)
print(f"torch {torch.__version__} | cuda={torch.cuda.is_available()} "
      f"| device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
# Fail fast & loud if torch can't actually run a kernel on this GPU, instead of
# burning the data-tokenisation time only to crash on the first forward pass.
if torch.cuda.is_available():
    try:
        _probe = (torch.ones(8, device="cuda") * 2).sum().item()
        assert _probe == 16.0
        print(f"[env] CUDA kernel probe OK on {torch.cuda.get_device_name(0)}")
    except Exception as _e:  # pragma: no cover
        raise RuntimeError(
            f"torch {torch.__version__} cannot execute kernels on "
            f"{torch.cuda.get_device_name(0)} (likely a compute-capability "
            f"mismatch from a torch upgrade): {_e}"
        )

WORK = "/kaggle/working"
DATA_DIR = os.path.join(WORK, "data")
CKPT_DIR = os.path.join(WORK, "checkpoints")
METRICS = os.path.join(WORK, "metrics.jsonl")
os.makedirs(CKPT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Data — tokenise WikiText-103 once (skip if already present)
# ---------------------------------------------------------------------------
if not os.path.exists(os.path.join(DATA_DIR, "meta.json")):
    print("\n[data] tokenising WikiText-103 (gpt2) → data/*.bin …")
    subprocess.run(
        [sys.executable, "prepare_data.py",
         "--dataset", "wikitext", "--config", "wikitext-103-raw-v1",
         "--tokenizer", "gpt2", "--out_dir", DATA_DIR],
        check=True,
    )
else:
    print("\n[data] reusing existing tokenised data/")

# ---------------------------------------------------------------------------
# 3. Resume detection (cross-session)
# ---------------------------------------------------------------------------
resume_args = []
# Prefer an attached previous-run checkpoint, else a checkpoint in this session.
candidates = [
    os.path.join(CKPT_DIR, "last.pt"),
]
# Any attached input dataset checkpoints
for root in ("/kaggle/input",):
    if os.path.isdir(root):
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if fn.endswith(".pt") and ("ckpt_" in fn or fn in ("last.pt", "final.pt")):
                    candidates.append(os.path.join(dp, fn))
existing = [c for c in candidates if os.path.exists(c)]
if existing:
    # Pick the most recently modified checkpoint.
    resume_ckpt = max(existing, key=os.path.getmtime)
    resume_args = ["--resume", resume_ckpt]
    print(f"\n[resume] will resume from {resume_ckpt}")
else:
    print("\n[resume] no checkpoint found — fresh start")

# ---------------------------------------------------------------------------
# 4. Train — 125M-class config, all v2.0 modules ON
# ---------------------------------------------------------------------------
# STABILITY-VALIDATION run: short. grad_accum=1 so `step` == optimizer step,
# making "first ~1000 steps" monitoring unambiguous. Flip STEPS up (and
# grad_accum to 64) once metrics confirm health.
STEPS = int(os.environ.get("M2_STEPS", "1200"))
BATCH = int(os.environ.get("M2_BATCH", "8"))
GRAD_ACCUM = int(os.environ.get("M2_GRAD_ACCUM", "1"))

cmd = [
    sys.executable, "train.py",
    "--data_dir", DATA_DIR,
    "--ckpt_dir", CKPT_DIR,
    "--metrics_jsonl", METRICS,
    "--metrics_every", "50",
    "--wandb_run_name", "awareliquid-m2",
    # 125M-class architecture (Tensor-Core aligned: 832 = 13 × 64)
    "--d_model", "832", "--n_layers", "12", "--n_heads", "13", "--n_kv_heads", "1",
    "--seq_len", "512",
    # schedule
    "--batch", str(BATCH), "--grad_accum", str(GRAD_ACCUM),
    "--lr", "6e-4", "--warmup_steps", "120", "--steps", str(STEPS),
    "--log_every", "50", "--eval_every", "400", "--eval_batches", "40",
    "--save_every", "400",
    # v2.0 modules — the whole point of M2
    "--competitive_gwtb", "--n_bids", "3",
    "--world_model", "--world_model_weight", "0.01", "--world_model_grad_clip", "1.0",
    "--hebbian", "--hebbian_lr", "1e-4",
] + resume_args

print("\n[train] " + " ".join(cmd) + "\n")
subprocess.run(cmd, check=True)

# Keep a stable filename for the next session's auto-resume.
final = os.path.join(CKPT_DIR, "final.pt")
last = os.path.join(CKPT_DIR, "last.pt")
if os.path.exists(final):
    import shutil
    shutil.copyfile(final, last)
    print(f"[resume] copied final.pt → last.pt for next-session continuation")

print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. "
      f"Outputs: {CKPT_DIR}/  +  {METRICS}")
