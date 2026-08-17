#!/usr/bin/env python3
"""AwareLiquid-M2 -- 125M-class MT-LNN from-scratch pretraining with v2.0 modules.

This kernel:
  1. Clones the M1 repo (AwareLiquid/M1) at HEAD.
  2. Tokenises WikiText-103 (gpt2 tokenizer) to data/{train,validation}.bin.
  3. Pretrains the MT-LNN backbone with all four v2.0 bio-inspired modules ON
     (Phase A competitive GWT, Phase C predictive world model, Phase D Hebbian).
  4. Streams v2 module health metrics to /kaggle/working/metrics.jsonl
     (bounded scalars: competition entropy, surprise, etc. -- the "eyes").

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
print("AwareLiquid-M2 pretraining -- setup")
print("=" * 70)

REPO = "https://github.com/AwareLiquid/M1.git"
DIR = "/tmp/M1"   # clone OUTSIDE /kaggle/working so the output stays clean
if not os.path.exists(DIR):                # (only checkpoints/ + metrics.jsonl ->
    subprocess.run(["git", "clone", "--depth", "1", REPO, DIR], check=True)  # fast DL)
os.chdir(DIR)
sys.path.insert(0, DIR)

# CRITICAL GPU/torch compatibility (diagnosed 2026-06-07 from a failed run):
# Kaggle's *pre-installed* torch (2.10.0+cu128) DROPPED Pascal (sm_60) kernels
# -- its supported set is sm_70..sm_120 -- yet Kaggle still hands out Tesla P100
# (sm_60) GPUs. On such a node the very first CUDA op dies with
# "CUDA error: no kernel image is available for execution on the device".
# get_device_capability() only reads device props (no kernel launch), so we can
# detect the mismatch up front and, if needed, install a torch build that still
# ships Pascal kernels (Pascal support was removed in torch 2.7, so 2.6.0+cu124
# is the last safe line). train.py / prepare_data.py run in *subprocesses*, so
# they pick up whatever torch we settle on here.
import torch as _pretorch  # noqa: E402  (Kaggle pre-installs torch)
_cap = _pretorch.cuda.get_device_capability(0) if _pretorch.cuda.is_available() else None
_dev = _pretorch.cuda.get_device_name(0) if _pretorch.cuda.is_available() else "cpu"
print(f"[env] pre-installed torch {_pretorch.__version__} | device={_dev} | "
      f"capability={_cap}")

if _cap is not None and _cap[0] < 7:
    # Pascal (or older): the cu128 wheel can't run here. Install a Pascal-capable
    # torch. cu124 wheels for 2.6.0 include sm_50..sm_90; forward-compatible with
    # the newer Kaggle driver.
    print(f"[env] {_dev} (sm_{_cap[0]}{_cap[1]}) is NOT supported by the "
          f"pre-installed torch -- installing Pascal-compatible torch 2.6.0+cu124")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "torch==2.6.0", "torchvision==0.21.0",
         "--index-url", "https://download.pytorch.org/whl/cu124"],
        check=True,
    )

# Whatever torch is now current, pin it so the extras' dep resolution can't
# replace it. Query a fresh interpreter (the parent's `torch` may be stale after
# a reinstall above).
_torch_public = subprocess.check_output(
    [sys.executable, "-c", "import torch;print(torch.__version__.split('+')[0])"]
).decode().strip()
_constraints = "/tmp/pip-constraints.txt"
with open(_constraints, "w") as _f:
    # `==X.Y.Z` (no local segment) matches the installed `X.Y.Z+cuNNN` build.
    _f.write(f"torch=={_torch_public}\n")
print(f"[env] pinning torch=={_torch_public} for the extras install")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-c", _constraints,
     "datasets", "transformers", "tokenizers", "tqdm", "einops"],
    check=True,
)

# Fail fast & loud if torch still can't execute a kernel on this GPU -- BEFORE
# the ~7-min data tokenisation, not on the first training forward pass. Run in a
# fresh subprocess so it reflects the final installed torch.
_probe = subprocess.run(
    [sys.executable, "-c",
     "import torch;"
     "ok=(torch.ones(8,device='cuda')*2).sum().item()==16.0;"
     "print('torch',torch.__version__,'on',torch.cuda.get_device_name(0),'OK' if ok else 'BAD');"
     "assert ok"],
)
if _probe.returncode != 0:
    raise RuntimeError(
        f"torch on {_dev} (capability {_cap}) still cannot execute CUDA "
        f"kernels after the compatibility install step -- aborting before "
        f"tokenisation."
    )

WORK = "/kaggle/working"
DATA_DIR = os.path.join(WORK, "data")
CKPT_DIR = os.path.join(WORK, "checkpoints")
METRICS = os.path.join(WORK, "metrics.jsonl")
os.makedirs(CKPT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Data -- tokenise WikiText-103 once (skip if already present)
# ---------------------------------------------------------------------------
if not os.path.exists(os.path.join(DATA_DIR, "meta.json")):
    print("\n[data] tokenising WikiText-103 (gpt2) -> data/*.bin ...")
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
    print("\n[resume] no checkpoint found -- fresh start")

# ---------------------------------------------------------------------------
# 4. Train -- 125M-class config, all v2.0 modules ON
# ---------------------------------------------------------------------------
# STABILITY-VALIDATION run: short. grad_accum=1 so `step` == optimizer step,
# making "first ~1000 steps" monitoring unambiguous. Flip STEPS up (and
# grad_accum to 64) once metrics confirm health.
#
# MEMORY (diagnosed 2026-06-07 from the v3 run): the 131.4M model at batch 8 x
# seq 512 OOM'd on a 16 GB T4 -- it reached 15.04 GiB allocated at the forward's
# cross-entropy with only 321 MiB free. That OOM was in the *forward* of step 1;
# backward needs strictly more, so batch 8 can never fit on a 16 GB card (both
# the T4 and the Pascal P100 Kaggle hands out are 16 GB). Drop the per-device
# micro-batch to 4 (still grad_accum=1 -> step == optimizer step) and turn on the
# allocator's expandable_segments to curb fragmentation. Raise M2_BATCH via env
# only after confirming headroom on the assigned GPU.
# The first run (1200 steps, grad_accum=1) was a stability check -- PASSED (no
# collapse, val PPL 877->800). This is SESSION 1 of the real run: sized to FINISH
# inside one ~9h Kaggle session so it saves a clean final.pt/serve.pt.
#   throughput ? 1620 tok/s; tokens/opt-step = batch*seq*grad_accum = 4*512*4 = 8192
#   ~7h ? 41M tokens ? ~5000 opt-steps  (?16x the tokens of the 1200-step probe).
# The cosine spans these 5000 steps. To extend later, raise M2_STEPS and resume
# (attach this run's output as a dataset; resume restores the global step).
STEPS = int(os.environ.get("M2_STEPS", "5000"))
BATCH = int(os.environ.get("M2_BATCH", "4"))
GRAD_ACCUM = int(os.environ.get("M2_GRAD_ACCUM", "4"))    # global batch 16

# Reduce CUDA fragmentation OOMs (the v3 crash had 244 MiB reserved-but-unallocated
# at the moment it ran out). Inherited by the train.py subprocess.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

cmd = [
    sys.executable, "train.py",
    "--data_dir", DATA_DIR,
    "--ckpt_dir", CKPT_DIR,
    "--metrics_jsonl", METRICS,
    "--metrics_every", "50",
    "--wandb_run_name", "awareliquid-m2",
    # 125M-class architecture (Tensor-Core aligned: 832 = 13 x 64)
    "--d_model", "832", "--n_layers", "12", "--n_heads", "13", "--n_kv_heads", "1",
    "--seq_len", "512",
    # schedule
    "--batch", str(BATCH), "--grad_accum", str(GRAD_ACCUM),
    "--lr", "6e-4", "--warmup_steps", "120", "--steps", str(STEPS),
    "--log_every", "50", "--eval_every", "400", "--eval_batches", "40",
    "--save_every", "400",
    # v2.0 modules -- the whole point of M2
    "--competitive_gwtb", "--n_bids", "3",
    "--world_model", "--world_model_weight", "0.01", "--world_model_grad_clip", "1.0",
    "--hebbian", "--hebbian_lr", "1e-4",
] + resume_args

print("\n[train] " + " ".join(cmd) + "\n")
subprocess.run(cmd, check=True)

# Keep a stable filename for the next session's auto-resume.
final = os.path.join(CKPT_DIR, "final.pt")
last = os.path.join(CKPT_DIR, "last.pt")
# IMPORTANT: this is a COSMETIC post-training step (stable resume name + a slim
# server checkpoint). It must NEVER be able to fail the whole kernel and mark a
# successful multi-thousand-step training run as ERROR (which is exactly what
# happened on 2026-06-16: torch.load defaulted to weights_only=True on the
# Kaggle torch and choked on the embedded MTLNNConfig, turning a clean 5000-step
# run -- collapse_gate=0.000, val PPL 136 -- into an ERROR status). So: load our
# OWN trusted checkpoint with weights_only=False, and guard everything so any
# failure here only WARNS.
if os.path.exists(final):
    try:
        import shutil
        shutil.copyfile(final, last)
        print(f"[resume] copied final.pt -> last.pt for next-session continuation")
        # Slim, server-loadable checkpoint (config + model_state, no optimizer
        # state) -- ~1/3 the size of final.pt, so it downloads over a flaky proxy
        # and drops straight into serve/server.py + the demo.
        import torch as _t
        _ck = _t.load(final, map_location="cpu", weights_only=False)
        _slim = {"config": _ck["config"], "model_state": _ck["model_state"],
                 "step": _ck.get("step"), "loss": _ck.get("loss")}
        _t.save(_slim, os.path.join(CKPT_DIR, "serve.pt"))
        print(f"[serve] wrote slim serve.pt "
              f"({os.path.getsize(os.path.join(CKPT_DIR, 'serve.pt'))/1e6:.0f} MB)")
    except Exception as _e:  # noqa: BLE001 - cosmetic step, never fail the run
        print(f"[serve] WARNING: post-training slim/resume step failed "
              f"({type(_e).__name__}: {_e}); training + checkpoints are intact, "
              f"continuing so the kernel still completes cleanly.")

print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. "
      f"Outputs: {CKPT_DIR}/  +  {METRICS}")
