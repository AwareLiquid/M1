#!/usr/bin/env python3
"""AwareLiquid-Tiny -- small MT-LNN trained on TinyStories for a COHERENT demo.

Why this kernel exists
----------------------
The 125M "M2" run validated architectural stability but is far too undertrained
(val PPL ~800) to produce coherent text on free Kaggle GPU within reasonable
time. TinyStories (Eldan & Li 2023) is a synthetic corpus of simple children's
stories engineered so that SMALL models (10-50M params) learn to generate
fluent, grammatical, coherent English in just a few GPU-hours. A ~49M MT-LNN
trained here gives the public demo real, readable output -- limited in topic
(simple stories) but genuinely coherent -- while the big model trains in parallel.

This kernel:
  1. Clones the M1 repo to /tmp (so /kaggle/working output stays clean: only
     checkpoints/ + metrics.jsonl -- fast to download, no repo/.git pollution).
  2. Tokenises TinyStories (gpt2 BPE) in-process to data/{train,validation}.bin.
  3. Trains a ~49M MT-LNN (d_model=416, 6 layers) with the signature v2 modules
     (competitive GWT + predictive world model) ON -- proven stable in the M2 run.
  4. Saves checkpoints + metrics to /kaggle/working. final.pt is server-loadable
     (state-dict + embedded config) -> drop-in for serve/server.py and the demo.

Cross-session resume: attach the previous run's output as a dataset; the script
auto-resumes from the newest ckpt_*.pt / last.pt found under /kaggle/input.
"""
import os
import subprocess
import sys
import time

t_start = time.time()
print("=" * 70)
print("AwareLiquid-Tiny -- TinyStories pretraining (coherent small-model demo)")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1. Environment -- clone to /tmp (keep /kaggle/working output minimal)
# ---------------------------------------------------------------------------
REPO = "https://github.com/everest-an/M1.git"
DIR = "/tmp/M1"                                   # NOT /kaggle/working -> clean output
if not os.path.exists(DIR):
    subprocess.run(["git", "clone", "--depth", "1", REPO, DIR], check=True)
os.chdir(DIR)
sys.path.insert(0, DIR)

# Pascal (sm_60) compatibility -- Kaggle's newest torch dropped Pascal kernels but
# still hands out P100s; reinstall a Pascal-capable torch BEFORE any CUDA op.
import torch as _pretorch  # noqa: E402
_cap = _pretorch.cuda.get_device_capability(0) if _pretorch.cuda.is_available() else None
_dev = _pretorch.cuda.get_device_name(0) if _pretorch.cuda.is_available() else "cpu"
print(f"[env] torch {_pretorch.__version__} | device={_dev} | capability={_cap}")
if _cap is not None and _cap[0] < 7:
    print(f"[env] {_dev} (sm_{_cap[0]}{_cap[1]}) unsupported -- installing torch 2.6.0+cu124")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "torch==2.6.0", "torchvision==0.21.0",
         "--index-url", "https://download.pytorch.org/whl/cu124"],
        check=True,
    )

_torch_public = subprocess.check_output(
    [sys.executable, "-c", "import torch;print(torch.__version__.split('+')[0])"]
).decode().strip()
_constraints = "/tmp/pip-constraints.txt"
with open(_constraints, "w") as _f:
    _f.write(f"torch=={_torch_public}\n")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-c", _constraints,
     "datasets", "transformers", "tokenizers", "tqdm", "einops"],
    check=True,
)

# Verify CUDA actually executes before the (short) tokenisation.
_probe = subprocess.run(
    [sys.executable, "-c",
     "import torch;assert (torch.ones(8,device='cuda')*2).sum().item()==16.0;"
     "print('torch',torch.__version__,'on',torch.cuda.get_device_name(0),'OK')"],
)
if _probe.returncode != 0:
    raise RuntimeError("torch cannot execute CUDA kernels on this GPU -- aborting.")

WORK = "/kaggle/working"
DATA_DIR = os.path.join(WORK, "data")
CKPT_DIR = os.path.join(WORK, "checkpoints")
METRICS = os.path.join(WORK, "metrics.jsonl")
os.makedirs(CKPT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Data -- tokenise TinyStories (gpt2 BPE). Call prepare_data.main() directly
#    so we can pass config=None (TinyStories has no named config; the CLI's
#    default --config would be WikiText's and would error).
# ---------------------------------------------------------------------------
if not os.path.exists(os.path.join(DATA_DIR, "meta.json")):
    print("\n[data] tokenising roneneldan/TinyStories (gpt2) -> data/*.bin ...")
    # Run in a SUBPROCESS (fresh interpreter) -- never import transformers/torch
    # in-process here: on a Pascal node we reinstalled torch above, and importing
    # transformers in the parent (which already loaded the old torch) triggers a
    # broken torch._dynamo import (flex_attention -> "cannot import name 'skip_code'").
    # prepare_data's CLI can't pass config=None (its default is WikiText's), so
    # invoke main() with an explicit None via -c.
    _tok = (
        "import prepare_data;from types import SimpleNamespace;"
        "prepare_data.main(SimpleNamespace(dataset='roneneldan/TinyStories',"
        "config=None,tokenizer='gpt2',out_dir=%r))" % DATA_DIR
    )
    subprocess.run([sys.executable, "-c", _tok], check=True, cwd=DIR)
else:
    print("\n[data] reusing existing tokenised data/")

# ---------------------------------------------------------------------------
# 3. Cross-session resume
# ---------------------------------------------------------------------------
candidates = [os.path.join(CKPT_DIR, "last.pt")]
if os.path.isdir("/kaggle/input"):
    for dp, _, fns in os.walk("/kaggle/input"):
        for fn in fns:
            if fn.endswith(".pt") and ("ckpt_" in fn or fn in ("last.pt", "final.pt")):
                candidates.append(os.path.join(dp, fn))
existing = [c for c in candidates if os.path.exists(c)]
resume_args = []
if existing:
    resume_ckpt = max(existing, key=os.path.getmtime)
    resume_args = ["--resume", resume_ckpt]
    print(f"\n[resume] resuming from {resume_ckpt}")
else:
    print("\n[resume] no checkpoint found -- fresh start")

# ---------------------------------------------------------------------------
# 4. Train -- ~49M config (d_model=416, 6 layers). TinyStories is easy -> coherent
#    fast. Tunable via env. v2 signature modules ON (proven stable in M2).
# ---------------------------------------------------------------------------
STEPS      = int(os.environ.get("T_STEPS", "20000"))
BATCH      = int(os.environ.get("T_BATCH", "12"))     # 49M (? d_model, ? layers vs 125M)
GRAD_ACCUM = int(os.environ.get("T_GRAD_ACCUM", "4"))  # -> global batch 48; conservative on 16GB
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

cmd = [
    sys.executable, "train.py",
    "--data_dir", DATA_DIR, "--ckpt_dir", CKPT_DIR,
    "--metrics_jsonl", METRICS, "--metrics_every", "50",
    "--d_model", "416", "--n_layers", "6", "--n_heads", "13", "--n_kv_heads", "1",
    "--gwtb_n_heads", "4", "--seq_len", "512",
    "--batch", str(BATCH), "--grad_accum", str(GRAD_ACCUM),
    "--lr", "6e-4", "--warmup_steps", "200", "--steps", str(STEPS),
    "--log_every", "100", "--eval_every", "500", "--eval_batches", "40",
    "--save_every", "1000",
    "--competitive_gwtb", "--n_bids", "3",
    "--world_model", "--world_model_weight", "0.01", "--world_model_grad_clip", "1.0",
] + resume_args

print("\n[train] " + " ".join(cmd) + "\n")
subprocess.run(cmd, check=True)

# Stable filename for next-session auto-resume.
final = os.path.join(CKPT_DIR, "final.pt")
last = os.path.join(CKPT_DIR, "last.pt")
if os.path.exists(final):
    import shutil
    shutil.copyfile(final, last)
    print("[resume] copied final.pt -> last.pt")
    # Slim, server-loadable checkpoint: drop the (large) optimizer state. The
    # server only needs config + model_state, so serve.pt is ~1/3 the size of
    # final.pt -- fast to download over a flaky proxy and a drop-in for the demo.
    # CRITICAL: this whole block is best-effort. A failure here must NEVER crash
    # the kernel -- a non-zero exit makes Kaggle discard the committed output,
    # which previously destroyed a successful 20k-step run (PPL 19.81). Wrap it.
    try:
        import torch as _t
        # torch>=2.6 defaults weights_only=True, which can't unpickle the
        # dataclass config (raises ModuleNotFoundError). Force full unpickle.
        try:
            _ck = _t.load(final, map_location="cpu", weights_only=False)
        except TypeError:
            _ck = _t.load(final, map_location="cpu")  # older torch: no kwarg
        _slim = {"config": _ck["config"], "model_state": _ck["model_state"],
                 "step": _ck.get("step"), "loss": _ck.get("loss")}
        _t.save(_slim, os.path.join(CKPT_DIR, "serve.pt"))
        print(f"[serve] wrote slim serve.pt ({os.path.getsize(os.path.join(CKPT_DIR,'serve.pt'))/1e6:.0f} MB)")
    except Exception as _e:
        import traceback
        print("[serve] WARNING: serve.pt creation failed (final.pt is safe):")
        traceback.print_exc()

print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. "
      f"Outputs: {CKPT_DIR}/  +  {METRICS}")
