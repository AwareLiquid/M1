#!/usr/bin/env python3
"""AwareLiquid -- TRUE cross-domain continual-learning differentiator validation.

The honest test of M1's pitch against a frozen frontier model: does adding the
liquid core (competitive GWT + predictive world model + Hebbian + predictive
coding) to a backbone REDUCE catastrophic forgetting when the model is trained
on a new domain, vs the SAME backbone with the liquid core OFF?

This kernel:
  1. Clones the M1 repo at HEAD.
  2. Tokenises TWO genuinely different corpora (gpt2 BPE):
       domain A = WikiText-103 (encyclopedic English)  -> data_wiki/
       domain B = TinyStories   (simple narrative)      -> data_tiny/
     Each capped to ~40M train tokens (far more than the experiment consumes) so
     tokenisation is a couple of minutes, not half an hour.
  3. Runs experiments/continual_crossdomain_liquid.py:
       train A -> measure A & B held-out PPL -> train B -> re-measure.
       forgetting = A_after - A_before, for arms {dense, liquid, consolidation},
       multiple seeds. `consolidation` = dense backbone + EWC (Fisher-weighted
       anchor to A), the real reconsolidation anti-forgetting mechanism.
  4. Copies the JSON + MD report to /kaggle/working so it downloads cleanly.

The verdict (SUPPORTED / NOT-SUPPORTED) is pre-registered in the experiment and
reported regardless of sign. A negative result is a valid finding.
"""
import os
import subprocess
import sys
import time

t_start = time.time()

print("=" * 70)
print("AwareLiquid cross-domain continual-learning validation -- setup")
print("=" * 70)

REPO = "https://github.com/everest-an/M1.git"
DIR = "/tmp/M1"
if not os.path.exists(DIR):
    subprocess.run(["git", "clone", "--depth", "1", REPO, DIR], check=True)
os.chdir(DIR)
sys.path.insert(0, DIR)

# --- GPU/torch compatibility (same dance as the M2 pretrain kernel) ----------
# Kaggle's pre-installed torch (cu128) dropped Pascal (sm_60) kernels yet Kaggle
# still hands out Tesla P100 (sm_60). Detect via device props (no kernel launch)
# and install a Pascal-capable torch (2.6.0+cu124) if needed.
import torch as _pretorch  # noqa: E402
_cap = _pretorch.cuda.get_device_capability(0) if _pretorch.cuda.is_available() else None
_dev = _pretorch.cuda.get_device_name(0) if _pretorch.cuda.is_available() else "cpu"
print(f"[env] pre-installed torch {_pretorch.__version__} | device={_dev} | capability={_cap}")

if _cap is not None and _cap[0] < 7:
    print(f"[env] {_dev} (sm_{_cap[0]}{_cap[1]}) unsupported by pre-installed torch "
          f"-- installing Pascal-compatible torch 2.6.0+cu124")
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
print(f"[env] pinning torch=={_torch_public} for the extras install")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-c", _constraints,
     "datasets", "transformers", "tokenizers", "tqdm", "einops"],
    check=True,
)

# Fail fast if torch still can't run a CUDA kernel here -- before tokenisation.
_probe = subprocess.run(
    [sys.executable, "-c",
     "import torch;"
     "ok=(torch.ones(8,device='cuda')*2).sum().item()==16.0;"
     "print('torch',torch.__version__,'on',torch.cuda.get_device_name(0),'OK' if ok else 'BAD');"
     "assert ok"],
)
if _probe.returncode != 0:
    raise RuntimeError(f"torch on {_dev} (cap {_cap}) cannot execute CUDA kernels -- aborting.")

WORK = "/kaggle/working"
DATA_WIKI = os.path.join(WORK, "data_wiki")
DATA_TINY = os.path.join(WORK, "data_tiny")

# ---------------------------------------------------------------------------
# Tokenise the two domains (capped for speed; the experiment uses < 4M tok/phase)
# ---------------------------------------------------------------------------
CAP = os.environ.get("CC_MAX_TOKENS", "40000000")  # per-split cap

if not os.path.exists(os.path.join(DATA_WIKI, "meta.json")):
    print("\n[data] tokenising WikiText-103 (domain A) ...")
    subprocess.run(
        [sys.executable, "prepare_data.py",
         "--dataset", "wikitext", "--config", "wikitext-103-raw-v1",
         "--tokenizer", "gpt2", "--max_tokens", CAP, "--out_dir", DATA_WIKI],
        check=True,
    )
else:
    print("\n[data] reusing existing data_wiki/")

if not os.path.exists(os.path.join(DATA_TINY, "meta.json")):
    print("\n[data] tokenising TinyStories (domain B) ...")
    subprocess.run(
        [sys.executable, "prepare_data.py",
         "--dataset", "roneneldan/TinyStories", "--config", "none",
         "--tokenizer", "gpt2", "--max_tokens", CAP, "--out_dir", DATA_TINY],
        check=True,
    )
else:
    print("\n[data] reusing existing data_tiny/")

# ---------------------------------------------------------------------------
# Run the experiment
# ---------------------------------------------------------------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

STEPS_A = os.environ.get("CC_STEPS_A", "1200")
STEPS_B = os.environ.get("CC_STEPS_B", "1200")
SEEDS = os.environ.get("CC_SEEDS", "0,1,2")
D_MODEL = os.environ.get("CC_D_MODEL", "512")
N_LAYERS = os.environ.get("CC_N_LAYERS", "6")
N_HEADS = os.environ.get("CC_N_HEADS", "8")
BATCH = os.environ.get("CC_BATCH", "16")
SEQ = os.environ.get("CC_SEQ", "256")
# Arms include `consolidation` = dense backbone + EWC (the real reconsolidation
# anti-forgetting mechanism). EWC strength / Fisher batches are tunable.
ARMS = os.environ.get("CC_ARMS", "dense,liquid,consolidation")
TREATMENT = os.environ.get("CC_TREATMENT", "consolidation")
EWC_LAMBDA = os.environ.get("CC_EWC_LAMBDA", "5000")
FISHER_BATCHES = os.environ.get("CC_FISHER_BATCHES", "50")

cmd = [
    sys.executable, "-m", "experiments.continual_crossdomain_liquid",
    "--data_a", DATA_WIKI, "--name_a", "wikitext103",
    "--data_b", DATA_TINY, "--name_b", "tinystories",
    "--steps_a", STEPS_A, "--steps_b", STEPS_B,
    "--d_model", D_MODEL, "--n_layers", N_LAYERS, "--n_heads", N_HEADS,
    "--batch", BATCH, "--seq_len", SEQ, "--eval_batches", "50",
    "--seeds", SEEDS,
    "--arms", ARMS, "--treatment", TREATMENT,
    "--ewc_lambda", EWC_LAMBDA, "--fisher_batches", FISHER_BATCHES,
    "--out", "report_continual_crossdomain_liquid",
]
print("\n[run] " + " ".join(cmd) + "\n", flush=True)
subprocess.run(cmd, check=True)

# ---------------------------------------------------------------------------
# Copy the report into /kaggle/working so it persists / downloads cleanly.
# ---------------------------------------------------------------------------
import shutil
for ext in (".json", ".md"):
    src = os.path.join(DIR, "experiments", "report_continual_crossdomain_liquid" + ext)
    if os.path.exists(src):
        dst = os.path.join(WORK, "report_continual_crossdomain_liquid" + ext)
        shutil.copyfile(src, dst)
        print(f"[out] {dst}")
        if ext == ".md":
            print("\n" + open(src).read())

print(f"\nDone in {(time.time() - t_start) / 60:.1f} min.")
