#!/usr/bin/env python3
"""Full-module effectiveness ablation -- Kaggle GPU runner (Plan-2 stage 3).

Scientific question
-------------------
Every self-developed module (Hebbian / predictive-coding / world-model /
competitive GWT-B) is now genuinely WIRED into the from-scratch MTLNNModel and
proven jointly trainable (no dead params after the v2.2 Hebbian decouple fix).
Trainable is necessary but NOT sufficient: does enabling each module -- and all
of them together -- actually move held-out perplexity beyond seed noise, on
REAL WikiText-103 (gpt2 tokens), at a scale a CPU cannot reach?

This kernel exists because the local CPU smoke (20-200 steps, PPL ~40k) is
scientifically meaningless -- only a GPU run at real depth/length can produce a
verdict. The verdict is COMPUTED (best arm's delta vs seed noise), not asserted.

What it does
------------
  1. Clone the repo (for mt_lnn + experiments.ablation_full_modules + train.py).
  2. Pascal (P100, sm_60) torch compatibility shim, then install deps.
  3. Tokenise WikiText-103 (gpt2 BPE) -> /kaggle/working/data/*.bin via
     prepare_data.main() in a fresh subprocess (defaults are wikitext-103-raw-v1).
  4. Run experiments.ablation_full_modules with GPU-scale knobs (env AB_*),
     pointing MTLNN_ABLATION_DATA at the tokenised corpus.
  5. Copy the report_ablation_full_modules.{json,md} into /kaggle/working so the
     PPL table + computed verdict land in the kernel output.

HONEST SCOPE: a within-seed-noise result is still the legitimate, publishable
outcome (it bounds the effect size). This measures; it does not pre-decide.
"""
import os
import shutil
import subprocess
import sys
import time

t_start = time.time()
REPO = "https://github.com/everest-an/M1.git"
DIR = "/tmp/M1"                      # NOT /kaggle/working -> keep output clean
WORK = "/kaggle/working"
DATA_DIR = os.path.join(WORK, "data")

print("=" * 70)
print("Full-module effectiveness ablation (Plan-2 stage 3)")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1. Clone repo
# ---------------------------------------------------------------------------
if not os.path.exists(DIR):
    subprocess.run(["git", "clone", "--depth", "1", REPO, DIR], check=True)
os.chdir(DIR)
sys.path.insert(0, DIR)

# ---------------------------------------------------------------------------
# 2. Pascal (sm_60) torch compatibility -- Kaggle hands out P100s but newest
#    torch dropped Pascal kernels. Reinstall a Pascal-capable torch BEFORE any
#    CUDA op, exactly as the proven tiny-pretrain kernel does.
# ---------------------------------------------------------------------------
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
     "datasets", "transformers", "tokenizers", "tqdm", "einops", "numpy"],
    check=True,
)

# Verify CUDA actually executes before the (longer) tokenisation + training.
_probe = subprocess.run(
    [sys.executable, "-c",
     "import torch;assert (torch.ones(8,device='cuda')*2).sum().item()==16.0;"
     "print('torch',torch.__version__,'on',torch.cuda.get_device_name(0),'OK')"],
)
if _probe.returncode != 0:
    raise RuntimeError("torch cannot execute CUDA kernels on this GPU -- aborting.")

# ---------------------------------------------------------------------------
# 3. Data -- tokenise WikiText-103 (gpt2). prepare_data defaults already point at
#    wikitext / wikitext-103-raw-v1 / gpt2, so call main() with the defaults.
#    Subprocess + fresh interpreter: never import transformers/torch in-process
#    after a Pascal torch reinstall (broken torch._dynamo flex_attention import).
# ---------------------------------------------------------------------------
if not os.path.exists(os.path.join(DATA_DIR, "meta.json")):
    print("\n[data] tokenising wikitext-103-raw-v1 (gpt2) -> data/*.bin ...")
    _tok = (
        "import prepare_data;from types import SimpleNamespace;"
        "prepare_data.main(SimpleNamespace(dataset='wikitext',"
        "config='wikitext-103-raw-v1',tokenizer='gpt2',out_dir=%r,max_tokens=0))"
        % DATA_DIR
    )
    subprocess.run([sys.executable, "-c", _tok], check=True, cwd=DIR)
else:
    print("\n[data] reusing existing tokenised data/")

# ---------------------------------------------------------------------------
# 4. Run the ablation at GPU scale. Knobs come from env (AB_*); defaults here
#    are a real run (not the CPU-smoke defaults baked into the script).
# ---------------------------------------------------------------------------
env = dict(os.environ)
env["MTLNN_ABLATION_DATA"] = DATA_DIR
env.setdefault("AB_SEEDS", "0,1,2")
env.setdefault("AB_STEPS", "2000")
env.setdefault("AB_BATCH", "16")
env.setdefault("AB_SEQ_LEN", "256")
env.setdefault("AB_LR", "6e-4")
env.setdefault("AB_D_MODEL", "256")
env.setdefault("AB_N_LAYERS", "4")
env.setdefault("AB_N_HEADS", "8")
env.setdefault("AB_EVAL_BATCHES", "60")
env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

print("\n[run] experiments.ablation_full_modules  "
      f"seeds={env['AB_SEEDS']} steps={env['AB_STEPS']} d_model={env['AB_D_MODEL']} "
      f"n_layers={env['AB_N_LAYERS']} seq_len={env['AB_SEQ_LEN']}\n")
subprocess.run([sys.executable, "-m", "experiments.ablation_full_modules"],
               check=True, cwd=DIR, env=env)

# ---------------------------------------------------------------------------
# 5. Surface the report into the kernel output.
# ---------------------------------------------------------------------------
for fn in ("report_ablation_full_modules.json", "report_ablation_full_modules.md"):
    src = os.path.join(DIR, "experiments", fn)
    if os.path.exists(src):
        shutil.copyfile(src, os.path.join(WORK, fn))
        print(f"[out] {fn} -> {WORK}")

# Echo the markdown verdict to the kernel log for at-a-glance reading.
md = os.path.join(WORK, "report_ablation_full_modules.md")
if os.path.exists(md):
    print("\n" + "=" * 70)
    with open(md) as fh:
        print(fh.read())
    print("=" * 70)

print(f"\n[done] total runtime {time.time() - t_start:.0f}s")
