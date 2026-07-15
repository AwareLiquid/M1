"""modal_e1_baselines.py -- E1 fair-baseline run on Modal T4 (Kaggle weekly
GPU quota exhausted by the first attempt's 8h timeout).

Runs the repo's scaling_comparison harness at branch physics-informed-head:
transformer / mt_lnn / mamba, seed 0, WikiText-103, 2000 steps, fp16 AMP
(T4 is sm_75: no bf16 hardware, so the reviewed fp32-master-weights +
autocast-fp16 + GradScaler path is exactly what executes).

Usage:
    modal run modal_e1_baselines.py                    # full E1, ~3-5h
    modal run modal_e1_baselines.py --preflight-only   # 5-step smoke, ~5 min
"""
import modal

BRANCH = "physics-informed-head"

app = modal.App("mt-lnn-e1-baselines")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "datasets",
        "transformers",
        "tokenizers",
        "tqdm",
        "einops",
    )
    # Clone at BUILD time for layer caching; the function refreshes to the
    # branch tip at RUN time so a rebuilt image is not needed per commit.
    .run_commands(
        "git clone --depth 1 --branch %s "
        "https://github.com/everest-an/M1.git /opt/M1" % BRANCH
    )
)

volume = modal.Volume.from_name("mt-lnn-e1-vol", create_if_missing=True)
OUT_MOUNT = "/results"


@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 60 * 8,
    volumes={OUT_MOUNT: volume},
)
def run_e1(preflight_only: bool = False):
    import json
    import os
    import subprocess
    import sys

    os.chdir("/opt/M1")
    subprocess.run(["git", "fetch", "--depth", "1", "origin", BRANCH], check=True)
    subprocess.run(["git", "checkout", "FETCH_HEAD"], check=True)
    print(subprocess.run(["git", "log", "--oneline", "-1"],
                         capture_output=True, text=True).stdout, flush=True)

    import torch
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} "
          f"| capability {torch.cuda.get_device_capability(0)}", flush=True)

    out_dir = os.path.join(OUT_MOUNT, "scaling_out")

    print("\n=== preflight: 5-step fp16-AMP smoke (wikitext-2) ===", flush=True)
    pf = subprocess.run(
        [sys.executable, "benchmarks/scaling_comparison.py",
         "--mode", "train", "--steps", "5", "--seeds", "0",
         "--archs", "transformer,mt_lnn,mamba",
         "--wikitext", "wikitext-2-raw-v1", "--eval_chunks", "5",
         "--out_dir", "/tmp/preflight_out"],
        timeout=1800,
    )
    if pf.returncode != 0:
        print(f"PREFLIGHT FAILED (exit {pf.returncode})", flush=True)
        sys.exit(pf.returncode)
    print("preflight OK", flush=True)
    if preflight_only:
        return

    print("\n=== E1 train: transformer, mt_lnn, mamba | seed 0 ===", flush=True)
    rc = 0
    try:
        r = subprocess.run(
            [sys.executable, "benchmarks/scaling_comparison.py",
             "--mode", "train", "--steps", "2000", "--seeds", "0",
             "--archs", "transformer,mt_lnn,mamba",
             "--train_token_cap", "50000000",
             "--out_dir", out_dir],
            timeout=7 * 3600,
        )
        rc = r.returncode
    except subprocess.TimeoutExpired:
        print("harness TIMED OUT - dumping completed archs below", flush=True)
        rc = 124

    print("\n=== results ===", flush=True)
    if os.path.isdir(out_dir):
        for f in sorted(os.listdir(out_dir)):
            if f.endswith(".json"):
                with open(os.path.join(out_dir, f)) as fh:
                    print(f, "->", json.dumps(json.load(fh))[:600], flush=True)
    volume.commit()
    sys.exit(rc)


@app.local_entrypoint()
def main(preflight_only: bool = False):
    run_e1.remote(preflight_only=preflight_only)
