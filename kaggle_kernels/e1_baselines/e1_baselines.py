#!/usr/bin/env python3
"""E1 (ITERATION_PRINCIPLES.md): fair-baseline falsification run, seed 0.

Question this run can kill: P4 "the hybrid architecture beats matched
baselines on from-scratch pretraining" - previously supported only by a
single-seed comparison against the repo's own simple Transformer.

Uses the repo's existing resume-safe harness benchmarks/scaling_comparison.py
(--mode train: WikiText-103, gpt2 tokenizer, B=4 T=512, 2000 steps, lr 3e-4)
and runs the NAMED external baseline it already supports but nobody ever ran:
HF Mamba-130m (pure-PyTorch fallback path, correct on P100/sm_60), plus the
matched-width Transformer and the lean MT-LNN core, all under identical data
order and identical trunk settings (E4 lean defaults).

Order: transformer -> mt_lnn -> mamba (mamba's slow path is the wall-clock
unknown; the resume-safe per-arch JSON means a follow-up kernel run finishes
whatever this session cannot).
"""
import json
import os
import subprocess
import sys

BRANCH = "physics-informed-head"
REPO = "https://github.com/everest-an/M1.git"
DIR = "/tmp/M1"
OUT = "/kaggle/working/scaling_out"   # small JSONs -> kernel output artifacts

print("=== setup ===", flush=True)
if not os.path.exists(DIR):
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO, DIR],
                   check=True, timeout=300)
os.chdir(DIR)
# P100 is sm_60: Kaggle's preinstalled torch has no sm_60 kernels. Pin the
# torch 2.4.1 cu121 build + an HF stack version-matched to torch 2.4 (the
# preinstalled transformers targets the newer torch and breaks after the
# downgrade - the ModuleNotFound cluster seen in the gpu-verify kernel).
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.4.1",
                "--index-url", "https://download.pytorch.org/whl/cu121"],
               check=True, timeout=900)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers==4.46.3", "datasets==3.2.0", "tokenizers>=0.20",
                "einops"], check=True, timeout=600)

import torch  # noqa: E402

assert torch.cuda.is_available(), "no CUDA"
print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}", flush=True)

print("\n=== E1 train: transformer, mt_lnn, mamba | seed 0 ===", flush=True)
r = subprocess.run(
    [sys.executable, "benchmarks/scaling_comparison.py",
     "--mode", "train", "--steps", "2000", "--seeds", "0",
     "--archs", "transformer,mt_lnn,mamba",
     "--out_dir", OUT],
    timeout=8 * 3600,  # leave margin inside the 9h GPU session
)
print(f"harness exit code: {r.returncode}", flush=True)

print("\n=== results ===", flush=True)
if os.path.isdir(OUT):
    for f in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, f)
        if f.endswith(".json"):
            with open(p) as fh:
                print(f, "->", json.dumps(json.load(fh))[:600], flush=True)
sys.exit(r.returncode)
