"""Parity length-generalization study: algorithm vs lookup (M2 separation).

Salvaged Kaggle data (2026-08-05, ABLATIONS.md) showed curriculum-mix parity
largely learnable for BOTH selective and stock arms up to L=16-32 — because
parity ∈ TC⁰ and the ATTENTION side can count bits and read out mod 2,
bypassing the liquid core entirely. The clean discriminator is LENGTH
EXTRAPOLATION: train on L ~ U{1..32}, evaluate at L ∈ {48, 64, 96, 128}.
An attention counting shortcut degrades out-of-length; a genuine flip-flop
recurrence (which only the selective, input-dependent-λ core can express)
generalizes indefinitely.

Prediction (falsifiable, both directions):
  selective arm >> stock arm at L >= 48, with stock → chance.
If selective ALSO collapses out-of-length, it learned the shortcut too and
the core's contribution is not established.

Run:  py -3.11 benchmarks/parity_lengthgen.py --seeds 0 1 2 --steps 30000
Results append to benchmarks/results/parity_lengthgen.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mt_lnn import MTLNNConfig, MTLNNModel
from benchmarks.reasoning_tasks import gen_parity, vocab_size

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "parity_lengthgen.jsonl")

TRAIN_MAX = 32
EVAL_IN = (8, 16, 32)
EVAL_OUT = (48, 64, 96, 128)


def build(seed, selective, max_len):
    torch.manual_seed(seed)
    cfg = MTLNNConfig(
        vocab_size=vocab_size(2), max_seq_len=max_len, d_model=104,
        n_layers=2, n_heads=4, n_kv_heads=2, d_head=26,
        dropout=0.0, attention_dropout=0.0, gwtb_n_heads=1,
        selective_decay=selective,
    )
    return MTLNNModel(cfg)


def run_arm(tag, seed, selective, steps, batch, lr, device,
            beta2=0.999, clip=0.0):
    """beta2/clip defaults are the GOOD recipe (ABLATIONS 2026-08-05 bisect:
    beta2=0.95 and clip=1.0 each independently veto the parity breakthrough —
    the 2026-08-05 overnight run used the vetoing recipe and its extrapolation
    collapse is therefore not interpretable)."""
    max_len = 1 + max(EVAL_OUT) + 2  # sized for the LONGEST eval, not train
    m = build(seed, selective, max_len).to(device).train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, beta2),
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for step in range(steps):
        k = int(rng.integers(1, TRAIN_MAX + 1))
        b = gen_parity(batch, k, rng)
        ids = torch.from_numpy(b.tokens).to(device)
        labels = torch.full_like(ids, -100)
        labels[:, b.ans_pos] = torch.from_numpy(b.answer).to(device)
        loss = m(ids, labels=labels)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(m.parameters(), clip)
        opt.step()
        sched.step()
        if step % 2000 == 0 or step == steps - 1:
            print(f"    {tag} s{seed} step {step:6d} k={k:2d} "
                  f"loss {loss.item():.4f}", flush=True)

    m.eval()
    accs = {}
    ev = np.random.default_rng(10_000 + seed)
    with torch.no_grad():
        for k in (*EVAL_IN, *EVAL_OUT):
            c = t = 0
            for _ in range(8):
                b = gen_parity(256, k, ev)
                ids = torch.from_numpy(b.tokens).to(device)
                pred = m(ids)["logits"][:, b.ans_pos - 1].argmax(-1)
                c += (pred == torch.from_numpy(b.answer).to(device)).sum().item()
                t += 256
            accs[k] = round(c / t, 4)
    row = {
        "tag": tag, "seed": seed, "selective": selective, "steps": steps,
        "beta2": beta2, "clip": clip, "lr": lr,
        "train_max": TRAIN_MAX, "accs": accs,
        "wall_s": round(time.time() - t0, 1),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    in_l = "  ".join(f"L{k}:{accs[k]}" for k in EVAL_IN)
    out_l = "  ".join(f"L{k}:{accs[k]}" for k in EVAL_OUT)
    print(f"  {tag} s{seed}  in-dist [{in_l}]  EXTRAP [{out_l}]", flush=True)
    del m
    if device == "cuda":
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--clip", type=float, default=0.0)
    p.add_argument("--only", choices=["sel", "stock", "both"], default="both",
                   help="single-arm mode for per-process GPU isolation "
                        "(the 2026-08-05 run OOMed on arm 2 of a shared "
                        "process; one arm per process is clean)")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"parity length-gen: train L<=32, extrapolate to {EVAL_OUT}; "
          f"device={device} beta2={args.beta2} clip={args.clip}")
    # Interleave arms so every partial run yields comparable pairs.
    for seed in args.seeds:
        if args.only in ("sel", "both"):
            run_arm("lengen-sel", seed, True, args.steps, args.batch,
                    args.lr, device, args.beta2, args.clip)
        if args.only in ("stock", "both"):
            run_arm("lengen-stock", seed, False, args.steps, args.batch,
                    args.lr, device, args.beta2, args.clip)
    print("ALL DONE")


if __name__ == "__main__":
    main()
