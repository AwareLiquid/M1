"""P0 depth study: does "thinking longer" make MT-LNN more accurate?

(docs/ROADMAP_M2.md §4 P0-B/P0-C — the minimal evidence for the M2 thesis.)

Protocol
--------
1. Train a SHALLOW MT-LNN (n_layers=2, so layer count doesn't confound depth)
   with latent recurrent depth enabled (config.core_iterations = max depth).
   Each training step samples a random depth in [1, max_depth]
   (Geiping-style depth randomization) so the weights work at every depth.
2. Evaluate the SAME weights at depth ∈ {1, 2, 4, 8} on held-out data.
   Thesis holds if accuracy climbs with depth on depth-sensitive tasks.
3. A ModernCausalTransformer of matched size trains on the same data as the
   fixed-depth control (it has no depth knob — one line per model).

Tasks (see reasoning_tasks.py): pointer_chase (k-hop composition — the clean
depth probe) and mod_chain (sequential accumulate).

Loss/accuracy only at the answer position: labels are -100 everywhere except
ans_pos, and both models shift internally (logits[:-1] vs labels[1:]), so the
answer is predicted at the [THINK] token position.

Run (RTX 5060 8GB, each config trains in minutes):
  py -3.11 benchmarks/reasoning_depth.py --task pointer_chase --difficulty 4
  py -3.11 benchmarks/reasoning_depth.py --task mod_chain --difficulty 8
  py -3.11 benchmarks/reasoning_depth.py --smoke        # 1-minute CPU sanity
Results append to benchmarks/results/reasoning_depth.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mt_lnn import MTLNNConfig, MTLNNModel
from benchmarks.baselines import BaselineConfig, ModernCausalTransformer
from benchmarks.reasoning_tasks import make_generator

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "reasoning_depth.jsonl")


# ── batches ──────────────────────────────────────────────────────────────────

def make_lm_batch(gen, batch, rng, device):
    """Returns (input_ids, labels) with loss masked to the answer position."""
    b = gen(batch, rng)
    ids = torch.from_numpy(b.tokens).to(device)
    labels = torch.full_like(ids, -100)
    labels[:, b.ans_pos] = torch.from_numpy(b.answer).to(device)
    return ids, labels, b.ans_pos


@torch.no_grad()
def evaluate(model, gen, rng, device, batches=20, batch=256):
    model.eval()
    correct = total = 0
    for _ in range(batches):
        ids, labels, ans_pos = make_lm_batch(gen, batch, rng, device)
        logits = model(ids)["logits"]
        # internal shift: answer at ans_pos is predicted from logits[ans_pos-1]
        pred = logits[:, ans_pos - 1, :].argmax(-1)
        correct += (pred == labels[:, ans_pos]).sum().item()
        total += batch
    model.train()
    return correct / total


# ── models ───────────────────────────────────────────────────────────────────

def build_mtlnn(vocab, seq_len, max_depth, seed, d_model=104, n_layers=2):
    torch.manual_seed(seed)
    cfg = MTLNNConfig(
        vocab_size=vocab,
        max_seq_len=seq_len,
        d_model=d_model,          # 104 = 13 protofilaments × 8
        n_layers=n_layers,        # shallow on purpose: depth comes from iteration
        n_heads=4,
        n_kv_heads=2,
        d_head=d_model // 4,
        dropout=0.0,
        attention_dropout=0.0,
        gwtb_n_heads=1,           # d_gw = d_model//8 = 13 → single-head workspace
        core_iterations=max_depth,
    )
    return MTLNNModel(cfg)


def build_transformer(vocab, seq_len, seed, d_model=104, n_layers=2):
    torch.manual_seed(seed)
    cfg = BaselineConfig(
        vocab_size=vocab, max_seq_len=seq_len,
        d_model=d_model, n_layers=n_layers, n_heads=4, d_ff=256,
    )
    return ModernCausalTransformer(cfg)


# ── training ─────────────────────────────────────────────────────────────────

def train_model(model, gen, device, steps, batch, lr, seed,
                depth_choices=None, log_every=200):
    """depth_choices: list of ints to sample per step (MT-LNN), or None."""
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    rng = np.random.default_rng(seed)
    depth_rng = np.random.default_rng(seed + 1)

    for step in range(steps):
        if depth_choices is not None:
            model.set_core_iterations(int(depth_rng.choice(depth_choices)))
        ids, labels, _ = make_lm_batch(gen, batch, rng, device)
        out = model(ids, labels=labels)
        loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps - 1:
            print(f"    step {step:5d}  loss {loss.item():.4f}", flush=True)
    return model


# ── experiment ───────────────────────────────────────────────────────────────

def run_fixed_sweep(task, difficulty, n_values, seeds, steps, batch, lr,
                    depths, device, tag=""):
    """HRM-style claim: train a FRESH model at each fixed depth d, evaluate at
    that same d. Same parameter count across depths (weight-tied iteration) —
    if accuracy climbs with d, extra latent iterations buy real capability.
    This avoids the anytime-training failure mode where random-depth sampling
    teaches the model to be depth-INVARIANT (ignore iterations)."""
    gen, vocab, _ = make_generator(task, difficulty, n_values, seed=0)
    probe = gen(1, np.random.default_rng(0))
    seq_len = probe.tokens.shape[1]
    print(f"== FIXED sweep {task} difficulty={difficulty} n_values={n_values} "
          f"T={seq_len} vocab={vocab} device={device} depths={depths} ==")

    rows = []
    for seed in seeds:
        t0 = time.time()
        accs, final_losses = {}, {}
        for d in depths:
            m = build_mtlnn(vocab, seq_len, max(d, 2), seed)  # gate exists even for d=1
            n_params = m.get_num_params()
            m.set_core_iterations(d)
            print(f"  [seed {seed}] mt_lnn depth={d} fixed "
                  f"({n_params/1e3:.0f}K params)")
            train_model(m, gen, device, steps, batch, lr, seed,
                        depth_choices=[d])
            acc = evaluate(m, gen, np.random.default_rng(10_000 + seed), device)
            accs[d] = acc
            print(f"    eval depth {d}: acc {acc:.4f}")
            del m
            if device == "cuda":
                torch.cuda.empty_cache()

        tr = build_transformer(vocab, seq_len, seed)
        tr_params = tr.get_num_params()
        print(f"  [seed {seed}] transformer {tr_params/1e3:.0f}K params")
        train_model(tr, gen, device, steps, batch, lr, seed)
        tr_acc = evaluate(tr, gen, np.random.default_rng(10_000 + seed), device)
        print(f"    eval: acc {tr_acc:.4f}")
        del tr
        if device == "cuda":
            torch.cuda.empty_cache()

        rows.append({
            "mode": "fixed_sweep",
            "task": task, "difficulty": difficulty, "n_values": n_values,
            "seq_len": seq_len, "seed": seed, "steps": steps, "batch": batch,
            "lr": lr, "mtlnn_params": n_params, "transformer_params": tr_params,
            "mtlnn_acc_by_depth": accs, "transformer_acc": tr_acc,
            "wall_s": round(time.time() - t0, 1), "tag": tag,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print("\n== summary (mean over seeds, each depth = fresh fixed-depth model) ==")
    for d in depths:
        vals = [r["mtlnn_acc_by_depth"][d] for r in rows]
        print(f"  mt_lnn depth {d}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    tvals = [r["transformer_acc"] for r in rows]
    print(f"  transformer   : {np.mean(tvals):.4f} ± {np.std(tvals):.4f}")
    print(f"\nresults appended to {RESULTS}")
    return rows


def run(task, difficulty, n_values, seeds, steps, batch, lr, max_depth,
        eval_depths, device, tag=""):
    gen, vocab, _ = make_generator(task, difficulty, n_values, seed=0)
    # probe seq_len from one sample
    probe = gen(1, np.random.default_rng(0))
    seq_len = probe.tokens.shape[1]
    print(f"== {task} difficulty={difficulty} n_values={n_values} "
          f"T={seq_len} vocab={vocab} device={device} ==")

    rows = []
    for seed in seeds:
        t0 = time.time()
        # MT-LNN with randomized-depth training
        m = build_mtlnn(vocab, seq_len, max_depth, seed)
        n_params = m.get_num_params()
        print(f"  [seed {seed}] mt_lnn {n_params/1e3:.0f}K params, "
              f"train depths 1..{max_depth}")
        train_model(m, gen, device, steps, batch, lr, seed,
                    depth_choices=list(range(1, max_depth + 1)))
        accs = {}
        for d in eval_depths:
            m.set_core_iterations(d)
            acc = evaluate(m, gen, np.random.default_rng(10_000 + seed), device)
            accs[d] = acc
            print(f"    eval depth {d}: acc {acc:.4f}")
        del m
        if device == "cuda":
            torch.cuda.empty_cache()

        # transformer control (no depth knob)
        tr = build_transformer(vocab, seq_len, seed)
        tr_params = tr.get_num_params()
        print(f"  [seed {seed}] transformer {tr_params/1e3:.0f}K params")
        train_model(tr, gen, device, steps, batch, lr, seed)
        tr_acc = evaluate(tr, gen, np.random.default_rng(10_000 + seed), device)
        print(f"    eval: acc {tr_acc:.4f}")
        del tr
        if device == "cuda":
            torch.cuda.empty_cache()

        rows.append({
            "task": task, "difficulty": difficulty, "n_values": n_values,
            "seq_len": seq_len, "seed": seed, "steps": steps, "batch": batch,
            "lr": lr, "max_train_depth": max_depth,
            "mtlnn_params": n_params, "transformer_params": tr_params,
            "mtlnn_acc_by_depth": accs, "transformer_acc": tr_acc,
            "wall_s": round(time.time() - t0, 1), "tag": tag,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # summary across seeds
    print("\n== summary (mean over seeds) ==")
    for d in eval_depths:
        vals = [r["mtlnn_acc_by_depth"][d] for r in rows]
        print(f"  mt_lnn depth {d}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    tvals = [r["transformer_acc"] for r in rows]
    print(f"  transformer   : {np.mean(tvals):.4f} ± {np.std(tvals):.4f}")
    print(f"\nresults appended to {RESULTS}")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["pointer_chase", "mod_chain"],
                   default="pointer_chase")
    p.add_argument("--difficulty", type=int, default=4,
                   help="k_hops (pointer_chase) or k_terms (mod_chain)")
    p.add_argument("--n_values", type=int, default=None,
                   help="n_nodes (pointer_chase, default 16) or modulus "
                        "(mod_chain, default 10)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_depth", type=int, default=8)
    p.add_argument("--eval_depths", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--tag", default="")
    p.add_argument("--mode", choices=["anytime", "fixed"], default="fixed",
                   help="fixed: fresh model per depth, trained AND evaluated "
                        "at that depth (HRM-style, the primary P0 claim). "
                        "anytime: one model, randomized-depth training "
                        "(known failure mode: learns depth-invariance)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny CPU run to sanity-check the pipeline")
    args = p.parse_args()

    if args.n_values is None:
        args.n_values = 16 if args.task == "pointer_chase" else 10

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        run(args.task, difficulty=2, n_values=8, seeds=[0], steps=30,
            batch=32, lr=3e-4, max_depth=2, eval_depths=[1, 2],
            device=device, tag="smoke")
        return

    if args.mode == "fixed":
        run_fixed_sweep(args.task, args.difficulty, args.n_values, args.seeds,
                        args.steps, args.batch, args.lr, args.eval_depths,
                        device, tag=args.tag)
    else:
        run(args.task, args.difficulty, args.n_values, args.seeds, args.steps,
            args.batch, args.lr, args.max_depth, args.eval_depths, device,
            tag=args.tag)


if __name__ == "__main__":
    main()
