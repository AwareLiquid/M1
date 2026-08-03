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
from benchmarks.reasoning_tasks import make_generator, gen_pointer_chase

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
def evaluate(model, gen, rng, device, batches=20, batch=256, fwd_kwargs=None):
    fwd_kwargs = fwd_kwargs or {}
    model.eval()
    correct = total = 0
    for _ in range(batches):
        ids, labels, ans_pos = make_lm_batch(gen, batch, rng, device)
        logits = model(ids, **fwd_kwargs)["logits"]
        # internal shift: answer at ans_pos is predicted from logits[ans_pos-1]
        pred = logits[:, ans_pos - 1, :].argmax(-1)
        correct += (pred == labels[:, ans_pos]).sum().item()
        total += batch
    model.train()
    return correct / total


# ── models ───────────────────────────────────────────────────────────────────

def build_mtlnn(vocab, seq_len, max_depth, seed, d_model=104, n_layers=2,
                gamma_init=None, full_mha=False, n_global_heads=0,
                n_heads=None, n_kv_heads=None):
    torch.manual_seed(seed)
    kw = {"n_global_heads": n_global_heads}
    if gamma_init is not None:
        kw["gamma_init"] = gamma_init  # GTP distance-decay ablation knob
    # Head-decoupling sweep knobs (ABLATIONS.md "Design-coupling audit").
    # None keeps the historical probe shape -- 4 heads, GQA 2:1 (or full MHA
    # via --full_mha) -- so existing command lines reproduce bit-identically.
    H = 4 if n_heads is None else int(n_heads)
    if d_model % H != 0:
        raise SystemExit(f"--n_heads {H} does not divide d_model {d_model}")
    if (d_model // H) % 2 != 0:
        raise SystemExit(f"--n_heads {H} gives odd d_head {d_model//H}; "
                         f"RoPE needs an even d_head")
    KV = (H if full_mha else max(1, H // 2)) if n_kv_heads is None else int(n_kv_heads)
    if H % KV != 0:
        raise SystemExit(f"--n_kv_heads {KV} does not divide n_heads {H}")
    cfg = MTLNNConfig(
        vocab_size=vocab,
        max_seq_len=seq_len,
        d_model=d_model,          # 104 = 13 protofilaments × 8
        n_layers=n_layers,        # shallow on purpose: depth comes from iteration
        n_heads=H,
        n_kv_heads=KV,
        d_head=d_model // H,
        dropout=0.0,
        attention_dropout=0.0,
        gwtb_n_heads=1,           # d_gw = d_model//8 = 13 → single-head workspace
        core_iterations=max_depth,
        **kw,
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
                depth_choices=None, log_every=200, fwd_kwargs=None,
                depth_setter="core"):
    """depth_choices: list of ints to sample per step (MT-LNN), or None.
    depth_setter: 'core' (LNN sub-layer iteration) or 'stack' (whole-block)."""
    fwd_kwargs = fwd_kwargs or {}
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    rng = np.random.default_rng(seed)
    depth_rng = np.random.default_rng(seed + 1)

    for step in range(steps):
        if depth_choices is not None:
            d = int(depth_rng.choice(depth_choices))
            if depth_setter == "stack":
                model.set_stack_iterations(d)
            elif depth_setter == "workspace":
                model.set_workspace_iterations(d)
            else:
                model.set_core_iterations(d)
        ids, labels, _ = make_lm_batch(gen, batch, rng, device)
        out = model(ids, labels=labels, **fwd_kwargs)
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

def make_mix_generator(difficulty, n_values):
    """Curriculum-style mixture: each batch draws hop count k ~ U{1..difficulty}
    (k is announced in the input tokens). Composition tasks grok much faster
    when the model can learn 1-hop lookup first and bootstrap deeper hops —
    fixed-k training left everyone (incl. transformer) on a shared plateau at
    k=4 (P0-C′ round 1). Sequence length is constant since only the k token
    varies."""
    def gen(batch, rng):
        k = int(rng.integers(1, difficulty + 1))
        return gen_pointer_chase(batch, n_values, k, rng)
    return gen


@torch.no_grad()
def evaluate_per_k(model, difficulty, n_values, rng, device,
                   batches=8, batch=256, fwd_kwargs=None):
    """Accuracy at EACH hop count 1..difficulty (mixture-trained models)."""
    fwd_kwargs = fwd_kwargs or {}
    model.eval()
    accs = {}
    for k in range(1, difficulty + 1):
        correct = total = 0
        for _ in range(batches):
            b = gen_pointer_chase(batch, n_values, k, rng)
            ids = torch.from_numpy(b.tokens).to(device)
            ans = torch.from_numpy(b.answer).to(device)
            logits = model(ids, **fwd_kwargs)["logits"]
            pred = logits[:, b.ans_pos - 1, :].argmax(-1)
            correct += (pred == ans).sum().item()
            total += batch
        accs[k] = correct / total
    model.train()
    return accs


def run_fixed_sweep(task, difficulty, n_values, seeds, steps, batch, lr,
                    depths, device, tag="", n_layers=2, no_scan=False,
                    skip_transformer=False, gamma_init=None, full_mha=False,
                    depth_setter="core", mix=False, n_global_heads=0,
                    n_heads=None, n_kv_heads=None):
    """HRM-style claim: train a FRESH model at each fixed depth d, evaluate at
    that same d. Same parameter count across depths (weight-tied iteration) —
    if accuracy climbs with d, extra latent iterations buy real capability.
    This avoids the anytime-training failure mode where random-depth sampling
    teaches the model to be depth-INVARIANT (ignore iterations)."""
    if mix:
        assert task == "pointer_chase", "--mix only supports pointer_chase"
        gen = make_mix_generator(difficulty, n_values)
        _, vocab, _ = make_generator(task, difficulty, n_values, seed=0)
    else:
        gen, vocab, _ = make_generator(task, difficulty, n_values, seed=0)
    probe = gen(1, np.random.default_rng(0))
    seq_len = probe.tokens.shape[1]
    print(f"== FIXED sweep {task} difficulty={difficulty} n_values={n_values} "
          f"T={seq_len} vocab={vocab} device={device} depths={depths} "
          f"mix={mix} ==")

    fwd_kwargs = {"use_lnn_recurrence": False} if no_scan else None
    rows = []
    for seed in seeds:
        t0 = time.time()
        accs = {}
        for d in depths:
            m = build_mtlnn(vocab, seq_len,
                            2 if depth_setter == "core" else 1,
                            seed, n_layers=n_layers, gamma_init=gamma_init,
                            full_mha=full_mha,
                            n_global_heads=n_global_heads,
                            n_heads=n_heads, n_kv_heads=n_kv_heads)
            n_params = m.get_num_params()
            if depth_setter == "stack":
                m.set_stack_iterations(d)
            elif depth_setter == "workspace":
                m.set_workspace_iterations(d)
            else:
                m.set_core_iterations(d)
            print(f"  [seed {seed}] mt_lnn {depth_setter}-depth={d} fixed "
                  f"({n_params/1e3:.0f}K params, n_layers={n_layers}, "
                  f"scan={'off' if no_scan else 'on'})")
            train_model(m, gen, device, steps, batch, lr, seed,
                        depth_choices=[d], fwd_kwargs=fwd_kwargs,
                        depth_setter=depth_setter)
            if mix:
                acc = evaluate_per_k(m, difficulty, n_values,
                                     np.random.default_rng(10_000 + seed),
                                     device, fwd_kwargs=fwd_kwargs)
                print(f"    eval depth {d}: " +
                      "  ".join(f"k={k}:{v:.3f}" for k, v in acc.items()))
            else:
                acc = evaluate(m, gen, np.random.default_rng(10_000 + seed),
                               device, fwd_kwargs=fwd_kwargs)
                print(f"    eval depth {d}: acc {acc:.4f}")
            accs[d] = acc
            del m
            if device == "cuda":
                torch.cuda.empty_cache()

        tr_acc, tr_params = None, None
        if not skip_transformer:
            tr = build_transformer(vocab, seq_len, seed)
            tr_params = tr.get_num_params()
            print(f"  [seed {seed}] transformer {tr_params/1e3:.0f}K params")
            train_model(tr, gen, device, steps, batch, lr, seed)
            if mix:
                tr_acc = evaluate_per_k(tr, difficulty, n_values,
                                        np.random.default_rng(10_000 + seed),
                                        device)
                print("    eval: " +
                      "  ".join(f"k={k}:{v:.3f}" for k, v in tr_acc.items()))
            else:
                tr_acc = evaluate(tr, gen, np.random.default_rng(10_000 + seed),
                                  device)
                print(f"    eval: acc {tr_acc:.4f}")
            del tr
            if device == "cuda":
                torch.cuda.empty_cache()

        rows.append({
            "mode": "fixed_sweep",
            "task": task, "difficulty": difficulty, "n_values": n_values,
            "seq_len": seq_len, "seed": seed, "steps": steps, "batch": batch,
            "lr": lr, "n_layers": n_layers, "no_scan": no_scan,
            "gamma_init": gamma_init, "full_mha": full_mha,
            "depth_setter": depth_setter, "mix": mix,
            "n_global_heads": n_global_heads,
            "mtlnn_params": n_params, "transformer_params": tr_params,
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
        if mix:
            ks = sorted(vals[0].keys())
            line = "  ".join(
                f"k={k}:{np.mean([v[k] for v in vals]):.3f}" for k in ks)
            print(f"  mt_lnn depth {d}: {line}")
        else:
            print(f"  mt_lnn depth {d}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    tvals = [r["transformer_acc"] for r in rows if r["transformer_acc"] is not None]
    if tvals:
        if mix:
            ks = sorted(tvals[0].keys())
            line = "  ".join(
                f"k={k}:{np.mean([v[k] for v in tvals]):.3f}" for k in ks)
            print(f"  transformer   : {line}")
        else:
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
    p.add_argument("--task",
                   choices=["pointer_chase", "mod_chain", "parity", "s5_word"],
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
    p.add_argument("--n_layers", type=int, default=2,
                   help="MT-LNN block count (ablation knob)")
    p.add_argument("--no_scan", action="store_true",
                   help="use_lnn_recurrence=False: liquid recurrence off, LNN "
                        "degenerates to a gated FFN (isolates whether the "
                        "recurrent scan impedes in-context lookup learning)")
    p.add_argument("--skip_transformer", action="store_true")
    p.add_argument("--gamma_init", type=float, default=None,
                   help="override GTP distance-decay init (default cfg 0.1; "
                        "0.001 makes all heads effectively global)")
    p.add_argument("--full_mha", action="store_true",
                   help="n_kv_heads=n_heads (disable GQA)")
    p.add_argument("--stack", action="store_true",
                   help="depth knob = stack_iterations (whole block stack, "
                        "attention included) instead of core_iterations "
                        "(LNN sub-layer only)")
    p.add_argument("--workspace", action="store_true",
                   help="depth knob = workspace_iterations (J-Space J1: "
                        "GWTB reverberation passes, the cheapest depth)")
    p.add_argument("--mix", action="store_true",
                   help="curriculum mixture: train on k ~ U{1..difficulty} "
                        "(pointer_chase only), evaluate per-k")
    p.add_argument("--n_global_heads", type=int, default=0,
                   help="reserve N truly-global heads per layer (架构原则#1); "
                        "0 = historical all-decaying init")
    p.add_argument("--n_heads", type=int, default=None,
                   help="attention heads, decoupled from the 13 protofilaments "
                        "(ABLATIONS.md design-coupling audit); must divide "
                        "d_model. Default: historical probe shape (4)")
    p.add_argument("--n_kv_heads", type=int, default=None,
                   help="KV heads for GQA; must divide n_heads. Default: "
                        "n_heads/2, or n_heads under --full_mha. Middle "
                        "ratios like 4:1 are the point of the sweep")
    args = p.parse_args()

    if args.n_values is None:
        args.n_values = {"pointer_chase": 16, "mod_chain": 10,
                         "parity": 2, "s5_word": 120}[args.task]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        run(args.task, difficulty=2, n_values=8, seeds=[0], steps=30,
            batch=32, lr=3e-4, max_depth=2, eval_depths=[1, 2],
            device=device, tag="smoke")
        return

    if args.mode == "fixed":
        run_fixed_sweep(args.task, args.difficulty, args.n_values, args.seeds,
                        args.steps, args.batch, args.lr, args.eval_depths,
                        device, tag=args.tag, n_layers=args.n_layers,
                        no_scan=args.no_scan,
                        skip_transformer=args.skip_transformer,
                        gamma_init=args.gamma_init, full_mha=args.full_mha,
                        depth_setter=("stack" if args.stack else
                                      "workspace" if args.workspace else "core"),
                        mix=args.mix, n_global_heads=args.n_global_heads,
                        n_heads=args.n_heads, n_kv_heads=args.n_kv_heads)
    else:
        run(args.task, args.difficulty, args.n_values, args.seeds, args.steps,
            args.batch, args.lr, args.max_depth, args.eval_depths, device,
            tag=args.tag)


if __name__ == "__main__":
    main()
