"""P0-4: locate the root cause of MT-LNN's fp16-only training divergence.

Context. Under fp16 automatic mixed precision MT-LNN's loss goes non-finite
(historically ~step 629) while a matched Transformer trains stably under the
identical recipe; full fp32 does not diverge. That makes it a *numerics* bug,
not an architectural one -- but "use fp32" is not an acceptable answer for a
paper, so this script finds WHICH tensor overflows FIRST.

Method. Train mt_lnn exactly as `scaling_comparison.py --dtype fp16` does, but
with forward hooks on every leaf module. On the first step where any module
output (or the loss) becomes non-finite, report:

  * the first module in execution order whose OUTPUT is non-finite while all
    its INPUTS are still finite -- i.e. the operation that actually created the
    Inf/NaN rather than one that merely propagated it;
  * per-module activation absolute maxima over the preceding steps, so a
    tensor that is silently climbing toward fp16's 65504 ceiling is visible
    before it overflows;
  * GradScaler scale, grad-norm, and loss trajectory around the failure.

fp16 max is 65504: any activation whose |max| grows past ~1e4 is one step away
from overflow, which is the signature this script is built to surface.

Usage:
    py -3.11 benchmarks/diagnose_fp16_divergence.py --steps 800
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import datasets as _datasets  # noqa: F401  (Windows DLL-order guard)

from benchmarks.scaling_comparison import build, build_chunks

FP16_MAX = 65504.0


def _finite(t):
    return bool(torch.isfinite(t).all()) if torch.is_tensor(t) else True


def _all_finite(obj):
    """Tensors can arrive nested in tuples/dicts from module I/O."""
    if torch.is_tensor(obj):
        return _finite(obj)
    if isinstance(obj, (tuple, list)):
        return all(_all_finite(o) for o in obj)
    if isinstance(obj, dict):
        return all(_all_finite(v) for v in obj.values())
    return True


def _absmax(obj):
    """Largest |value| anywhere in a nested tensor structure (0.0 if none)."""
    if torch.is_tensor(obj) and obj.is_floating_point() and obj.numel():
        finite = obj[torch.isfinite(obj)]
        return float(finite.abs().max()) if finite.numel() else float("inf")
    if isinstance(obj, (tuple, list)):
        vals = [_absmax(o) for o in obj]
        return max(vals) if vals else 0.0
    if isinstance(obj, dict):
        vals = [_absmax(v) for v in obj.values()]
        return max(vals) if vals else 0.0
    return 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d_model", type=int, default=832)   # must be /13
    ap.add_argument("--n_layers", type=int, default=12)
    ap.add_argument("--vocab", type=int, default=50257)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--train_token_cap", type=int, default=20_000_000)
    ap.add_argument("--wikitext", default="wikitext-103-raw-v1")
    ap.add_argument("--top_k", type=int, default=12,
                    help="how many hottest modules to report")
    ap.add_argument("--out", default="benchmarks/fp16_divergence_report.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("fp16 AMP divergence only reproduces on CUDA; aborting.")
        return
    print(f"device={device} | reproducing mt_lnn fp16 divergence "
          f"({args.steps} steps, seed {args.seed})", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    train_c = build_chunks(tok, "train", args.seq_len, args.wikitext,
                           max_tokens=args.train_token_cap)

    g = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(train_c), generator=g)
    torch.manual_seed(args.seed)

    # fp16 AMP keeps fp32 master weights (GradScaler refuses fp16 grads), which
    # is exactly what scaling_comparison.py does -- mirror it so the repro is
    # faithful.
    m = build("mt_lnn", args.d_model, args.n_layers, args.vocab, args.seq_len,
              device, torch.float32)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda")
    m.train()

    # ---- instrumentation ----------------------------------------------------
    # exec_order is rebuilt every step: it is the forward execution order, which
    # is what makes "first module to produce a non-finite output" meaningful.
    state = {"exec_order": [], "culprit": None}
    peak = {}          # module name -> running max |activation| over all steps
    handles = []

    def make_hook(name):
        def hook(mod, inp, out):
            state["exec_order"].append(name)
            a = _absmax(out)
            if a > peak.get(name, 0.0):
                peak[name] = a
            if state["culprit"] is None and not _all_finite(out):
                # Inputs still finite => this module CREATED the non-finite
                # value. Inputs already bad => it only propagated it.
                state["culprit"] = {
                    "module": name,
                    "type": type(mod).__name__,
                    "inputs_finite": _all_finite(inp),
                    "peak_seen_here": peak.get(name, 0.0),
                }
        return hook

    for name, mod in m.named_modules():
        if not list(mod.children()):          # leaf modules only
            handles.append(mod.register_forward_hook(make_hook(name)))

    # ---- training loop ------------------------------------------------------
    history, step, failed_at = [], 0, None
    cursor = 0
    while step < args.steps and failed_at is None:
        if cursor + args.batch > len(order):
            cursor = 0
        sel = order[cursor:cursor + args.batch]
        cursor += args.batch
        ids = train_c[sel].to(device)

        state["exec_order"].clear()
        state["culprit"] = None

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = m(ids, labels=ids)
            loss = out["loss"]

        loss_finite = _finite(loss.detach())
        if state["culprit"] is not None or not loss_finite:
            failed_at = step
            gnorm = None
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0))
            scaler.step(opt)
            scaler.update()

        rec = {
            "step": step,
            "loss": float(loss.detach()) if loss_finite else None,
            "loss_finite": loss_finite,
            "scale": float(scaler.get_scale()),
            "grad_norm": gnorm,
            "max_activation": max(peak.values()) if peak else 0.0,
        }
        history.append(rec)
        if step % 50 == 0 or failed_at is not None:
            print(f"  step {step:4d} | loss "
                  f"{rec['loss'] if rec['loss'] is not None else float('nan'):.4f}"
                  f" | scale {rec['scale']:.0f} | gnorm "
                  f"{gnorm if gnorm is not None else float('nan'):.3f}"
                  f" | max|act| {rec['max_activation']:.1f}", flush=True)
        step += 1

    for h in handles:
        h.remove()

    # ---- report -------------------------------------------------------------
    hottest = sorted(peak.items(), key=lambda kv: kv[1], reverse=True)[:args.top_k]
    report = {
        "diverged": failed_at is not None,
        "failed_at_step": failed_at,
        "steps_run": step,
        "culprit": state["culprit"],
        "fp16_max": FP16_MAX,
        "hottest_modules": [{"module": n, "peak_abs_activation": v,
                             "headroom_ratio": (v / FP16_MAX) if v else 0.0}
                            for n, v in hottest],
        "history_tail": history[-25:],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    if failed_at is None:
        print(f"NO DIVERGENCE in {step} steps — fp16 issue did NOT reproduce "
              f"at this budget/seed.")
    else:
        print(f"DIVERGED at step {failed_at}")
        c = state["culprit"]
        if c:
            print(f"First non-finite OUTPUT: {c['module']}  ({c['type']})")
            print(f"  inputs still finite: {c['inputs_finite']}"
                  f"  -> {'THIS OP CREATED IT' if c['inputs_finite'] else 'propagated from upstream'}")
        else:
            print("Loss went non-finite but no module output did — suspect the "
                  "loss/reduction path itself.")
    print("\nHottest activations (fp16 ceiling 65504):")
    for n, v in hottest:
        flag = "  <-- near overflow" if v > FP16_MAX * 0.15 else ""
        print(f"  {v:12.1f}  {n}{flag}")
    print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
