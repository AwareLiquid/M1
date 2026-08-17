"""
benchmarks/long_context.py — Long-context Selective Copy sweep.

Tests whether MT-LNN's temporal-recurrence advantage *grows* with sequence
length. We sweep T_noise across multiple values and train each of:

    Vanilla Transformer  (baseline)
    LNN (CfLTC FFN)      (baseline — isolates "liquid" without microtubule)
    MT-LNN (with pscan)  (ours)

at each length with matched ~200K params and identical recipes.

Expected behaviour:
  - Transformer: should hold ~similar held-out token-acc as T grows because
    attention is content-aware over the whole sequence, but sequence-exact
    will likely degrade as the search becomes harder.
  - LNN baseline: limited by attention since the LTC layer here has no
    real recurrence; should also struggle at length.
  - MT-LNN: parallel-scan recurrence should let it accumulate the K_mem
    "memorable" tokens compactly inside the protofilament state, so
    held-out accuracy at long T should *not* degrade much.

If MT-LNN's advantage SHRINKS at long T, the temporal-state hypothesis is
weak. If it GROWS, the project's central claim is real.

Output: a Markdown table you can paste into BENCHMARKS.md, plus a JSON of
the raw numbers in `benchmarks/long_context_results.json`.
"""

import json
import os
import sys
import time
import warnings

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn import MTLNNConfig, MTLNNModel
from mt_lnn.utils import make_param_groups
from benchmarks.selective_copy import (
    SelectiveCopyConfig, make_selective_copy_batch, evaluate_selective_copy,
)
from benchmarks.baselines import (
    BaselineConfig, SimpleCausalTransformer, SimpleCausalLNN,
)


def build_models(task: SelectiveCopyConfig, device: str):
    """Three models, all ~200K params, matched to the task's seq length."""
    transformer_cfg = BaselineConfig(
        vocab_size=task.vocab_size, max_seq_len=task.T_total,
        d_model=104, n_layers=2, n_heads=4, d_ff=256, dropout=0.0,
    )
    lnn_cfg = BaselineConfig(
        vocab_size=task.vocab_size, max_seq_len=task.T_total,
        d_model=104, n_layers=2, n_heads=4, d_ff=256, dropout=0.0,
    )
    mt_cfg = MTLNNConfig(
        vocab_size=task.vocab_size, max_seq_len=task.T_total,
        d_model=104, n_layers=2, n_heads=4, n_kv_heads=2, d_head=26,
        dropout=0.0, attention_dropout=0.0,
        gwtb_compression_ratio=4, gwtb_n_heads=2, coherence_heads=2,
    )
    return {
        "Transformer": SimpleCausalTransformer(transformer_cfg).to(device),
        "LNN":         SimpleCausalLNN(lnn_cfg).to(device),
        "MT-LNN":      MTLNNModel(mt_cfg).to(device),
    }


def train_model(model, task: SelectiveCopyConfig, label: str,
                 device: str = "cpu", verbose: bool = False) -> dict:
    model.train()
    is_mt = isinstance(model, MTLNNModel)
    if is_mt:
        opt = torch.optim.AdamW(make_param_groups(model, task.lr),
                                 betas=(0.9, 0.95))
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=task.lr, betas=(0.9, 0.95))

    t0 = time.time()
    out = None
    for step in range(task.steps):
        ids, labels = make_selective_copy_batch(task, task.batch, device=device)
        opt.zero_grad()
        out = model(ids, labels=labels)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if verbose and ((step + 1) % task.log_every == 0 or step == 0):
            print(f"    {label} step {step+1:5d}  loss {out['loss'].item():.4f}")
    final_loss = out["loss"].item() if out is not None else float("nan")
    return {"train_time": time.time() - t0, "final_loss": final_loss}


def run_one_length(T_noise: int, steps: int, device: str) -> dict:
    """Train + eval all three models at this T_noise. Return per-model dict."""
    task = SelectiveCopyConfig(
        K_mem=4,
        T_noise=T_noise,
        vocab_size=16,
        batch=8 if T_noise > 96 else 16,    # smaller batch for longer seqs
        steps=steps,
        lr=3e-3,
        eval_batches=8,
        log_every=200,
    )
    print(f"\n{'='*64}")
    print(f" T_noise = {T_noise}  (T_total = {task.T_total})  steps = {steps}  batch = {task.batch}")
    print(f"{'='*64}")

    torch.manual_seed(0)
    models = build_models(task, device)
    results = {"T_noise": T_noise, "T_total": task.T_total,
                "steps": steps, "batch": task.batch, "models": {}}

    for name, model in models.items():
        torch.manual_seed(0)
        train_info = train_model(model, task, name, device=device)
        torch.manual_seed(42)
        eval_info = evaluate_selective_copy(model, task, device=device, n_batches=8)
        results["models"][name] = {
            "tok_acc":    eval_info["token_accuracy"],
            "seq_exact":  eval_info["sequence_exact"],
            "train_time": train_info["train_time"],
            "final_loss": train_info["final_loss"],
        }
        print(f"  {name:<12s}  tok {eval_info['token_accuracy']:.3f}  "
              f"seq {eval_info['sequence_exact']:.3f}  "
              f"final-loss {train_info['final_loss']:.4f}  "
              f"time {train_info['train_time']:.0f}s")
    return results


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", action="store_true",
                    help="step-budget ablation at long T: is long-sequence "
                         "collapse undertraining or an architectural ceiling? "
                         "Runs each --lens at each --steps_list; if seq-exact "
                         "keeps climbing with steps it was undertraining, if it "
                         "plateaus it is a capacity ceiling.")
    ap.add_argument("--lens", default="224,480", help="T_noise values (sweep)")
    ap.add_argument("--steps_list", default="1500,3000,6000")
    ap.add_argument("--out", default="benchmarks/long_context_results.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Long-context Selective Copy sweep")
    print(f"Device: {device}")

    if args.sweep:
        lens = [int(x) for x in args.lens.split(",")]
        steps_list = [int(x) for x in args.steps_list.split(",")]
        schedule = [(T, s) for T in lens for s in steps_list]
        out_dir = os.path.splitext(args.out)[0] + "_sweep"
        os.makedirs(out_dir, exist_ok=True)
    else:
        # T_total = T_noise + 1 + K_mem(=4). Equal, adequate budget per length.
        schedule = [(32, 1500), (96, 1500), (224, 1500)]
        out_dir = None

    all_results = []
    t0 = time.time()
    for T_noise, steps in schedule:
        # Resume-safe in sweep mode: each (T,steps) point cached to its own file.
        pt = os.path.join(out_dir, f"T{T_noise}_s{steps}.json") if out_dir else None
        if pt and os.path.exists(pt):
            all_results.append(json.load(open(pt)))
            print(f"[skip] T_noise={T_noise} steps={steps}", flush=True)
            continue
        r = run_one_length(T_noise, steps, device)
        all_results.append(r)
        if pt:
            json.dump(r, open(pt, "w"), indent=2)

    total = time.time() - t0
    print(f"\nTotal wall-clock: {total:.0f}s")

    # Save raw JSON
    os.makedirs("benchmarks", exist_ok=True)
    out_path = args.out
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved raw results -> {out_path}")

    if args.sweep:
        print("\n" + "=" * 64)
        print(" ABLATION: seq-exact vs step budget (climbing=undertrain, flat=ceiling)")
        print("=" * 64)
        by_t = {}
        for r in all_results:
            by_t.setdefault(r["T_total"], []).append(r)
        for T_total, rs in sorted(by_t.items()):
            rs.sort(key=lambda x: x["steps"])
            print(f"\n T_total={T_total}:")
            for m in ["Transformer", "LNN", "MT-LNN"]:
                cells = "  ".join(f"{x['steps']}s:{x['models'][m]['seq_exact']:.3f}"
                                  for x in rs)
                print(f"   {m:<12} {cells}")

    # Print summary table
    print("\n" + "=" * 64)
    print(" Summary: held-out sequence-exact accuracy vs T_total")
    print("=" * 64)
    # steps column so sweep mode (multiple step budgets share one T_total)
    # never prints ambiguous duplicate-labelled rows.
    header = f"{'T_total':>8s}  {'steps':>7s}  " + "  ".join(f"{m:>14s}" for m in ["Transformer", "LNN", "MT-LNN"])
    print(header)
    print(f"{'-'*8:>8s}  {'-'*7:>7s}  " + "  ".join(f"{'-'*14:>14s}" for _ in range(3)))
    for r in all_results:
        row = f"{r['T_total']:>8d}  {r['steps']:>7d}  "
        row += "  ".join(f"{r['models'][m]['seq_exact']:>14.3f}" for m in ["Transformer", "LNN", "MT-LNN"])
        print(row)

    print("\n" + "=" * 64)
    print(" Summary: held-out token accuracy vs T_total")
    print("=" * 64)
    print(header)
    print(f"{'-'*8:>8s}  {'-'*7:>7s}  " + "  ".join(f"{'-'*14:>14s}" for _ in range(3)))
    for r in all_results:
        row = f"{r['T_total']:>8d}  {r['steps']:>7d}  "
        row += "  ".join(f"{r['models'][m]['tok_acc']:>14.3f}" for m in ["Transformer", "LNN", "MT-LNN"])
        print(row)

    print()


if __name__ == "__main__":
    main()
