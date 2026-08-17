"""
experiments/ablation_hebbian_lr.py — does the Hebbian co-activation term DO anything?

WHY THIS EXISTS
---------------
The Hebbian regularizer (mt_lnn/plasticity.py) adds L_hebb = -alpha * mean(co-activation)
to the LM loss. Two facts make its real effect ambiguous:
  1. Its *value* is ~1e-8 (alpha ~ hebbian_lr * sigmoid-gate, co-activation ~1e-4),
     ~9 orders of magnitude below the cross-entropy loss (~5). Looks dead.
  2. But the *gradient* it injects scales with alpha = hebbian_lr, NOT with the loss
     value. A tiny scalar can still move weights if its gradient is non-trivial.
"value small" != "gradient small", so we must measure EMPIRICALLY, not assume.

WHAT THIS MEASURES (per setting, per seed)
  - final held-out PPL  (does it help / hurt the LM objective?)
  - stability proxy: NaN count + final full-loss grad norm (collapse check)
  - |L_hebb| magnitude over training (is the term itself non-zero?)
  - **the Hebbian term's GRADIENT NORM** vs the full-loss gradient norm
    (the key diagnostic: what fraction of each weight update comes from Hebb?)

SWEEP: hebbian_lr in {off, 1e-4 (default), 1e-2, 1e-1}, seeds {0,1}, on REAL
WikiText-103 (gpt2 tokens) via train.py's BinDataset. Small model, short run:
this is a fast local screen (小规模端到端验证), NOT a full pretrain.

HONEST READ of outcomes:
  - if higher hebbian_lr gives a measurable, STABLE (no NaN, grad finite) PPL
    change AND a non-negligible gradient fraction -> Hebbian is a real knob, keep
    it in the full run and tune lr.
  - if PPL is flat within seed noise AND gradient fraction stays ~0 even at
    hebbian_lr=1e-1 -> the term is inert at these scales; turn it off (or move
    its strength up by orders of magnitude before it matters).

Run:  python -m experiments.ablation_hebbian_lr
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel
from train import BinDataset, evaluate  # faithful reuse of the real data reader + eval

DATA_DIR = os.path.join(_ROOT, "kaggle_out", "tiny", "data")

# Sweep: None == Hebbian OFF; the rest are use_hebbian=True at varying base_lr.
SETTINGS = [
    ("off", None),
    ("1e-4", 1e-4),
    ("1e-2", 1e-2),
    ("1e-1", 1e-1),
]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_model(hebbian_lr: Optional[float], vocab_size: int, *,
                d_model: int, n_layers: int, n_heads: int, seq_len: int,
                device: str) -> MTLNNModel:
    cfg = MTLNNConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=1,
        d_head=d_model // n_heads,
        max_seq_len=seq_len,
        dropout=0.0,
        gwtb_n_heads=1,
        # isolate Hebbian: every other v2.0 module OFF.
        use_competitive_gwtb=False,
        use_world_model=False,
        use_hebbian=(hebbian_lr is not None),
        hebbian_lr=(hebbian_lr if hebbian_lr is not None else 1e-4),
        hebbian_lavi_gate=True,
    )
    return MTLNNModel(cfg).to(device)


@torch.no_grad()
def _grad_global_norm(model: MTLNNModel) -> float:
    tot = 0.0
    for p in model.parameters():
        if p.grad is not None:
            tot += float(p.grad.detach().pow(2).sum().item())
    return math.sqrt(tot)


def measure_hebb_gradient(model: MTLNNModel, inp, lbl) -> Dict[str, float]:
    """Isolate the Hebbian term's gradient norm vs the full loss's gradient norm.

    Both are measured on the SAME batch so the ratio is meaningful: it answers
    'of the weight update this batch produces, what share is Hebbian?'.
    """
    model.train()

    # (1) full-loss gradient norm (includes Hebb, since forward adds it in train).
    model.zero_grad(set_to_none=True)
    out = model(inp, labels=lbl)
    full_loss = out["loss"]
    hebb_val = out.get("hebbian_loss")
    full_loss.backward(retain_graph=False)
    g_full = _grad_global_norm(model)

    # (2) Hebbian-only gradient norm: re-run forward to repopulate _hebb_signal,
    # then backward ONLY the Hebbian term.
    g_hebb = 0.0
    hebb_mag = 0.0
    if model.hebbian_reg is not None:
        model.zero_grad(set_to_none=True)
        _ = model(inp, labels=lbl)  # repopulate per-block _hebb_signal in the graph
        hebb_loss = model.hebbian_reg.compute_loss(model)
        if hebb_loss is not None:
            hebb_mag = abs(float(hebb_loss.detach().item()))
            hebb_loss.backward(retain_graph=False)
            g_hebb = _grad_global_norm(model)

    model.zero_grad(set_to_none=True)
    return {
        "g_full": g_full,
        "g_hebb": g_hebb,
        "hebb_frac": (g_hebb / g_full) if g_full > 0 else 0.0,
        "hebb_mag": hebb_mag if hebb_mag else (
            abs(float(hebb_val.item())) if hebb_val is not None else 0.0),
    }


def run_one(hebbian_lr: Optional[float], seed: int, *, steps: int, batch: int,
            seq_len: int, lr: float, d_model: int, n_layers: int, n_heads: int,
            measure_every: int, eval_batches: int, device: str) -> Dict:
    set_seed(seed)
    train_ds = BinDataset(os.path.join(DATA_DIR, "train.bin"), seq_len)
    val_ds = BinDataset(os.path.join(DATA_DIR, "validation.bin"), seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False,
                            drop_last=True, num_workers=0)

    vocab_size = json.load(open(os.path.join(DATA_DIR, "meta.json")))["vocab_size"]
    model = build_model(hebbian_lr, vocab_size=vocab_size,
                        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                        seq_len=seq_len, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            eps=1e-8, weight_decay=0.01)

    nan_count = 0
    hebb_mags: List[float] = []
    hebb_fracs: List[float] = []
    g_fulls: List[float] = []
    g_hebbs: List[float] = []
    last_grad_norm = 0.0

    model.train()
    step = 0
    t0 = time.time()
    while step < steps:
        for inp, lbl in train_loader:
            if step >= steps:
                break
            inp = inp.to(device)
            lbl = lbl.to(device)

            # Periodic isolated gradient diagnostic (separate from the real step).
            if step % measure_every == 0:
                d = measure_hebb_gradient(model, inp, lbl)
                g_fulls.append(d["g_full"])
                g_hebbs.append(d["g_hebb"])
                hebb_fracs.append(d["hebb_frac"])
                hebb_mags.append(d["hebb_mag"])

            # Real optimisation step.
            opt.zero_grad(set_to_none=True)
            out = model(inp, labels=lbl)
            loss = out["loss"]
            if not torch.isfinite(loss):
                nan_count += 1
                opt.zero_grad(set_to_none=True)
                step += 1
                continue
            loss.backward()
            last_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
            opt.step()
            step += 1

    final_ppl = evaluate(model, val_loader, device, max_batches=eval_batches)
    return {
        "final_ppl": round(final_ppl, 4),
        "nan_count": nan_count,
        "final_grad_norm": round(last_grad_norm, 5),
        "hebb_mag_mean": statistics.fmean(hebb_mags) if hebb_mags else 0.0,
        "hebb_frac_mean": statistics.fmean(hebb_fracs) if hebb_fracs else 0.0,
        "g_full_mean": statistics.fmean(g_fulls) if g_fulls else 0.0,
        "g_hebb_mean": statistics.fmean(g_hebbs) if g_hebbs else 0.0,
        "seconds": round(time.time() - t0, 1),
    }


def main(*, seeds=(0, 1), steps: int = 150, batch: int = 8, seq_len: int = 64,
         lr: float = 3e-4, d_model: int = 128, n_layers: int = 2, n_heads: int = 8,
         measure_every: int = 25, eval_batches: int = 30) -> Dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert os.path.exists(os.path.join(DATA_DIR, "train.bin")), \
        f"missing data at {DATA_DIR}"
    t0 = time.time()

    results: Dict[str, Dict] = {}
    for name, hlr in SETTINGS:
        per_seed = []
        for s in seeds:
            r = run_one(hlr, s, steps=steps, batch=batch, seq_len=seq_len, lr=lr,
                        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                        measure_every=measure_every, eval_batches=eval_batches,
                        device=device)
            per_seed.append(r)
            print(f"[{name:>4s} seed {s}] ppl={r['final_ppl']:.3f} "
                  f"nan={r['nan_count']} gnorm={r['final_grad_norm']:.4f} "
                  f"|L_hebb|={r['hebb_mag_mean']:.2e} "
                  f"g_hebb/g_full={r['hebb_frac_mean']:.2e} "
                  f"({r['seconds']:.0f}s)", flush=True)
        ppls = [r["final_ppl"] for r in per_seed]
        results[name] = {
            "hebbian_lr": hlr,
            "ppl_mean": statistics.fmean(ppls),
            "ppl_std": statistics.pstdev(ppls) if len(ppls) > 1 else 0.0,
            "ppl_vals": ppls,
            "nan_total": sum(r["nan_count"] for r in per_seed),
            "hebb_mag_mean": statistics.fmean(r["hebb_mag_mean"] for r in per_seed),
            "hebb_frac_mean": statistics.fmean(r["hebb_frac_mean"] for r in per_seed),
            "g_full_mean": statistics.fmean(r["g_full_mean"] for r in per_seed),
            "g_hebb_mean": statistics.fmean(r["g_hebb_mean"] for r in per_seed),
            "per_seed": per_seed,
        }

    report = {
        "config": dict(seeds=list(seeds), steps=steps, batch=batch, seq_len=seq_len,
                       lr=lr, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                       measure_every=measure_every, eval_batches=eval_batches,
                       device=device, data=DATA_DIR),
        "settings": results,
        "seconds": round(time.time() - t0, 1),
    }
    _save(report)
    _print(report)
    return report


def _save(report: Dict) -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "report_ablation_hebbian_lr.json"), "w") as f:
        json.dump(report, f, indent=2)

    cfg = report["config"]
    s = report["settings"]
    off_ppl = s["off"]["ppl_mean"]
    lines = ["# Hebbian-lr ablation: is the co-activation term a real knob?", ""]
    lines.append(f"Seeds `{cfg['seeds']}` | {cfg['steps']} steps | "
                 f"model {cfg['d_model']}d x {cfg['n_layers']}L x {cfg['n_heads']}H | "
                 f"seq {cfg['seq_len']} batch {cfg['batch']} | real WikiText-103 "
                 f"(gpt2) | {cfg['device']} | runtime {report['seconds']}s")
    lines.append("")
    lines.append("Isolated Hebbian (every other v2.0 module OFF). `g_hebb/g_full` is "
                 "the fraction of the per-batch gradient norm contributed by the "
                 "Hebbian term alone -- the key diagnostic (value can be ~1e-8 while "
                 "gradient still matters, or not).")
    lines.append("")
    lines.append("| hebbian_lr | val PPL (mean+/-std) | dPPL vs off | NaN | "
                 "|L_hebb| | g_hebb/g_full |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, _ in SETTINGS:
        v = s[name]
        dppl = v["ppl_mean"] - off_ppl
        dstr = "--" if name == "off" else f"{dppl:+.3f}"
        lines.append(
            f"| {name} | {v['ppl_mean']:.3f} +/- {v['ppl_std']:.3f} | {dstr} | "
            f"{v['nan_total']} | {v['hebb_mag_mean']:.2e} | "
            f"{v['hebb_frac_mean']:.2e} |")
    lines.append("")
    # Honest verdict, computed not asserted.
    max_frac = max(s[n]["hebb_frac_mean"] for n, _ in SETTINGS if n != "off")
    max_dppl = max(abs(s[n]["ppl_mean"] - off_ppl) for n, _ in SETTINGS if n != "off")
    noise = max(s[n]["ppl_std"] for n, _ in SETTINGS)
    lines.append(f"**Read:** largest Hebbian gradient fraction across the sweep = "
                 f"{max_frac:.2e}; largest |dPPL vs off| = {max_dppl:.3f} "
                 f"(seed noise ~{noise:.3f}). "
                 + ("PPL moves beyond noise AND gradient share is non-trivial -> "
                    "Hebbian is a real knob worth tuning."
                    if (max_dppl > 2 * noise and max_frac > 1e-3) else
                    "PPL change is within seed noise and the gradient share is "
                    "negligible even at hebbian_lr=1e-1 -> the term is effectively "
                    "inert at these scales; safe to leave OFF (or raise its strength "
                    "by orders of magnitude before it can matter)."))
    lines.append("")
    with open(os.path.join(here, "report_ablation_hebbian_lr.md"), "w") as f:
        f.write("\n".join(lines))


def _print(report: Dict) -> None:
    s = report["settings"]
    off_ppl = s["off"]["ppl_mean"]
    print("\n=== HEBBIAN-LR ABLATION ===")
    for name, _ in SETTINGS:
        v = s[name]
        dppl = v["ppl_mean"] - off_ppl
        dstr = "  (ref)" if name == "off" else f"  dPPL={dppl:+.3f}"
        print(f"{name:>4s}  PPL={v['ppl_mean']:.3f}+/-{v['ppl_std']:.3f}{dstr}  "
              f"NaN={v['nan_total']}  |L_hebb|={v['hebb_mag_mean']:.2e}  "
              f"g_hebb/g_full={v['hebb_frac_mean']:.2e}")


if __name__ == "__main__":
    main()
