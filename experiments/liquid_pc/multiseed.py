"""
experiments/liquid_pc/multiseed.py — multi-seed robustness pass.

WHY THIS EXISTS (an honesty correction, 2026-06-15)
---------------------------------------------------
The single-seed reports (report.md / report_prec.md / ...) turned out to be
FRAGILE. While adding the dynamic-precision gate to ``PCLiquidCore`` I noticed
the *baselines* swung wildly between runs (e.g. GRU rollout 0.051 -> 0.832, LSTM
0.032 -> 0.015) even though only PCLiquidCore changed. The cause is mundane but
fatal to a single-seed comparison: constructing PCLiquidCore consumes RNG draws,
and ``build_models`` builds it BEFORE the baselines under one shared seed, so any
change to PCLiquidCore's parameter count shifts every downstream model's random
init. A single seed cannot tell signal from this init noise.

This script fixes BOTH problems so a conclusion can be trusted:
  1. **Independent per-model seeding** — every model is constructed under its own
     ``manual_seed`` (not a shared draw order), so adding params to one model
     cannot perturb another. Static-vs-dynamic PCLiquidCore still differ in their
     own readout init across the gate, but that is exactly the random-init
     variation we now AVERAGE OVER.
  2. **Multiple seeds, report mean +/- std** — the only honest way to claim one
     model is better than another on noisy single runs.

It re-asks the live question with error bars: across seeds, does precision-
weighting (static) robustly improve PCLiquidCore's rollout vs the baselines, and
does INPUT-DEPENDENT (dynamic) precision beat static precision — without making
catastrophic forgetting worse?

Run:  python -m experiments.liquid_pc.multiseed
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from typing import Dict, List

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.liquid_pc.data import make_signal, train_test_split, two_regimes
from experiments.liquid_pc.model import (
    PCLiquidCore,
    GRUBaseline,
    LSTMBaseline,
    TinyTransformer,
    count_params,
)
from experiments.liquid_pc.run import (
    train,
    one_step_mse,
    rollout_mse,
    persistence_mse,
)


def make_model(label: str, d_in: int, seed: int):
    """Construct one model UNDER ITS OWN SEED so its init is independent of any
    other model's parameter count (the fix for the cross-model RNG confound)."""
    torch.manual_seed(seed)
    if label == "PC-static":
        return PCLiquidCore(d_in, d=48, n_levels=3)
    if label == "PC-dynamic":
        return PCLiquidCore(d_in, d=48, n_levels=3, dynamic_precision=True)
    if label == "PC-astro":
        # dynamic precision (validated >= static) + astrocyte consolidation gate,
        # targeting the robust low-forgetting advantage.
        return PCLiquidCore(d_in, d=48, n_levels=3, dynamic_precision=True,
                            use_astrocyte=True)
    if label == "GRU":
        return GRUBaseline(d_in, hidden=70)
    if label == "LSTM":
        return LSTMBaseline(d_in, hidden=60)
    if label == "Transformer":
        return TinyTransformer(d_in, d_model=32, n_heads=4, n_layers=2, max_len=160)
    raise ValueError(f"unknown model label: {label}")


LABELS = ["PC-static", "PC-dynamic", "PC-astro", "GRU", "LSTM", "Transformer"]


def _mean_std(xs: List[float]) -> Dict[str, float]:
    return {
        "mean": statistics.fmean(xs),
        "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
        "n": len(xs),
        "vals": [round(v, 5) for v in xs],
    }


def main(
    *,
    seeds=(0, 1, 2, 3, 4),
    n_seq: int = 320,
    seq_len: int = 96,
    d_in: int = 1,
    epochs: int = 40,
    batch: int = 64,
    lr: float = 3e-3,
    prime: int = 48,
) -> Dict:
    t0 = time.time()
    acc = {lab: {"one_step": [], "rollout": [], "forgetting": []} for lab in LABELS}
    persist_vals: List[float] = []
    params: Dict[str, int] = {}

    for seed in seeds:
        # data is seed-matched across all models within a seed.
        x = make_signal(n_seq, seq_len, d_in, seed=seed)
        x_tr, x_te = train_test_split(x, 0.8)
        persist_vals.append(persistence_mse(x_te))
        a, b = two_regimes(n_seq, seq_len, d_in, seed=seed)
        a_tr, a_te = train_test_split(a, 0.8)
        b_tr, _ = train_test_split(b, 0.8)

        for lab in LABELS:
            # main task (independent init per model).
            m = make_model(lab, d_in, seed)
            params.setdefault(lab, count_params(m))
            train(m, x_tr, epochs=epochs, batch=batch, lr=lr, seed=seed)
            acc[lab]["one_step"].append(one_step_mse(m, x_te))
            acc[lab]["rollout"].append(rollout_mse(m, x_te, prime=prime))

            # forgetting probe (fresh model: train A -> train B -> re-measure A).
            mf = make_model(lab, d_in, seed)
            train(mf, a_tr, epochs=epochs, batch=batch, lr=lr, seed=seed)
            a_before = one_step_mse(mf, a_te)
            train(mf, b_tr, epochs=epochs, batch=batch, lr=lr, seed=seed + 1)
            a_after = one_step_mse(mf, a_te)
            acc[lab]["forgetting"].append(a_after - a_before)

            print(f"[seed {seed}] {lab:13s} "
                  f"1step={acc[lab]['one_step'][-1]:.5f} "
                  f"rollout={acc[lab]['rollout'][-1]:.5f} "
                  f"forget={acc[lab]['forgetting'][-1]:+.5f}")

    summary = {
        lab: {
            "params": params[lab],
            "one_step": _mean_std(acc[lab]["one_step"]),
            "rollout": _mean_std(acc[lab]["rollout"]),
            "forgetting": _mean_std(acc[lab]["forgetting"]),
        }
        for lab in LABELS
    }
    report = {
        "config": dict(seeds=list(seeds), n_seq=n_seq, seq_len=seq_len, d_in=d_in,
                       epochs=epochs, batch=batch, lr=lr, prime=prime),
        "persistence_mse": _mean_std(persist_vals),
        "summary": summary,
        "seconds": round(time.time() - t0, 1),
    }
    _save(report)
    _print(report)
    return report


def _save(report: Dict) -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "report_multiseed.json"), "w") as f:
        json.dump(report, f, indent=2)

    s = report["summary"]
    seeds = report["config"]["seeds"]
    lines = ["# PC-Liquid-Core multi-seed robustness report", ""]
    lines.append(f"Seeds: `{seeds}` (n={len(seeds)})  |  "
                 f"runtime {report['seconds']}s")
    lines.append("")
    lines.append(f"Persistence 1-step MSE: "
                 f"**{report['persistence_mse']['mean']:.5f} "
                 f"+/- {report['persistence_mse']['std']:.5f}**")
    lines.append("")
    lines.append("Mean +/- std over seeds (lower is better everywhere):")
    lines.append("")
    lines.append("| model | #params | 1-step MSE | rollout MSE | forgetting |")
    lines.append("|---|---:|---:|---:|---:|")
    for lab in LABELS:
        v = s[lab]
        lines.append(
            f"| {lab} | {v['params']} | "
            f"{v['one_step']['mean']:.5f} +/- {v['one_step']['std']:.5f} | "
            f"{v['rollout']['mean']:.5f} +/- {v['rollout']['std']:.5f} | "
            f"{v['forgetting']['mean']:+.5f} +/- {v['forgetting']['std']:.5f} |"
        )
    lines.append("")
    with open(os.path.join(here, "report_multiseed.md"), "w") as f:
        f.write("\n".join(lines))


def _print(report: Dict) -> None:
    s = report["summary"]
    print("\n=== MULTI-SEED SUMMARY (mean +/- std) ===")
    for lab in LABELS:
        v = s[lab]
        print(f"{lab:13s} rollout={v['rollout']['mean']:.5f}+/-{v['rollout']['std']:.5f} "
              f"forget={v['forgetting']['mean']:+.5f}+/-{v['forgetting']['std']:.5f}")
    pst = s["PC-static"]["rollout"]["mean"]
    pdy = s["PC-dynamic"]["rollout"]["mean"]
    print(f"\nrollout: PC-dynamic {pdy:.5f} vs PC-static {pst:.5f} "
          f"-> dynamic {'better' if pdy < pst else 'NOT better'}")
    fdy = s["PC-dynamic"]["forgetting"]["mean"]
    fas = s["PC-astro"]["forgetting"]["mean"]
    print(f"forgetting: PC-astro {fas:+.5f} vs PC-dynamic {fdy:+.5f} "
          f"-> astrocyte {'reduces forgetting' if fas < fdy else 'does NOT reduce'}")


if __name__ == "__main__":
    main()
