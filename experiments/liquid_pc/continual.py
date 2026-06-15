"""
experiments/liquid_pc/continual.py — continual-learning probe: which consolidation
lever actually reduces catastrophic forgetting, and is any of it PCLiquidCore-
specific?

WHY THIS EXISTS
---------------
The multi-seed pass (report_multiseed.md) established ONE robust, error-barred
PCLiquidCore advantage: less catastrophic forgetting than the RNN baselines. We
then tried to AMPLIFY it with two explicit consolidation levers and measure each
honestly with error bars:
  * EWC — Elastic Weight Consolidation (Kirkpatrick 2017): protect weights in
    PARAMETER space via a diagonal empirical-Fisher importance + quadratic anchor.
    (A naive (mean-grad)^2 Fisher collapses to ~0 at the task-A minimum and makes
    EWC a no-op — the per-sample EMPIRICAL Fisher is required; see model.py.)
  * Replay — experience replay / rehearsal (the systems-consolidation / hippocampal
    -replay analogue): interleave a small buffer of the OLD task during the new
    task. Revisits old DATA directly rather than protecting weights.

Protocol per seed (forgetting probe, focused — no rollout/main-task here):
  two_regimes -> train A -> measure A-error (a_before) -> [consolidate A for EWC]
  -> train B WITH the lever active -> re-measure A-error (a_after).
  forgetting = a_after - a_before. We also record B-error after B, to confirm a
  lever did not simply BLOCK learning B (low forgetting is worthless if the model
  never learned the new task).

Variants (PC-* all share the SAME astrocyte architecture; the only difference is
the consolidation protocol — a clean ablation):
  * PC-astro      : no explicit consolidation (the multiseed reference).
  * PC-ewc        : EWC (uniform empirical Fisher).
  * PC-replay     : experience replay only.
  * PC-ewc-replay : EWC + replay (do they STACK?).
  * GRU-replay    : RNN + replay (is replay PCLiquidCore-specific, or generic?).

Honesty notes: both EWC and replay are well-known GENERIC methods applied
identically to every architecture, so this is a fair head-to-head; replay stores
raw old data (O(buffer) memory) whereas EWC is O(params). ``lam`` and the replay
buffer/batch are fixed across variants. Whatever the result, it is saved verbatim.

Run:  python -m experiments.liquid_pc.continual
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

from experiments.liquid_pc.data import train_test_split, two_regimes
from experiments.liquid_pc.model import (
    PCLiquidCore,
    GRUBaseline,
    count_params,
)
from experiments.liquid_pc.run import train, one_step_mse


# variant -> (factory needs, ewc config). All PC variants share architecture.
def make_model(label: str, d_in: int, seed: int):
    torch.manual_seed(seed)
    if label.startswith("PC"):
        return PCLiquidCore(d_in, d=48, n_levels=3, dynamic_precision=True,
                            use_astrocyte=True)
    if label.startswith("GRU"):
        return GRUBaseline(d_in, hidden=70)
    raise ValueError(f"unknown model label: {label}")


# per-variant consolidation config: (uses_ewc, calcium_weighted, uses_replay).
# EWC protects weights in parameter space; replay rehearses a small buffer of the
# OLD task. Both are training protocols applied identically across architectures.
CFG = {
    "PC-astro": (False, False, False),       # no explicit consolidation (ref)
    "PC-ewc": (True, False, False),          # EWC (uniform empirical Fisher)
    "PC-replay": (False, False, True),       # experience replay only
    "PC-ewc-replay": (True, False, True),    # EWC + replay (do they stack?)
    "GRU-replay": (False, False, True),      # replay reference (no PC machinery)
}
LABELS = ["PC-astro", "PC-ewc", "PC-replay", "PC-ewc-replay", "GRU-replay"]


def _mean_std(xs: List[float]) -> Dict:
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
    ewc_lambda: float = 1e5,
    replay_buffer: int = 32,
    replay_batch: int = 16,
) -> Dict:
    t0 = time.time()
    acc = {lab: {"a_before": [], "a_after": [], "forgetting": [], "b_after": []}
           for lab in LABELS}
    params: Dict[str, int] = {}

    for seed in seeds:
        a, b = two_regimes(n_seq, seq_len, d_in, seed=seed)
        a_tr, a_te = train_test_split(a, 0.8)
        b_tr, b_te = train_test_split(b, 0.8)

        for lab in LABELS:
            uses_ewc, cal, uses_replay = CFG[lab]
            m = make_model(lab, d_in, seed)
            params.setdefault(lab, count_params(m))

            # task A.
            train(m, a_tr, epochs=epochs, batch=batch, lr=lr, seed=seed)
            a_before = one_step_mse(m, a_te)

            # consolidate A (EWC variants only).
            if uses_ewc:
                m.consolidate(a_tr, calcium_weighted=cal)

            # task B: EWC penalty (param-space) and/or replay (a small A buffer).
            lam = ewc_lambda if uses_ewc else 0.0
            rbuf = a_tr[:replay_buffer] if uses_replay else None
            train(m, b_tr, epochs=epochs, batch=batch, lr=lr, seed=seed + 1,
                  ewc_lambda=lam, replay_x=rbuf, replay_batch=replay_batch)
            a_after = one_step_mse(m, a_te)
            b_after = one_step_mse(m, b_te)

            acc[lab]["a_before"].append(a_before)
            acc[lab]["a_after"].append(a_after)
            acc[lab]["forgetting"].append(a_after - a_before)
            acc[lab]["b_after"].append(b_after)

            print(f"[seed {seed}] {lab:13s} "
                  f"A_before={a_before:.5f} A_after={a_after:.5f} "
                  f"forget={a_after - a_before:+.5f} B_after={b_after:.5f}")

    summary = {
        lab: {
            "params": params[lab],
            "a_before": _mean_std(acc[lab]["a_before"]),
            "a_after": _mean_std(acc[lab]["a_after"]),
            "forgetting": _mean_std(acc[lab]["forgetting"]),
            "b_after": _mean_std(acc[lab]["b_after"]),
        }
        for lab in LABELS
    }
    # paired (per-seed) deltas: which consolidation lever cuts forgetting, and do
    # EWC + replay STACK? Same seeds across variants -> a paired comparison.
    def _paired(lab_a: str, lab_b: str) -> Dict:
        da = acc[lab_a]["forgetting"]
        db = acc[lab_b]["forgetting"]
        diffs = [x - y for x, y in zip(da, db)]
        wins = sum(1 for d in diffs if d < 0)        # lab_a forgets less
        return {"mean_diff": statistics.fmean(diffs),
                "a_lower_in": f"{wins}/{len(diffs)}",
                "diffs": [round(d, 5) for d in diffs]}

    paired = {
        "ewc_vs_astro": _paired("PC-ewc", "PC-astro"),
        "replay_vs_astro": _paired("PC-replay", "PC-astro"),
        "replay_vs_ewc": _paired("PC-replay", "PC-ewc"),
        "ewcreplay_vs_replay": _paired("PC-ewc-replay", "PC-replay"),
        "pcreplay_vs_grureplay": _paired("PC-replay", "GRU-replay"),
    }

    report = {
        "config": dict(seeds=list(seeds), n_seq=n_seq, seq_len=seq_len, d_in=d_in,
                       epochs=epochs, batch=batch, lr=lr, ewc_lambda=ewc_lambda,
                       replay_buffer=replay_buffer, replay_batch=replay_batch),
        "summary": summary,
        "paired": paired,
        "seconds": round(time.time() - t0, 1),
    }
    _save(report)
    _print(report)
    return report


def _save(report: Dict) -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "report_continual.json"), "w") as f:
        json.dump(report, f, indent=2)

    s = report["summary"]
    cfg = report["config"]
    p = report["paired"]
    lines = ["# PC-Liquid-Core continual-learning: EWC vs replay report", ""]
    lines.append(f"Seeds: `{cfg['seeds']}` (n={len(cfg['seeds'])})  |  "
                 f"EWC lambda={cfg['ewc_lambda']}  |  "
                 f"replay buffer={cfg['replay_buffer']} (batch {cfg['replay_batch']})  |  "
                 f"runtime {report['seconds']}s")
    lines.append("")
    lines.append("Forgetting probe: train A -> consolidate A -> train B (EWC "
                 "and/or replay on) -> re-measure A. Lower `forgetting` = less "
                 "catastrophic forgetting; `B_after` must stay low or the lever "
                 "merely blocked learning B.")
    lines.append("")
    lines.append("| variant | #params | A_before | A_after | forgetting | B_after |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for lab in LABELS:
        v = s[lab]
        lines.append(
            f"| {lab} | {v['params']} | "
            f"{v['a_before']['mean']:.5f} +/- {v['a_before']['std']:.5f} | "
            f"{v['a_after']['mean']:.5f} +/- {v['a_after']['std']:.5f} | "
            f"{v['forgetting']['mean']:+.5f} +/- {v['forgetting']['std']:.5f} | "
            f"{v['b_after']['mean']:.5f} +/- {v['b_after']['std']:.5f} |"
        )
    lines.append("")
    lines.append("Paired (per-seed) forgetting deltas (negative = first variant "
                 "forgets LESS):")
    lines.append("")
    lines.append("| comparison | mean diff | first-lower-in |")
    lines.append("|---|---:|---:|")
    rows = [
        ("EWC vs astro-only", "ewc_vs_astro"),
        ("replay vs astro-only", "replay_vs_astro"),
        ("replay vs EWC", "replay_vs_ewc"),
        ("EWC+replay vs replay", "ewcreplay_vs_replay"),
        ("PC-replay vs GRU-replay", "pcreplay_vs_grureplay"),
    ]
    for label, key in rows:
        lines.append(f"| {label} | {p[key]['mean_diff']:+.5f} | "
                     f"{p[key]['a_lower_in']} |")
    lines.append("")
    with open(os.path.join(here, "report_continual.md"), "w") as f:
        f.write("\n".join(lines))


def _print(report: Dict) -> None:
    s = report["summary"]
    p = report["paired"]
    print("\n=== CONTINUAL-LEARNING SUMMARY (mean +/- std) ===")
    for lab in LABELS:
        v = s[lab]
        print(f"{lab:13s} forget={v['forgetting']['mean']:+.5f}"
              f"+/-{v['forgetting']['std']:.5f} "
              f"B_after={v['b_after']['mean']:.5f}")
    for label, key in [
        ("EWC vs astro      ", "ewc_vs_astro"),
        ("replay vs astro   ", "replay_vs_astro"),
        ("replay vs EWC     ", "replay_vs_ewc"),
        ("EWC+replay vs repl", "ewcreplay_vs_replay"),
        ("PC-repl vs GRU-rep", "pcreplay_vs_grureplay"),
    ]:
        print(f"{label}: mean diff {p[key]['mean_diff']:+.5f} "
              f"(first lower in {p[key]['a_lower_in']})")


if __name__ == "__main__":
    main()
