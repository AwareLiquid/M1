"""
experiments/liquid_pc/continual.py — calcium-weighted EWC continual-learning probe.

WHY THIS EXISTS
---------------
The multi-seed pass (report_multiseed.md) established ONE robust, error-barred
PCLiquidCore advantage: less catastrophic forgetting than the RNN baselines. The
astrocyte consolidation gate then shaved that forgetting a little further (~3%,
5/5 seeds). This script asks the next honest question: can an EXPLICIT
consolidation protocol — Elastic Weight Consolidation (Kirkpatrick 2017) — reduce
forgetting MORE, and does the PCLiquidCore-specific CALCIUM WEIGHTING of the
Fisher importance beat plain (uniform) EWC?

Protocol per seed (the forgetting probe, focused — no rollout/main-task here):
  two_regimes -> train on A -> measure A-error (a_before) -> CONSOLIDATE on A
  (estimate Fisher importance + anchor weights) -> train on B WITH the EWC
  penalty active -> re-measure A-error (a_after). forgetting = a_after - a_before.
  We also record B-error after training B, to confirm EWC did not simply block
  learning B (low forgetting is worthless if the model never learned the new task).

Variants (all share the SAME astrocyte architecture, so the only difference is
the consolidation protocol — a clean ablation):
  * PC-astro      : astrocyte gate, NO EWC (the multiseed reference).
  * PC-ewc-plain  : astrocyte gate + EWC with UNIFORM Fisher.
  * PC-ewc-cal    : astrocyte gate + EWC with CALCIUM-weighted Fisher.
  * GRU           : RNN reference (no consolidation mechanism), for scale.

Honesty notes: EWC is a well-known generic method; the only PCLiquidCore-specific
claim under test is whether glial-calcium importance weighting helps OVER plain
EWC. Whatever the result, it is printed and saved verbatim. ``lam`` (EWC strength)
is a hyperparameter — too large blocks learning B, too small gives no protection;
it is fixed across variants so the calcium-vs-uniform comparison is fair.

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
    if label in ("PC-astro", "PC-ewc-plain", "PC-ewc-cal"):
        return PCLiquidCore(d_in, d=48, n_levels=3, dynamic_precision=True,
                            use_astrocyte=True)
    if label == "GRU":
        return GRUBaseline(d_in, hidden=70)
    raise ValueError(f"unknown model label: {label}")


# per-variant consolidation config: (uses_ewc, calcium_weighted).
EWC_CFG = {
    "PC-astro": (False, False),
    "PC-ewc-plain": (True, False),
    "PC-ewc-cal": (True, True),
    "GRU": (False, False),
}
LABELS = ["PC-astro", "PC-ewc-plain", "PC-ewc-cal", "GRU"]


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
            uses_ewc, cal = EWC_CFG[lab]
            m = make_model(lab, d_in, seed)
            params.setdefault(lab, count_params(m))

            # task A.
            train(m, a_tr, epochs=epochs, batch=batch, lr=lr, seed=seed)
            a_before = one_step_mse(m, a_te)

            # consolidate A (EWC variants only).
            if uses_ewc:
                m.consolidate(a_tr, calcium_weighted=cal)

            # task B (with EWC penalty active for EWC variants).
            lam = ewc_lambda if uses_ewc else 0.0
            train(m, b_tr, epochs=epochs, batch=batch, lr=lr, seed=seed + 1,
                  ewc_lambda=lam)
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
    # paired (per-seed) deltas: does EWC reduce forgetting vs astrocyte-only,
    # and does calcium weighting beat plain EWC? Same seeds -> a paired test.
    def _paired(lab_a: str, lab_b: str) -> Dict:
        da = acc[lab_a]["forgetting"]
        db = acc[lab_b]["forgetting"]
        diffs = [x - y for x, y in zip(da, db)]
        wins = sum(1 for d in diffs if d < 0)        # lab_a forgets less
        return {"mean_diff": statistics.fmean(diffs),
                "a_lower_in": f"{wins}/{len(diffs)}",
                "diffs": [round(d, 5) for d in diffs]}

    paired = {
        "ewc_plain_vs_astro": _paired("PC-ewc-plain", "PC-astro"),
        "ewc_cal_vs_astro": _paired("PC-ewc-cal", "PC-astro"),
        "ewc_cal_vs_plain": _paired("PC-ewc-cal", "PC-ewc-plain"),
    }

    report = {
        "config": dict(seeds=list(seeds), n_seq=n_seq, seq_len=seq_len, d_in=d_in,
                       epochs=epochs, batch=batch, lr=lr, ewc_lambda=ewc_lambda),
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
    lines = ["# PC-Liquid-Core continual-learning (calcium-weighted EWC) report", ""]
    lines.append(f"Seeds: `{cfg['seeds']}` (n={len(cfg['seeds'])})  |  "
                 f"EWC lambda={cfg['ewc_lambda']}  |  runtime {report['seconds']}s")
    lines.append("")
    lines.append("Forgetting probe: train A -> consolidate A -> train B (EWC on) "
                 "-> re-measure A. Lower `forgetting` = less catastrophic "
                 "forgetting; `B_after` must stay low or EWC merely blocked "
                 "learning B.")
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
    lines.append(f"| EWC-plain vs astro-only | {p['ewc_plain_vs_astro']['mean_diff']:+.5f} | "
                 f"{p['ewc_plain_vs_astro']['a_lower_in']} |")
    lines.append(f"| EWC-cal vs astro-only | {p['ewc_cal_vs_astro']['mean_diff']:+.5f} | "
                 f"{p['ewc_cal_vs_astro']['a_lower_in']} |")
    lines.append(f"| EWC-cal vs EWC-plain | {p['ewc_cal_vs_plain']['mean_diff']:+.5f} | "
                 f"{p['ewc_cal_vs_plain']['a_lower_in']} |")
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
    print(f"\nEWC-plain vs astro-only: mean diff "
          f"{p['ewc_plain_vs_astro']['mean_diff']:+.5f} "
          f"(plain forgets less in {p['ewc_plain_vs_astro']['a_lower_in']})")
    print(f"EWC-cal   vs astro-only: mean diff "
          f"{p['ewc_cal_vs_astro']['mean_diff']:+.5f} "
          f"(cal forgets less in {p['ewc_cal_vs_astro']['a_lower_in']})")
    print(f"EWC-cal   vs EWC-plain : mean diff "
          f"{p['ewc_cal_vs_plain']['mean_diff']:+.5f} "
          f"(cal forgets less in {p['ewc_cal_vs_plain']['a_lower_in']})")


if __name__ == "__main__":
    main()
