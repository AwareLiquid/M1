"""
experiments/liquid_pc/genreplay.py — GENERATIVE replay (pseudo-rehearsal) probe.

WHY THIS EXISTS
---------------
Retest #7 (report_continual.md) found that experience replay nearly SOLVES
forgetting, but it is a GENERIC lever (PC-replay ~= GRU-replay) and it stores RAW
old data. The genuinely interesting follow-up is GENERATIVE replay / pseudo-
rehearsal (Robins 1995; Shin et al. 2017): a snapshot of the just-trained model
DREAMS up old-task-like sequences by free-running its own autoregressive
prediction, and the new task rehearses on those SYNTHETIC sequences — no raw old
data stored.

The honest, PC-relevant question: PCLiquidCore has an EXPLICIT hierarchical
generative pathway (the top-down generate[] weights of predictive coding); does
that make its dreams higher-fidelity than a GRU's implicit rollout, and therefore
reduce forgetting MORE? A reason it might NOT: PCLiquidCore's free-running ROLLOUT
was its weak axis (it lost rollout MSE to the GRU robustly), so its dreams could
be LESS stable. This experiment finds out with error bars instead of guessing.

Variants (forgetting probe: train A -> [dream A] -> train B with replay -> remeasure A):
  * PC-astro       : no replay (the no-consolidation reference).
  * PC-replay      : REAL-data replay (the empirical upper bound for PC).
  * PC-genreplay   : GENERATIVE replay — PC rehearses on its OWN dreamed A-data.
  * GRU-replay     : REAL-data replay (upper bound for GRU).
  * GRU-genreplay  : GENERATIVE replay — GRU rehearses on its OWN dreamed A-data.

So we can read: how close each generative variant gets to its real-data upper
bound, and whether PC's explicit generative structure dreams BETTER than GRU's
(PC-genreplay vs GRU-genreplay) — the only PC-specific claim under test. Whatever
the result, it is saved verbatim.

Run:  python -m experiments.liquid_pc.genreplay
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
from experiments.liquid_pc.model import PCLiquidCore, GRUBaseline, count_params
from experiments.liquid_pc.run import (
    train,
    one_step_mse,
    generate_synthetic,
)


def make_model(label: str, d_in: int, seed: int):
    torch.manual_seed(seed)
    if label.startswith("PC"):
        return PCLiquidCore(d_in, d=48, n_levels=3, dynamic_precision=True,
                            use_astrocyte=True)
    if label.startswith("GRU"):
        return GRUBaseline(d_in, hidden=70)
    raise ValueError(f"unknown model label: {label}")


# variant -> replay mode in {"none", "real", "gen"}.
MODE = {
    "PC-astro": "none",
    "PC-replay": "real",
    "PC-genreplay": "gen",
    "GRU-replay": "real",
    "GRU-genreplay": "gen",
}
LABELS = ["PC-astro", "PC-replay", "PC-genreplay", "GRU-replay", "GRU-genreplay"]


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
    replay_buffer: int = 32,
    replay_batch: int = 16,
    warmup: int = 16,
) -> Dict:
    t0 = time.time()
    acc = {lab: {"a_before": [], "a_after": [], "forgetting": [], "b_after": [],
                 "dream_mse": []} for lab in LABELS}
    params: Dict[str, int] = {}

    for seed in seeds:
        a, b = two_regimes(n_seq, seq_len, d_in, seed=seed)
        a_tr, a_te = train_test_split(a, 0.8)
        b_tr, b_te = train_test_split(b, 0.8)

        for lab in LABELS:
            mode = MODE[lab]
            m = make_model(lab, d_in, seed)
            params.setdefault(lab, count_params(m))

            # task A.
            train(m, a_tr, epochs=epochs, batch=batch, lr=lr, seed=seed)
            a_before = one_step_mse(m, a_te)

            # build the replay buffer.
            dream_mse = float("nan")
            if mode == "real":
                rbuf = a_tr[:replay_buffer]
            elif mode == "gen":
                rbuf = generate_synthetic(m, n_syn=replay_buffer, seq_len=seq_len,
                                          d_in=d_in, seed=seed + 7, warmup=warmup)
                # a transparency proxy for dream fidelity: how well a held-out
                # A-error model (the just-trained m) one-step-predicts the dream
                # vs real A. We report the dream's *self*-consistency by scoring m
                # on the dream (low = the dream lies on m's learned manifold).
                dream_mse = one_step_mse(m, rbuf)
            else:
                rbuf = None

            # task B with the chosen replay buffer.
            train(m, b_tr, epochs=epochs, batch=batch, lr=lr, seed=seed + 1,
                  replay_x=rbuf, replay_batch=replay_batch)
            a_after = one_step_mse(m, a_te)
            b_after = one_step_mse(m, b_te)

            acc[lab]["a_before"].append(a_before)
            acc[lab]["a_after"].append(a_after)
            acc[lab]["forgetting"].append(a_after - a_before)
            acc[lab]["b_after"].append(b_after)
            acc[lab]["dream_mse"].append(dream_mse)

            print(f"[seed {seed}] {lab:14s} "
                  f"A_before={a_before:.5f} A_after={a_after:.5f} "
                  f"forget={a_after - a_before:+.5f} B_after={b_after:.5f}"
                  + (f" dream={dream_mse:.5f}" if mode == "gen" else ""))

    summary = {
        lab: {
            "params": params[lab],
            "a_before": _mean_std(acc[lab]["a_before"]),
            "a_after": _mean_std(acc[lab]["a_after"]),
            "forgetting": _mean_std(acc[lab]["forgetting"]),
            "b_after": _mean_std(acc[lab]["b_after"]),
            "dream_mse": _mean_std([v for v in acc[lab]["dream_mse"]
                                    if v == v]) if MODE[lab] == "gen" else None,
        }
        for lab in LABELS
    }

    def _paired(lab_a: str, lab_b: str) -> Dict:
        diffs = [x - y for x, y in zip(acc[lab_a]["forgetting"],
                                       acc[lab_b]["forgetting"])]
        wins = sum(1 for d in diffs if d < 0)
        return {"mean_diff": statistics.fmean(diffs),
                "a_lower_in": f"{wins}/{len(diffs)}",
                "diffs": [round(d, 5) for d in diffs]}

    paired = {
        "pc_gen_vs_astro": _paired("PC-genreplay", "PC-astro"),
        "pc_gen_vs_pc_real": _paired("PC-genreplay", "PC-replay"),
        "gru_gen_vs_astro": _paired("GRU-genreplay", "PC-astro"),
        "pc_gen_vs_gru_gen": _paired("PC-genreplay", "GRU-genreplay"),
    }

    report = {
        "config": dict(seeds=list(seeds), n_seq=n_seq, seq_len=seq_len, d_in=d_in,
                       epochs=epochs, batch=batch, lr=lr,
                       replay_buffer=replay_buffer, replay_batch=replay_batch,
                       warmup=warmup),
        "summary": summary,
        "paired": paired,
        "seconds": round(time.time() - t0, 1),
    }
    _save(report)
    _print(report)
    return report


def _save(report: Dict) -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "report_genreplay.json"), "w") as f:
        json.dump(report, f, indent=2)

    s = report["summary"]
    cfg = report["config"]
    p = report["paired"]
    lines = ["# PC-Liquid-Core generative-replay (pseudo-rehearsal) report", ""]
    lines.append(f"Seeds: `{cfg['seeds']}` (n={len(cfg['seeds'])})  |  "
                 f"replay buffer={cfg['replay_buffer']} (batch {cfg['replay_batch']}, "
                 f"warmup {cfg['warmup']})  |  runtime {report['seconds']}s")
    lines.append("")
    lines.append("Forgetting probe: train A -> build replay buffer (REAL subset, "
                 "or GENERATIVE dream from the post-A model) -> train B rehearsing "
                 "that buffer -> re-measure A. `dream` = the post-A model's own "
                 "1-step MSE on its dream (low = self-consistent, NOT a fidelity-"
                 "to-real-A measure).")
    lines.append("")
    lines.append("| variant | #params | forgetting | B_after | dream self-MSE |")
    lines.append("|---|---:|---:|---:|---:|")
    for lab in LABELS:
        v = s[lab]
        dream = (f"{v['dream_mse']['mean']:.5f} +/- {v['dream_mse']['std']:.5f}"
                 if v["dream_mse"] else "-")
        lines.append(
            f"| {lab} | {v['params']} | "
            f"{v['forgetting']['mean']:+.5f} +/- {v['forgetting']['std']:.5f} | "
            f"{v['b_after']['mean']:.5f} +/- {v['b_after']['std']:.5f} | {dream} |"
        )
    lines.append("")
    lines.append("Paired (per-seed) forgetting deltas (negative = first variant "
                 "forgets LESS):")
    lines.append("")
    lines.append("| comparison | mean diff | first-lower-in |")
    lines.append("|---|---:|---:|")
    rows = [
        ("PC-genreplay vs astro", "pc_gen_vs_astro"),
        ("PC-genreplay vs PC-replay(real)", "pc_gen_vs_pc_real"),
        ("GRU-genreplay vs astro", "gru_gen_vs_astro"),
        ("PC-genreplay vs GRU-genreplay", "pc_gen_vs_gru_gen"),
    ]
    for label, key in rows:
        lines.append(f"| {label} | {p[key]['mean_diff']:+.5f} | "
                     f"{p[key]['a_lower_in']} |")
    lines.append("")
    with open(os.path.join(here, "report_genreplay.md"), "w") as f:
        f.write("\n".join(lines))


def _print(report: Dict) -> None:
    s = report["summary"]
    p = report["paired"]
    print("\n=== GENERATIVE-REPLAY SUMMARY (mean +/- std) ===")
    for lab in LABELS:
        v = s[lab]
        print(f"{lab:14s} forget={v['forgetting']['mean']:+.5f}"
              f"+/-{v['forgetting']['std']:.5f} B_after={v['b_after']['mean']:.5f}")
    for label, key in [
        ("PC-gen vs astro       ", "pc_gen_vs_astro"),
        ("PC-gen vs PC-real     ", "pc_gen_vs_pc_real"),
        ("GRU-gen vs astro      ", "gru_gen_vs_astro"),
        ("PC-gen vs GRU-gen     ", "pc_gen_vs_gru_gen"),
    ]:
        print(f"{label}: mean diff {p[key]['mean_diff']:+.5f} "
              f"(first lower in {p[key]['a_lower_in']})")


if __name__ == "__main__":
    main()
