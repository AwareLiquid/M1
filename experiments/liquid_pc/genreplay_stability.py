"""
experiments/liquid_pc/genreplay_stability.py — is PCLiquidCore's generative
replay genuinely MORE STABLE than a GRU's, dream-for-dream, at a scarce budget?

WHY THIS EXISTS
---------------
Retest #9 (report_genreplay_scarcity.md) found that as the dream buffer shrinks
the PC-minus-GRU MEAN forgetting gap widens (-0.0142 -> -0.0266), but the paired
win-count DEGRADES (4/5 -> 2/5). The widening mean turned out to be driven by
GRU's forgetting VARIANCE exploding under scarcity (two seeds blew up to ~0.15),
not by PC robustly winning more often. So the honest, reproducible PC-specific
property is STABILITY (low variance) of generative replay under scarce dreams.

BUT retest #9's variance was measured ACROSS SEEDS, which conflates three noise
sources: model init, task data, and the single dream draw. To test the stability
claim cleanly we must hold the model FIXED and vary ONLY the dream. This script
does exactly that: per (seed, kind) it trains A once, snapshots the post-A
weights, then repeats R times -- each repeat dreams a fresh buffer with a
DIFFERENT dream seed, restores the identical snapshot, and trains B with an
IDENTICAL training RNG. The only thing that changes across the R repeats is the
dream content. The spread of forgetting across those R dreams is therefore PURE
dream-induced instability, with model init / data / train-B order all fixed.

First-class metric: within-(seed) std of forgetting over R dreams. If PC's
explicit hierarchical generative pathway dreams more reliably, PC's within-seed
std should be tighter than GRU's, paired across seeds.

Protocol note: train-B RNG uses a fixed per-(seed,kind) seed (NOT the dream
index), so dream content is the sole varying factor. Buffer is fixed scarce.

Run:  python -m experiments.liquid_pc.genreplay_stability
"""

from __future__ import annotations

import copy
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
from experiments.liquid_pc.model import PCLiquidCore, GRUBaseline
from experiments.liquid_pc.run import train, one_step_mse, generate_synthetic


def make_model(kind: str, d_in: int, seed: int):
    torch.manual_seed(seed)
    if kind == "PC":
        return PCLiquidCore(d_in, d=48, n_levels=3, dynamic_precision=True,
                            use_astrocyte=True)
    return GRUBaseline(d_in, hidden=70)


KINDS = ["PC", "GRU"]


def _stats(xs: List[float]) -> Dict:
    return {
        "mean": statistics.fmean(xs),
        "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
        "min": min(xs),
        "max": max(xs),
        "n": len(xs),
        "vals": [round(v, 5) for v in xs],
    }


def main(
    *,
    seeds=(0, 1, 2, 3, 4),
    n_dreams: int = 6,
    buf: int = 8,
    n_seq: int = 320,
    seq_len: int = 96,
    d_in: int = 1,
    epochs: int = 40,
    batch: int = 64,
    lr: float = 3e-3,
    replay_batch: int = 8,
    warmup: int = 16,
) -> Dict:
    t0 = time.time()
    # per-(seed,kind): forgetting over n_dreams repeats (model fixed, dream varies).
    # within[kind] = list over seeds of {"std":..., "mean":..., "vals":[...]}
    within = {k: [] for k in KINDS}
    # also keep the flat pool of every (seed,dream) forgetting for an overall view.
    pooled = {k: [] for k in KINDS}

    for seed in seeds:
        a, b = two_regimes(n_seq, seq_len, d_in, seed=seed)
        a_tr, a_te = train_test_split(a, 0.8)
        b_tr, b_te = train_test_split(b, 0.8)

        for kind in KINDS:
            m = make_model(kind, d_in, seed)
            train(m, a_tr, epochs=epochs, batch=batch, lr=lr, seed=seed)
            a_before = one_step_mse(m, a_te)
            snapshot = copy.deepcopy(m.state_dict())        # fixed post-A model
            b_seed = seed + 1                               # fixed train-B RNG

            forgets = []
            for r in range(n_dreams):
                m.load_state_dict(snapshot)                 # identical start
                dream = generate_synthetic(m, n_syn=buf, seq_len=seq_len,
                                           d_in=d_in, seed=1000 * seed + 7 * r + 7,
                                           warmup=warmup)
                train(m, b_tr, epochs=epochs, batch=batch, lr=lr, seed=b_seed,
                      replay_x=dream, replay_batch=min(replay_batch, buf))
                f = one_step_mse(m, a_te) - a_before
                forgets.append(f)
                print(f"[seed {seed}] {kind:3s} dream={r} forget={f:+.5f}",
                      flush=True)

            st = _stats(forgets)
            within[kind].append(st)
            pooled[kind].extend(forgets)
            print(f"[seed {seed}] {kind:3s} within-seed: "
                  f"mean={st['mean']:+.5f} std={st['std']:.5f} "
                  f"[{st['min']:+.5f},{st['max']:+.5f}]", flush=True)

    # ---- aggregate the stability metric --------------------------------- #
    # mean within-seed std (the dream-induced instability) per kind.
    within_std = {k: [s["std"] for s in within[k]] for k in KINDS}
    within_mean = {k: [s["mean"] for s in within[k]] for k in KINDS}
    # paired: in how many seeds is PC's within-seed std LOWER (more stable)?
    std_diffs = [pc - gru for pc, gru in zip(within_std["PC"], within_std["GRU"])]
    pc_more_stable = sum(1 for d in std_diffs if d < 0)
    mean_diffs = [pc - gru for pc, gru in zip(within_mean["PC"], within_mean["GRU"])]
    pc_lower_mean = sum(1 for d in mean_diffs if d < 0)

    summary = {
        "within_seed_std": {k: _stats(within_std[k]) for k in KINDS},
        "within_seed_mean": {k: _stats(within_mean[k]) for k in KINDS},
        "pooled_forgetting": {k: _stats(pooled[k]) for k in KINDS},
        "stability_edge": {
            "std_diff_pc_minus_gru_mean": statistics.fmean(std_diffs),
            "pc_more_stable_in": f"{pc_more_stable}/{len(std_diffs)}",
            "std_diffs": [round(d, 5) for d in std_diffs],
            "mean_diff_pc_minus_gru": statistics.fmean(mean_diffs),
            "pc_lower_mean_in": f"{pc_lower_mean}/{len(mean_diffs)}",
        },
        "per_seed": {k: within[k] for k in KINDS},
    }
    report = {
        "config": dict(seeds=list(seeds), n_dreams=n_dreams, buf=buf, n_seq=n_seq,
                       seq_len=seq_len, d_in=d_in, epochs=epochs, batch=batch,
                       lr=lr, replay_batch=replay_batch, warmup=warmup),
        "summary": summary,
        "seconds": round(time.time() - t0, 1),
    }
    _save(report)
    _print(report)
    return report


def _save(report: Dict) -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "report_genreplay_stability.json"), "w") as f:
        json.dump(report, f, indent=2)

    s = report["summary"]
    cfg = report["config"]
    e = s["stability_edge"]
    lines = ["# PC-Liquid-Core generative-replay STABILITY probe", ""]
    lines.append(f"Seeds: `{cfg['seeds']}` (n={len(cfg['seeds'])})  |  "
                 f"{cfg['n_dreams']} dreams/seed at scarce buffer={cfg['buf']} "
                 f"(warmup {cfg['warmup']})  |  runtime {report['seconds']}s")
    lines.append("")
    lines.append("Isolation: per (seed,kind) train A ONCE, snapshot post-A, then "
                 "repeat N dreams -- each restores the IDENTICAL snapshot and trains "
                 "B with an IDENTICAL training RNG, so the ONLY varying factor is the "
                 "dream content. Within-seed std of forgetting = pure dream-induced "
                 "instability. Lower std = more reliable dreams.")
    lines.append("")
    lines.append("| kind | within-seed forgetting std (mean over seeds) | within-seed forgetting mean | pooled forgetting |")
    lines.append("|---|---:|---:|---:|")
    for k in KINDS:
        ws = s["within_seed_std"][k]
        wm = s["within_seed_mean"][k]
        pf = s["pooled_forgetting"][k]
        lines.append(
            f"| {k} | {ws['mean']:.5f} +/- {ws['std']:.5f} | "
            f"{wm['mean']:+.5f} | {pf['mean']:+.5f} +/- {pf['std']:.5f} |"
        )
    lines.append("")
    lines.append(f"**Stability edge (paired, within-seed std):** PC more stable "
                 f"(tighter dream-to-dream forgetting) in **{e['pc_more_stable_in']}** "
                 f"seeds; mean std diff PC-GRU = **{e['std_diff_pc_minus_gru_mean']:+.5f}** "
                 f"(negative = PC more stable).")
    lines.append("")
    lines.append(f"**Mean forgetting (paired):** PC lower in {e['pc_lower_mean_in']} "
                 f"seeds; mean diff PC-GRU = {e['mean_diff_pc_minus_gru']:+.5f}.")
    lines.append("")
    with open(os.path.join(here, "report_genreplay_stability.md"), "w") as f:
        f.write("\n".join(lines))


def _print(report: Dict) -> None:
    s = report["summary"]
    e = s["stability_edge"]
    print("\n=== GENERATIVE-REPLAY STABILITY PROBE (dream-for-dream) ===")
    for k in KINDS:
        ws = s["within_seed_std"][k]
        wm = s["within_seed_mean"][k]
        print(f"{k:3s}  within-seed std={ws['mean']:.5f}+/-{ws['std']:.5f}  "
              f"within-seed mean forget={wm['mean']:+.5f}")
    print(f"PC more stable (lower within-seed std) in {e['pc_more_stable_in']}  "
          f"(mean std diff PC-GRU {e['std_diff_pc_minus_gru_mean']:+.5f})")
    print(f"PC lower mean forgetting in {e['pc_lower_mean_in']}  "
          f"(mean diff {e['mean_diff_pc_minus_gru']:+.5f})")


if __name__ == "__main__":
    main()
