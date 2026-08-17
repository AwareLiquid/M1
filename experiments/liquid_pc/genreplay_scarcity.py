"""
experiments/liquid_pc/genreplay_scarcity.py — does PCLiquidCore's generative-replay
edge WIDEN as dreaming gets harder (a scarce replay budget)?

WHY THIS EXISTS
---------------
Retest #8 (report_genreplay.md) found the first PC-specific forgetting signal:
PCLiquidCore's DREAMS (generative-replay buffer) reduce forgetting more than a
GRU's dreams (-0.0142 paired, 4/5) at a 32-sequence buffer. But at that budget
BOTH models nearly solved forgetting, so the absolute gap was tiny. If PC's
better dreams are genuinely meaningful, the edge should GROW when the dream buffer
is SCARCE — i.e. when dream quality has to carry more of the load. This script
tests exactly that by sweeping the buffer size.

A tried-and-rejected alternative (recorded for honesty): training task A with
scheduled sampling to explicitly improve "dream fidelity" moved forgetting only
~5% on 1 seed and did NOT widen PC's edge — and the dream's free-running MSE on
REAL A stayed high (~0.6), revealing that generative replay works via SELF-
CONSISTENCY on the model's own manifold, not fidelity to A. So the right question
is robustness under scarcity, not a fidelity auxiliary.

Protocol (fair across buffer sizes): per (seed, kind) train A ONCE, snapshot the
post-A weights, dream ONE max-size buffer; then for each buffer size, RESTORE the
snapshot and train B rehearsing the first `buf` dreamed sequences. So every buffer
size starts from the identical post-A model and the identical dreams (subsetted).

Run:  python -m experiments.liquid_pc.genreplay_scarcity
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
from experiments.liquid_pc.model import PCLiquidCore, GRUBaseline, count_params
from experiments.liquid_pc.run import train, one_step_mse, generate_synthetic


def make_model(kind: str, d_in: int, seed: int):
    torch.manual_seed(seed)
    if kind == "PC":
        return PCLiquidCore(d_in, d=48, n_levels=3, dynamic_precision=True,
                            use_astrocyte=True)
    return GRUBaseline(d_in, hidden=70)


KINDS = ["PC", "GRU"]


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
    buffers=(4, 8, 16, 32),
    n_seq: int = 320,
    seq_len: int = 96,
    d_in: int = 1,
    epochs: int = 40,
    batch: int = 64,
    lr: float = 3e-3,
    replay_batch: int = 16,
    warmup: int = 16,
) -> Dict:
    t0 = time.time()
    max_buf = max(buffers)
    # forgetting[kind][buf] = list over seeds.
    forget = {k: {b: [] for b in buffers} for k in KINDS}
    bafter = {k: {b: [] for b in buffers} for k in KINDS}

    for seed in seeds:
        a, b = two_regimes(n_seq, seq_len, d_in, seed=seed)
        a_tr, a_te = train_test_split(a, 0.8)
        b_tr, b_te = train_test_split(b, 0.8)

        for kind in KINDS:
            m = make_model(kind, d_in, seed)
            train(m, a_tr, epochs=epochs, batch=batch, lr=lr, seed=seed)
            a_before = one_step_mse(m, a_te)
            # dream ONE max-size buffer + snapshot the post-A weights.
            dream = generate_synthetic(m, n_syn=max_buf, seq_len=seq_len,
                                       d_in=d_in, seed=seed + 7, warmup=warmup)
            snapshot = copy.deepcopy(m.state_dict())

            for buf in buffers:
                m.load_state_dict(snapshot)              # identical post-A start
                rbuf = dream[:buf]
                train(m, b_tr, epochs=epochs, batch=batch, lr=lr, seed=seed + 1,
                      replay_x=rbuf, replay_batch=min(replay_batch, buf))
                a_after = one_step_mse(m, a_te)
                forget[kind][buf].append(a_after - a_before)
                bafter[kind][buf].append(one_step_mse(m, b_te))
                print(f"[seed {seed}] {kind:3s} buf={buf:3d} "
                      f"forget={a_after - a_before:+.5f}", flush=True)

    # summarise + paired PC-vs-GRU edge at each buffer size.
    summary = {k: {b: {"forgetting": _mean_std(forget[k][b]),
                       "b_after": _mean_std(bafter[k][b])}
                   for b in buffers} for k in KINDS}
    edge = {}
    for buf in buffers:
        diffs = [pc - gru for pc, gru in zip(forget["PC"][buf], forget["GRU"][buf])]
        wins = sum(1 for d in diffs if d < 0)            # PC forgets less
        edge[buf] = {"mean_diff": statistics.fmean(diffs),
                     "pc_lower_in": f"{wins}/{len(diffs)}",
                     "diffs": [round(d, 5) for d in diffs]}

    report = {
        "config": dict(seeds=list(seeds), buffers=list(buffers), n_seq=n_seq,
                       seq_len=seq_len, d_in=d_in, epochs=epochs, batch=batch,
                       lr=lr, replay_batch=replay_batch, warmup=warmup),
        "summary": summary,
        "edge_pc_minus_gru": edge,
        "seconds": round(time.time() - t0, 1),
    }
    _save(report)
    _print(report)
    return report


def _save(report: Dict) -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "report_genreplay_scarcity.json"), "w") as f:
        json.dump(report, f, indent=2)

    s = report["summary"]
    cfg = report["config"]
    e = report["edge_pc_minus_gru"]
    buffers = cfg["buffers"]
    lines = ["# PC-Liquid-Core generative-replay SCARCITY sweep", ""]
    lines.append(f"Seeds: `{cfg['seeds']}` (n={len(cfg['seeds'])})  |  "
                 f"buffers {buffers} (warmup {cfg['warmup']})  |  "
                 f"runtime {report['seconds']}s")
    lines.append("")
    lines.append("Does PCLiquidCore's generative-replay forgetting edge over a GRU "
                 "WIDEN as the dream buffer shrinks? Lower `forgetting` is better.")
    lines.append("")
    lines.append("| buffer | PC forgetting | GRU forgetting | PC-GRU diff | PC-lower-in |")
    lines.append("|---:|---:|---:|---:|---:|")
    for buf in buffers:
        pc = s["PC"][buf]["forgetting"]
        gru = s["GRU"][buf]["forgetting"]
        lines.append(
            f"| {buf} | {pc['mean']:+.5f} +/- {pc['std']:.5f} | "
            f"{gru['mean']:+.5f} +/- {gru['std']:.5f} | "
            f"{e[buf]['mean_diff']:+.5f} | {e[buf]['pc_lower_in']} |"
        )
    lines.append("")
    with open(os.path.join(here, "report_genreplay_scarcity.md"), "w") as f:
        f.write("\n".join(lines))


def _print(report: Dict) -> None:
    e = report["edge_pc_minus_gru"]
    s = report["summary"]
    print("\n=== GENERATIVE-REPLAY SCARCITY SWEEP (PC-GRU forgetting edge) ===")
    for buf in report["config"]["buffers"]:
        print(f"buf={buf:3d}  PC={s['PC'][buf]['forgetting']['mean']:+.5f}  "
              f"GRU={s['GRU'][buf]['forgetting']['mean']:+.5f}  "
              f"diff={e[buf]['mean_diff']:+.5f}  (PC lower in {e[buf]['pc_lower_in']})")


if __name__ == "__main__":
    main()
