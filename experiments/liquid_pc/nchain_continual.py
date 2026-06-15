"""
experiments/liquid_pc/nchain_continual.py — does PC's bounded-failure dream edge
turn into a DIVERGENCE at GREATER continual-learning DEPTH (N tasks, N>3)?

WHY THIS EXISTS
---------------
Retest #11 (abc_continual.py) tested the strategic hypothesis "GRU collapses
exponentially as tasks accumulate" on a 3-task chain and FALSIFIED the dramatic
version: at full epochs neither model collapsed, GRU even won mean forgetting,
and PC's only edge was a small-but-5/5-consistent worst-task advantage. The
honest open question that left: depth=3 may simply be too shallow for chained
dream drift to accumulate. This script pushes the SAME mechanism to a deeper
chain (default N=5) to see whether:
  (a) the gap stays flat/modest (the bounded-failure picture extends), or
  (b) a divergence finally appears -- GRU's worst-task creeps up with depth
      while PC stays flat (the compounding-drift picture), or
  (c) BOTH degrade with depth (chaining hurts the mechanism itself).
All three are reportable; we do NOT assume (b).

PROTOCOL (Deep Generative Replay / "scholar", Shin et al. 2017), generalized to N:
  for i in 0..N-1:
    train task i, rehearsing the dream-buffer of all prior tasks (D_{i-1});
    record err_after[i] = held-out 1-step MSE right after learning task i;
    the current model (knows tasks 0..i via its own dreams) dreams D_i (~0..i).
  final: measure held-out 1-step MSE on every task.
Forgetting_i = final_i - err_after[i] for i < N-1 (last task has none).
worst_final = max_i final_i = the "is the whole system still usable" number.
Same dream mechanism, same buffer, same regimes for PC and GRU -> fair test of
dream QUALITY at depth.

Run:  python -m experiments.liquid_pc.nchain_continual
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

from experiments.liquid_pc.data import SignalSpec, make_signal, train_test_split
from experiments.liquid_pc.model import PCLiquidCore, GRUBaseline
from experiments.liquid_pc.run import train, one_step_mse, generate_synthetic


# Five well-separated timescale regimes, slow -> fast. Distinct enough that
# naive sequential training catastrophically overwrites earlier regimes unless
# old knowledge is rehearsed via dreams.
REGIME_BANDS = [
    (40.0, 100.0),  # T0 very slow
    (22.0, 55.0),   # T1 slow
    (11.0, 28.0),   # T2 mid
    (6.0, 14.0),    # T3 fast
    (2.5, 7.0),     # T4 very fast
]

KINDS = ["PC", "GRU"]


def make_regimes(n_seq: int, seq_len: int, d_in: int, n_tasks: int, seed: int):
    """N timescale regimes, each with a well-separated per-task seed offset."""
    out = []
    for i in range(n_tasks):
        pmin, pmax = REGIME_BANDS[i % len(REGIME_BANDS)]
        spec = SignalSpec(period_min=pmin, period_max=pmax, amp_decay=0.6)
        out.append(make_signal(n_seq, seq_len, d_in, spec=spec,
                               seed=seed + 1000 * i))
    return out


def make_model(kind: str, d_in: int, seed: int):
    torch.manual_seed(seed)
    if kind == "PC":
        return PCLiquidCore(d_in, d=48, n_levels=3, dynamic_precision=True,
                            use_astrocyte=True)
    return GRUBaseline(d_in, hidden=70)


def _stats(xs: List[float]) -> Dict:
    return {
        "mean": statistics.fmean(xs),
        "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
        "max": max(xs),
        "n": len(xs),
        "vals": [round(v, 5) for v in xs],
    }


def main(
    *,
    seeds=(0, 1, 2, 3, 4),
    n_tasks: int = 5,
    buf: int = 16,
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
    # per kind: lists over seeds.
    rec = {k: {"mean_forget": [], "worst_final": [], "final_per_task": [],
               "worst_task_idx": []}
           for k in KINDS}

    for seed in seeds:
        regimes = make_regimes(n_seq, seq_len, d_in, n_tasks, seed)
        splits = [train_test_split(r, 0.8) for r in regimes]  # [(tr,te),...]

        for kind in KINDS:
            m = make_model(kind, d_in, seed)
            after = [0.0] * n_tasks
            dream_prev = None  # dream buffer of all prior tasks

            for i in range(n_tasks):
                tr_i, te_i = splits[i]
                kw = {}
                if dream_prev is not None:
                    kw = dict(replay_x=dream_prev,
                              replay_batch=min(replay_batch, buf))
                train(m, tr_i, epochs=epochs, batch=batch, lr=lr,
                      seed=seed + i, **kw)
                after[i] = one_step_mse(m, te_i)
                # current model now knows tasks 0..i -> dream that buffer.
                dream_prev = generate_synthetic(
                    m, n_syn=buf, seq_len=seq_len, d_in=d_in,
                    seed=seed + 7 + 13 * i, warmup=warmup)

            final = [one_step_mse(m, splits[i][1]) for i in range(n_tasks)]
            # forgetting only defined for tasks before the last.
            forgets = [final[i] - after[i] for i in range(n_tasks - 1)]
            worst = max(final)
            worst_idx = int(max(range(n_tasks), key=lambda i: final[i]))

            rec[kind]["mean_forget"].append(statistics.fmean(forgets))
            rec[kind]["worst_final"].append(worst)
            rec[kind]["final_per_task"].append([round(x, 5) for x in final])
            rec[kind]["worst_task_idx"].append(worst_idx)
            print(f"[seed {seed}] {kind:3s} "
                  f"mean_forget={statistics.fmean(forgets):+.5f} "
                  f"WORST={worst:.5f} (task {worst_idx}) "
                  f"final={['%.4f' % x for x in final]}", flush=True)

    summary = {k: {"mean_forget": _stats(rec[k]["mean_forget"]),
                   "worst_final": _stats(rec[k]["worst_final"]),
                   "final_per_task": rec[k]["final_per_task"],
                   "worst_task_idx": rec[k]["worst_task_idx"]}
               for k in KINDS}

    def _paired(key):
        diffs = [pc - gru for pc, gru in zip(rec["PC"][key], rec["GRU"][key])]
        wins = sum(1 for d in diffs if d < 0)
        return {"mean_diff_pc_minus_gru": statistics.fmean(diffs),
                "pc_lower_in": f"{wins}/{len(diffs)}",
                "diffs": [round(d, 5) for d in diffs]}
    edge = {key: _paired(key) for key in ("mean_forget", "worst_final")}

    report = {
        "config": dict(seeds=list(seeds), n_tasks=n_tasks, buf=buf, n_seq=n_seq,
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
    with open(os.path.join(here, "report_nchain_continual.json"), "w") as f:
        json.dump(report, f, indent=2)

    s = report["summary"]
    cfg = report["config"]
    e = report["edge_pc_minus_gru"]
    lines = [f"# PC-Liquid-Core {cfg['n_tasks']}-task generative-replay "
             f"continual learning (deep chain)", ""]
    lines.append(f"Seeds: `{cfg['seeds']}` (n={len(cfg['seeds'])})  |  "
                 f"{cfg['n_tasks']} tasks  |  chained dream buffer={cfg['buf']} "
                 f"(warmup {cfg['warmup']})  |  runtime {report['seconds']}s")
    lines.append("")
    lines.append(f"Deep Generative Replay (scholar) over {cfg['n_tasks']} "
                 "well-separated timescale regimes (slow->fast). After each task "
                 "the current model dreams a buffer of all tasks seen so far; the "
                 "next task rehearses it. Forgetting is measured vs each task's "
                 "error right after it was learned. `worst_final` = max held-out "
                 "1-step MSE across all tasks = the 'is the whole system still "
                 "usable' number (lower=better).")
    lines.append("")
    lines.append("| kind | mean forget | WORST task final MSE |")
    lines.append("|---|---:|---:|")
    for k in KINDS:
        v = s[k]
        lines.append(
            f"| {k} | {v['mean_forget']['mean']:+.5f} +/- "
            f"{v['mean_forget']['std']:.5f} | "
            f"{v['worst_final']['mean']:.5f} +/- {v['worst_final']['std']:.5f} "
            f"(max {v['worst_final']['max']:.5f}) |")
    lines.append("")
    lines.append(f"**Paired PC-GRU mean forgetting:** diff "
                 f"{e['mean_forget']['mean_diff_pc_minus_gru']:+.5f}, PC lower in "
                 f"{e['mean_forget']['pc_lower_in']}.")
    lines.append("")
    lines.append(f"**Paired PC-GRU WORST-task final MSE:** diff "
                 f"{e['worst_final']['mean_diff_pc_minus_gru']:+.5f}, PC lower in "
                 f"{e['worst_final']['pc_lower_in']}.  (worst-task is the "
                 f"usability number; large positive GRU worst = collapse.)")
    lines.append("")
    lines.append("Per-seed worst-task index (which timescale regime is the "
                 "system's weakest link):")
    lines.append(f"- PC : {s['PC']['worst_task_idx']}")
    lines.append(f"- GRU: {s['GRU']['worst_task_idx']}")
    lines.append("")
    with open(os.path.join(here, "report_nchain_continual.md"), "w") as f:
        f.write("\n".join(lines))


def _print(report: Dict) -> None:
    s = report["summary"]
    e = report["edge_pc_minus_gru"]
    cfg = report["config"]
    print(f"\n=== {cfg['n_tasks']}-TASK GENERATIVE-REPLAY CONTINUAL LEARNING ===")
    for k in KINDS:
        v = s[k]
        print(f"{k:3s}  mean_forget={v['mean_forget']['mean']:+.5f}"
              f"+/-{v['mean_forget']['std']:.5f}  "
              f"WORST={v['worst_final']['mean']:.5f}"
              f"+/-{v['worst_final']['std']:.5f} (max {v['worst_final']['max']:.5f})")
    print(f"PC-GRU mean_forget diff {e['mean_forget']['mean_diff_pc_minus_gru']:+.5f} "
          f"(PC lower in {e['mean_forget']['pc_lower_in']})")
    print(f"PC-GRU WORST diff {e['worst_final']['mean_diff_pc_minus_gru']:+.5f} "
          f"(PC lower in {e['worst_final']['pc_lower_in']})")


if __name__ == "__main__":
    main()
