"""
experiments/liquid_pc/abc_continual.py — does PCLiquidCore's dream STABILITY turn
into a usable advantage under MULTI-task (A->B->C) generative-replay continual
learning, where dream drift can COMPOUND?

WHY THIS EXISTS
---------------
Retests #8-#10 established, on a single A->B transition, that PCLiquidCore's
generative-replay dreams are more SELF-CONSISTENT and more STABLE (lower dream-to-
dream forgetting variance) than a GRU's -- and that the GRU occasionally produces
a "toxic dream" that blows its forgetting up 3x while PC stays flat. On one
transition the ABSOLUTE forgetting gap stayed small because both models nearly
solved the single-step forgetting.

The hypothesis to TEST here (NOT to assume): under multi-task sequential learning
the dream is chained -- after each task the CURRENT model (which learned earlier
tasks only via its own dreams) dreams the buffer for the NEXT task. Errors in the
dream therefore feed the next dream (a dream-of-a-dream), so any per-step
instability can COMPOUND across tasks. If PC's dreams are genuinely more stable,
its worst-retained-task error after A->B->C should stay flat while the GRU's
worst task is at higher risk of collapsing. Equally possible (and to be reported
honestly): the gap stays modest, or the chaining hurts both equally.

PROTOCOL (Deep Generative Replay / "scholar", Shin et al. 2017):
  1. train A; record a_after_A; model_A dreams buffer D1 (~A).
  2. train B rehearsing D1; record b_after_B; model_AB dreams buffer D2 (~A+B).
  3. train C rehearsing D2; record final errors on held-out A, B, C.
Forgetting_A = a_final - a_after_A;  Forgetting_B = b_final - b_after_B;
C is last so it has no forgetting. Worst-task = max(a_final, b_final, c_final)
is the "is the whole system still usable" number. Same procedure, same dream
mechanism, same buffer for PC and GRU -> a fair test of dream QUALITY at depth.

Run:  python -m experiments.liquid_pc.abc_continual
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

from experiments.liquid_pc.data import train_test_split, three_regimes
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
        "max": max(xs),
        "n": len(xs),
        "vals": [round(v, 5) for v in xs],
    }


def main(
    *,
    seeds=(0, 1, 2, 3, 4),
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
    rec = {k: {"forget_A": [], "forget_B": [], "mean_forget": [],
               "a_final": [], "b_final": [], "c_final": [], "worst_final": []}
           for k in KINDS}

    for seed in seeds:
        a, b, c = three_regimes(n_seq, seq_len, d_in, seed=seed)
        a_tr, a_te = train_test_split(a, 0.8)
        b_tr, b_te = train_test_split(b, 0.8)
        c_tr, c_te = train_test_split(c, 0.8)

        for kind in KINDS:
            m = make_model(kind, d_in, seed)

            # --- task A ---------------------------------------------------- #
            train(m, a_tr, epochs=epochs, batch=batch, lr=lr, seed=seed)
            a_after_A = one_step_mse(m, a_te)
            d1 = generate_synthetic(m, n_syn=buf, seq_len=seq_len, d_in=d_in,
                                    seed=seed + 7, warmup=warmup)

            # --- task B (rehearse dreams of A) ----------------------------- #
            train(m, b_tr, epochs=epochs, batch=batch, lr=lr, seed=seed + 1,
                  replay_x=d1, replay_batch=min(replay_batch, buf))
            b_after_B = one_step_mse(m, b_te)
            # current model now knows A+B -> dream the A+B buffer for stage C.
            d2 = generate_synthetic(m, n_syn=buf, seq_len=seq_len, d_in=d_in,
                                    seed=seed + 17, warmup=warmup)

            # --- task C (rehearse dreams of A+B) --------------------------- #
            train(m, c_tr, epochs=epochs, batch=batch, lr=lr, seed=seed + 2,
                  replay_x=d2, replay_batch=min(replay_batch, buf))

            a_final = one_step_mse(m, a_te)
            b_final = one_step_mse(m, b_te)
            c_final = one_step_mse(m, c_te)
            fA = a_final - a_after_A
            fB = b_final - b_after_B
            worst = max(a_final, b_final, c_final)

            rec[kind]["forget_A"].append(fA)
            rec[kind]["forget_B"].append(fB)
            rec[kind]["mean_forget"].append(0.5 * (fA + fB))
            rec[kind]["a_final"].append(a_final)
            rec[kind]["b_final"].append(b_final)
            rec[kind]["c_final"].append(c_final)
            rec[kind]["worst_final"].append(worst)
            print(f"[seed {seed}] {kind:3s} fA={fA:+.5f} fB={fB:+.5f} "
                  f"a={a_final:.5f} b={b_final:.5f} c={c_final:.5f} "
                  f"WORST={worst:.5f}", flush=True)

    summary = {k: {key: _stats(rec[k][key]) for key in rec[k]} for k in KINDS}
    # paired PC-vs-GRU on the two headline numbers.
    def _paired(key):
        diffs = [pc - gru for pc, gru in zip(rec["PC"][key], rec["GRU"][key])]
        wins = sum(1 for d in diffs if d < 0)
        return {"mean_diff_pc_minus_gru": statistics.fmean(diffs),
                "pc_lower_in": f"{wins}/{len(diffs)}",
                "diffs": [round(d, 5) for d in diffs]}
    edge = {key: _paired(key) for key in ("mean_forget", "worst_final")}

    report = {
        "config": dict(seeds=list(seeds), buf=buf, n_seq=n_seq, seq_len=seq_len,
                       d_in=d_in, epochs=epochs, batch=batch, lr=lr,
                       replay_batch=replay_batch, warmup=warmup),
        "summary": summary,
        "edge_pc_minus_gru": edge,
        "seconds": round(time.time() - t0, 1),
    }
    _save(report)
    _print(report)
    return report


def _save(report: Dict) -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "report_abc_continual.json"), "w") as f:
        json.dump(report, f, indent=2)

    s = report["summary"]
    cfg = report["config"]
    e = report["edge_pc_minus_gru"]
    lines = ["# PC-Liquid-Core A->B->C generative-replay continual learning", ""]
    lines.append(f"Seeds: `{cfg['seeds']}` (n={len(cfg['seeds'])})  |  "
                 f"chained dream buffer={cfg['buf']} (warmup {cfg['warmup']})  |  "
                 f"runtime {report['seconds']}s")
    lines.append("")
    lines.append("Deep Generative Replay (scholar): train A; dream D1(~A); train B "
                 "rehearsing D1; dream D2(~A+B); train C rehearsing D2. Forgetting is "
                 "measured vs each task's error right after it was learned. "
                 "`worst_final` = max held-out 1-step MSE across A,B,C = the "
                 "'is the whole system still usable' number (lower=better).")
    lines.append("")
    lines.append("| kind | forget A | forget B | mean forget | A final | B final | C final | WORST task |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for k in KINDS:
        v = s[k]
        lines.append(
            f"| {k} | {v['forget_A']['mean']:+.5f} | {v['forget_B']['mean']:+.5f} | "
            f"{v['mean_forget']['mean']:+.5f} +/- {v['mean_forget']['std']:.5f} | "
            f"{v['a_final']['mean']:.5f} | {v['b_final']['mean']:.5f} | "
            f"{v['c_final']['mean']:.5f} | "
            f"{v['worst_final']['mean']:.5f} +/- {v['worst_final']['std']:.5f} "
            f"(max {v['worst_final']['max']:.5f}) |"
        )
    lines.append("")
    lines.append(f"**Paired PC-GRU mean forgetting:** diff "
                 f"{e['mean_forget']['mean_diff_pc_minus_gru']:+.5f}, PC lower in "
                 f"{e['mean_forget']['pc_lower_in']}.")
    lines.append("")
    lines.append(f"**Paired PC-GRU WORST-task final MSE:** diff "
                 f"{e['worst_final']['mean_diff_pc_minus_gru']:+.5f}, PC lower in "
                 f"{e['worst_final']['pc_lower_in']}.  (worst-task is the usability "
                 f"number; large positive GRU worst = collapse.)")
    lines.append("")
    with open(os.path.join(here, "report_abc_continual.md"), "w") as f:
        f.write("\n".join(lines))


def _print(report: Dict) -> None:
    s = report["summary"]
    e = report["edge_pc_minus_gru"]
    print("\n=== A->B->C GENERATIVE-REPLAY CONTINUAL LEARNING ===")
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
