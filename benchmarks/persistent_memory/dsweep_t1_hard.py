#!/usr/bin/env python3
"""B-17 d-sweep driver: run ONLY the t1 hard cell (default 64,16) for the
parametric system across state dimensions.

Prereg (notes/overnight/backlog.md B-17): d in {256,1024,4096} x 3 seeds;
d=4096 hard-cell recall within +0.05 of d=256 -> "interference bottleneck"
(DEAD for the capacity hypothesis); >= +0.05 -> capacity cliff confirmed.

Full-grid runs at d=4096 would write tens of GB of d×d snapshots for no extra
information — the capacity hypothesis is tested at the hard cell alone. The
episode and scoring path is byte-identical to run_pmb.py's run_t1 (same
t1_grid generator, same run_episode persistence discipline); only the grid
iteration is restricted, so d=256 output must reproduce the committed full-grid
JSONs exactly (determinism gate).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_pmb import _run_in_tmp          # noqa: E402
from systems import make_system          # noqa: E402
from tasks import t1_grid                # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="B-17/B-19 parametric d-sweep, t1 hard cell")
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--system", default="parametric",
                   choices=["parametric", "micro_fw", "micro_rls"])
    p.add_argument("--update-rule", default="sum",
                   choices=["sum", "delta", "falcon_nlms", "kalman_delta"])
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--out_json", required=True)
    a = p.parse_args()

    def factory():
        return make_system(a.system, dim=a.dim, update_rule=a.update_rule)

    for params, episode in t1_grid(a.seed):
        if params["n_facts"] == a.n and params["k_distractors"] == a.k:
            cell = _run_in_tmp(factory, episode)
            break
    else:
        raise SystemExit(f"cell ({a.n},{a.k}) not in t1_grid for seed {a.seed}")

    report = {
        "benchmark": "pmb-v0",
        "system": a.system,
        "seed": a.seed,
        "config": {"update_rule": a.update_rule, "dim": a.dim,
                   "cell": [a.n, a.k], "driver": "dsweep_t1_hard.py (B-17/B-19)"},
        "results": {"t1_hard": {"task": "t1", "n": a.n, "k": a.k, **cell}},
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    Path(a.out_json).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
