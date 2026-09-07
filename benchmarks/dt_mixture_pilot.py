"""Δt-mixture training pilot — a RECIPE question, not a mechanism claim.

Context. Every architecture (liquid, RNN, GRU-D, transformer) degrades
~10× on the Δt-shift tier (train gaps 0.05, test gaps 0.4 — BENCHMARKS §6/
§8). The mechanism line is closed at three levels (Δt-feature wiring /
decay rate / attractor ablation), so the remaining practical question is
a TRAINING-RECIPE one: if the model simply SEES multiple gap scales at
training time, does the shift degradation disappear — for every
architecture equally, or differentially?

Design (within-pilot control, exploratory, NO pre-registered claim):
  * narrow arm  — train on dt_mean=0.05 cv=1 only (the formal extrap arm);
  * mixed arm   — train on an equal mixture of dt_mean {0.05, 0.15, 0.4}
                  cv=1, SAME total window count;
  * identical eval for both arms: the shift tier (dt 0.4, Δt channel in
    the narrow arm's units) and the narrow in-distribution tier;
  * sensor/target standardisation frozen to the NARROW arm's train stats
    for both arms, so preprocessing cannot explain a difference;
  * 5 archs × 2 arms × 5 seeds, 12 epochs, CPU.

Reads as: (a) does mixture training repair the shift tier (mixed vs
narrow, per arch, Welch t)? (b) does it change the architecture ranking
on the shift tier? (c) does it cost anything in-distribution?

    python benchmarks/dt_mixture_pilot.py \
        --out benchmarks/results/dt_mixture_pilot.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from benchmarks import synth_ct_control as scc
from benchmarks.battery_soh_edge import build


def _a(train_windows):
    class A:
        pass
    a = A()
    a.mu, a.dt_fine, a.t_end = 2.0, 0.002, 120.0
    a.n_samples, a.horizon = 48, 0.5
    return a


def make_set(a, dt_mean, n, rng, dt_norm):
    return scc.build_dataset(a, dt_mean, 1.0, rng, n, dt_norm=dt_norm)


def train_model(arch, Xtr, ytr_n, d_model, n_layers, epochs, lr, batch,
                seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = build(arch, d_model, n_layers, Xtr.shape[1])
    m.inp = nn.Linear(Xtr.shape[-1], d_model)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr_n)
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch):
            idx = perm[i:i + batch]
            opt.zero_grad(set_to_none=True)
            loss = nn.MSELoss()(m(Xt[idx]), yt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
    return m


def rmse_raw(m, X, y, ymu, ysd):
    m.eval()
    with torch.no_grad():
        p = m(torch.from_numpy(X)).numpy()
    return float(np.sqrt(((p * ysd + ymu - y) ** 2).mean()))


def welch(a, b):
    se = math.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
    return (np.mean(b) - np.mean(a)) / se if se else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-arm-windows", type=int, default=1500)
    ap.add_argument("--eval-windows", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--archs", default="mt_lnn,mt_lnn_dt,gru_d,gru,lstm")
    ap.add_argument("--dev", default="cpu")
    ap.add_argument("--out",
                    default="benchmarks/results/dt_mixture_pilot.json")
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    archs = [a for a in args.archs.split(",") if a]
    a = _a(args.per_arm_windows)

    # ---- data. Δt channel always in NARROW-arm units (0.05). -----------
    rng = np.random.default_rng(777)
    Xn, yn = make_set(a, 0.05, args.per_arm_windows, rng, dt_norm=0.05)
    parts = []
    n3 = args.per_arm_windows // 3
    for dm in (0.05, 0.15, 0.4):
        Xp, yp = make_set(a, dm, n3, rng, dt_norm=0.05)
        parts.append((Xp, yp))
    Xtr_m = np.concatenate([p[0] for p in parts] + [Xn[n3 * 3:]])
    ytr_m = np.concatenate([p[1] for p in parts] + [yn[n3 * 3:]])
    Xs, ys = make_set(a, 0.4, args.eval_windows, rng, dt_norm=0.05)  # shift
    Xe, ye = make_set(a, 0.05, args.eval_windows, rng, dt_norm=0.05)  # in-dist

    # Standardisation frozen to the NARROW arm for both arms.
    mu_f = Xn[..., :-1].reshape(-1, Xn.shape[-1] - 1).mean(0)
    sd_f = Xn[..., :-1].reshape(-1, Xn.shape[-1] - 1).std(0) + 1e-8

    def prep(X):
        out = X.copy()
        out[..., :-1] = ((X[..., :-1] - mu_f) / sd_f).astype(np.float32)
        return out.astype(np.float32)

    ymu, ysd = float(yn.mean()), float(yn.std() + 1e-8)
    arms = {
        "narrow": (prep(Xn), ((yn - ymu) / ysd).astype(np.float32)),
        "mixed": (prep(Xtr_m), ((ytr_m - ymu) / ysd).astype(np.float32)),
    }
    ev = {"shift_dt0.4": (prep(Xs), ys), "in_dist_dt0.05": (prep(Xe), ye)}
    print(f"Δt-mixture pilot | arms narrow({len(Xn)}) / mixed({len(Xtr_m)})"
          f" | eval shift {len(Xs)} / in-dist {len(Xe)} | {args.epochs} ep"
          f" | archs {','.join(archs)} | dev {args.dev}"
          f" | EXPLORATORY, no pre-registered claim")

    res = {}
    t_all = time.time()
    for arch in archs:
        res[arch] = {}
        for arm, (Xtr, ytr_n) in arms.items():
            vals = {"shift_dt0.4": [], "in_dist_dt0.05": []}
            for s in seeds:
                t0 = time.time()
                m = train_model(arch, Xtr, ytr_n, 78, 2, args.epochs,
                                3e-3, 32, s)
                for k, (X, y) in ev.items():
                    vals[k].append(rmse_raw(m, X, y, ymu, ysd))
                print(f"  {arch:10s} {arm:6s} seed {s}: shift "
                      f"{vals['shift_dt0.4'][-1]:.4f} | in-dist "
                      f"{vals['in_dist_dt0.05'][-1]:.4f} "
                      f"({time.time() - t0:.0f}s)")
            res[arch][arm] = {k: [round(v, 4) for v in vs]
                              for k, vs in vals.items()}
            res[arch][arm]["shift_mean"] = round(
                float(np.mean(vals["shift_dt0.4"])), 4)
            res[arch][arm]["in_dist_mean"] = round(
                float(np.mean(vals["in_dist_dt0.05"])), 4)
        t = welch(res[arch]["mixed"]["shift_dt0.4"],
                  res[arch]["narrow"]["shift_dt0.4"])
        res[arch]["welch_t_shift_mixed_vs_narrow"] = round(t, 3)
        print(f"  {arch:10s} shift: narrow "
              f"{res[arch]['narrow']['shift_mean']:.4f} -> mixed "
              f"{res[arch]['mixed']['shift_mean']:.4f} "
              f"(t={t:+.2f}); in-dist: "
              f"{res[arch]['narrow']['in_dist_mean']:.4f} -> "
              f"{res[arch]['mixed']['in_dist_mean']:.4f}\n")

    ranking = sorted(archs, key=lambda x: res[x]["mixed"]["shift_mean"])
    print(f"{'arch':12s}{'shift(narrow)':>14}{'shift(mixed)':>14}"
          f"{'in-dist(mixed)':>16}")
    for x in ranking:
        print(f"{x:12s}{res[x]['narrow']['shift_mean']:>14.4f}"
              f"{res[x]['mixed']['shift_mean']:>14.4f}"
              f"{res[x]['mixed']['in_dist_mean']:>16.4f}")
    print(f"\ntotal {time.time() - t_all:.0f}s | exploratory pilot")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"purpose": "Δt-mixture training pilot — recipe question, "
                              "exploratory, no pre-registered claim",
                   "config": {"per_arm_windows": args.per_arm_windows,
                              "eval_windows": args.eval_windows,
                              "epochs": args.epochs, "seeds": seeds,
                              "archs": archs, "dev": args.dev,
                              "data_seed": 777,
                              "mixture": "equal {0.05,0.15,0.4} cv=1, "
                                         "dt channel in narrow units"},
                   "results": res,
                   "shift_ranking_mixed": ranking}, f, indent=2)
    print(f"results -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
