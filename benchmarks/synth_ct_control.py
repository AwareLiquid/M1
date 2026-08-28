"""Irregular-sampling robustness, domain 3 of 3: synthetic continuous-time
control — the only domain where the continuous-time claim is attributable.

Why this exists. The battery and air-quality domains mix irregular sampling
with real-world confounds (sensor physics, station microclimate). Here the
data-generating process is FULLY KNOWN — Van der Pol oscillators integrated
with RK4 on a fine grid — and the sampling times are drawn from configurable
distributions. If a continuous-time architecture has an inductive advantage,
it must show up here, and only here can a failure be blamed on the
architecture rather than on confounds.

Design. Two axes are swept independently:
  * sampling DENSITY — mean Δt ∈ {0.05, 0.15, 0.4} time units;
  * sampling JITTER — cv(Δt) = 0.0 (equally spaced) or 1.0 (exponential,
    the memoryless/event-driven extreme; other cv values use a mean/cv-exact
    lognormal).
Window = 48 consecutive samples of (x, ẋ) + Δt appended as the LAST input
feature and supplied to EVERY architecture (the fairness guardrail); target =
x at a FIXED real-time lookahead of 0.5 beyond the last sample, so difficulty
is comparable across density tiers. Train and test use DISJOINT trajectories
(different initial conditions, batched RK4) and independent sampling
realisations; features/targets are standardised on train only.

Pre-registered judgement (fixed before any run, not movable after): the
robustness claim needs mt_lnn to beat gru_d AND lstm AND gru with Welch |t|
>= 2 at the high-irregularity tiers (cv=1) on at least 2 of the 3 task
domains (battery / air quality / synthetic). Anything less is recorded as a
negative.

    python benchmarks/synth_ct_control.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from benchmarks.battery_soh_edge import build

DT_MIN = 1e-3          # sampling gaps are clipped to the fine-grid step


def vdp(s, mu):
    x, v = s[..., 0], s[..., 1]
    return np.stack([v, mu * (1.0 - x * x) * v - x], axis=-1)


def simulate_batch(mu, t_end, dt_fine, n_traj, rng):
    """Batch of Van der Pol trajectories on a fine RK4 grid.

    Returns (times (n_steps+1,), states (n_traj, n_steps+1, 2)).
    """
    n = int(t_end / dt_fine)
    t = np.arange(n + 1) * dt_fine
    s = rng.uniform(-2.0, 2.0, size=(n_traj, 2))
    out = np.empty((n_traj, n + 1, 2))
    out[:, 0] = s
    for i in range(n):
        k1 = vdp(s, mu)
        k2 = vdp(s + dt_fine / 2 * k1, mu)
        k3 = vdp(s + dt_fine / 2 * k2, mu)
        k4 = vdp(s + dt_fine * k3, mu)
        s = s + dt_fine / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[:, i + 1] = s
    return t, out


def draw_dts(dt_mean, dt_cv, n, rng):
    """n gaps with the requested mean and cv.

    cv=0: equally spaced. cv=1: exponential (memoryless). Other values: a
    lognormal with exactly that mean and cv.
    """
    if dt_cv == 0.0:
        return np.full(n, dt_mean)
    if dt_cv == 1.0:
        return rng.exponential(dt_mean, n)
    sig = math.sqrt(math.log(1.0 + dt_cv ** 2))
    mu_l = math.log(dt_mean) - sig ** 2 / 2
    return np.clip(rng.lognormal(mu_l, sig, n), DT_MIN, None)


def windows_from_traj(times, states, dt_mean, dt_cv, n_samples, horizon, rng,
                      burn_in=5.0):
    """Walk one trajectory, drawing an independent sampling realisation per
    window. Returns (inputs (n, n_samples, 3), targets (n,))."""
    Xs, ys = [], []
    t0 = burn_in
    while True:
        dts = draw_dts(dt_mean, dt_cv, n_samples, rng)
        ts = np.concatenate([[t0], t0 + np.cumsum(dts)])
        if ts[-1] + horizon > times[-1]:
            break
        idx = np.clip(np.searchsorted(times, ts), 1, len(times) - 1)
        obs = states[idx]                                    # (n_samples, 2)
        dt_ch = np.concatenate([[dt_mean], np.diff(ts)]).astype(np.float32)
        Xs.append(np.concatenate([obs, (dt_ch / dt_mean)[:, None]],
                                 axis=1).astype(np.float32))
        ys.append(np.float32(
            np.interp(ts[-1] + horizon, times, states[:, 0])))
        t0 = ts[-1] + horizon + dt_mean                      # past the target
    return Xs, ys


def build_dataset(args, dt_mean, dt_cv, rng, n_windows):
    """Fresh trajectories + sampling realisations until n_windows exist."""
    per_batch = max(8, n_windows // 6)
    allX, ally = [], []
    while sum(map(len, allX)) < n_windows:
        t, st = simulate_batch(args.mu, args.t_end, args.dt_fine,
                               per_batch, rng)
        for i in range(per_batch):
            xs, ys = windows_from_traj(t, st[i], dt_mean, dt_cv,
                                       args.n_samples, args.horizon, rng)
            allX.extend(xs)
            ally.extend(ys)
    return np.stack(allX[:n_windows]), np.array(ally[:n_windows])


def run(arch, Xtr, ytr, Xte, yte, args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = build(arch, args.d_model, args.n_layers, Xtr.shape[1]).to(args.dev)
    m.inp = nn.Linear(Xtr.shape[-1], args.d_model).to(args.dev)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr)
    lossf = nn.MSELoss()
    Xtr_t = torch.from_numpy(Xtr).to(args.dev)
    ytr_t = torch.from_numpy(ytr).to(args.dev)
    Xte_t = torch.from_numpy(Xte).to(args.dev)
    m.train()
    n = len(Xtr_t)
    for _ in range(args.epochs):
        perm = torch.randperm(n, device=args.dev)
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad(set_to_none=True)
            loss = lossf(m(Xtr_t[idx]), ytr_t[idx])
            if not torch.isfinite(loss):
                return None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
    m.eval()
    with torch.no_grad():
        pred = m(Xte_t).cpu().numpy()
    return float(np.sqrt(((pred - yte) ** 2).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", default="mt_lnn,lstm,gru,gru_d,transformer")
    ap.add_argument("--mu", type=float, default=2.0,
                    help="Van der Pol nonlinearity")
    ap.add_argument("--dt-fine", type=float, default=0.002)
    ap.add_argument("--t-end", type=float, default=120.0,
                    help="simulation length per trajectory (time units)")
    ap.add_argument("--n-samples", type=int, default=48)
    ap.add_argument("--horizon", type=float, default=0.5)
    ap.add_argument("--dt-means", default="0.05,0.15,0.4")
    ap.add_argument("--dt-cvs", default="0.0,1.0")
    ap.add_argument("--train-windows", type=int, default=2500)
    ap.add_argument("--test-windows", type=int, default=800)
    ap.add_argument("--d_model", type=int, default=78)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--dev", default="cpu")
    ap.add_argument("--out",
                    default="benchmarks/results/synth_ct_control.json")
    args = ap.parse_args()

    means = [float(v) for v in args.dt_means.split(",") if v != ""]
    cvs = [float(v) for v in args.dt_cvs.split(",") if v != ""]
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    archs = [a for a in args.archs.split(",") if a]

    print(f"synthetic continuous-time control | Van der Pol mu={args.mu} | "
          f"{len(seeds)} seeds | Δt supplied to every arch | "
          f"target: x(t+{args.horizon}) from {args.n_samples} irregular "
          f"samples\n")

    results = {}
    for dt_mean in means:
        for dt_cv in cvs:
            rng = np.random.default_rng(777)          # same data for all archs
            Xtr, ytr = build_dataset(args, dt_mean, dt_cv, rng,
                                     args.train_windows)
            Xte, yte = build_dataset(args, dt_mean, dt_cv, rng,
                                     args.test_windows)
            mu_f = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
            sd_f = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-8
            Xtr = ((Xtr - mu_f) / sd_f).astype(np.float32)
            Xte = ((Xte - mu_f) / sd_f).astype(np.float32)
            ymu, ysd = float(ytr.mean()), float(ytr.std() + 1e-8)
            ytr_n = ((ytr - ymu) / ysd).astype(np.float32)
            yte_n = ((yte - ymu) / ysd).astype(np.float32)
            base = float(np.sqrt(((yte - ytr.mean()) ** 2).mean()))
            key = f"dt{dt_mean}_cv{dt_cv}"
            print(f"Δt mean={dt_mean} cv={dt_cv}  train {len(Xtr)} / test "
                  f"{len(Xte)} windows (mean-predictor RMSE {base:.4f})")
            results[key] = {"dt_mean": dt_mean, "dt_cv": dt_cv,
                            "baseline_rmse": base}
            for arch in archs:
                vals = [run(arch, Xtr, ytr_n, Xte, yte_n, args, s)
                        for s in seeds]
                vals = [v * ysd for v in vals if v is not None]
                if not vals:
                    print(f"    {arch:<12} UNSTABLE")
                    continue
                a = np.array(vals)
                results[key][arch] = {"rmse_mean": float(a.mean()),
                                      "rmse_std": float(a.std(ddof=1)),
                                      "n": len(a)}
                print(f"    {arch:<12} RMSE {a.mean():.4f} ± "
                      f"{a.std(ddof=1):.4f}")
            print()

    # pairwise Welch at every high-irregularity tier (cv=1 configs)
    print("=" * 66)
    pairwise = {}
    for key in [k for k in results if k.endswith("cv1.0")]:
        for a in archs:
            for b in archs:
                if a >= b or a not in results[key] or b not in results[key]:
                    continue
                ra, rb = results[key][a], results[key][b]
                se = math.sqrt(ra["rmse_std"] ** 2 / ra["n"]
                               + rb["rmse_std"] ** 2 / rb["n"])
                t = (rb["rmse_mean"] - ra["rmse_mean"]) / se if se else 0.0
                pairwise[f"{key}|{a}|{b}"] = round(t, 3)
                print(f"  {key:<16} {a} vs {b:<12} t={t:+.2f} "
                      f"{'SIGNIFICANT' if abs(t) > 2 else 'within noise'}")

    with open(args.out, "w") as f:
        json.dump({"system": "van_der_pol", "mu": args.mu, "seeds": len(seeds),
                   "horizon": args.horizon, "results": results,
                   "welch_pairwise_high_irregularity": pairwise,
                   "pre_registered_rule": "claim holds iff |t|>=2 vs gru_d "
                   "AND lstm AND gru at high-irregularity tiers on >=2 of 3 "
                   "domains"}, f, indent=2)
    print(f"\nresults -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
