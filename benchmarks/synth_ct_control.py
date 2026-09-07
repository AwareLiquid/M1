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

MECHANISM PROBE ADDITION (2026-08-29, registered BEFORE the mt_lnn_dt run).
Diagnosis of the 1/3 verdict: the production liquid layer decays by a
construction-time CONSTANT dt (mt_lnn/mt_lnn_layer.py:55), so Δt reached the
model only as a feature — the CT mechanism was never wired. The "mt_lnn_dt"
arch wires it: per-step lambda = exp(-dt_t / tau). Its OWN pre-registered
rules, evaluated on THIS domain:
  (R1 in-distribution) at >= 2 of the 3 cv=1 density tiers, mt_lnn_dt beats
      gru_d AND lstm AND gru AND mt_lnn with |t| >= 2;
  (R2 distribution shift) on the extrapolation tier (train dt_mean=0.05
      cv=1, test dt_mean=0.4 cv=1 — gaps the feature-learners never saw),
      mt_lnn_dt beats all four of them with |t| >= 2.
Verdict mapping, fixed now: R1+R2 -> the continuous-time mechanism claim is
REVIVED with evidence; R2 only -> the mechanism is a Δt-shift-robustness
advantage (narrower claim, recorded as partial); R1 only -> partial,
capacity-suspect; neither -> the mechanism story is CLOSED — record the null
and stop telling it. Params are reported alongside; the probe must stay in
the mt_lnn/gru parameter band (enforced by tests/test_liquid_dt.py) so a win
cannot be explained by size.

ADAPTIVE-DECAY PROBE ADDITION II (2026-08-29, registered BEFORE the
mt_lnn_ad run). mt_lnn_dt closed negative on both rules; its most
informative loss was the shift tier, where GRU-D's LEARNED decay beat the
fixed structural exp(-Δt/τ) (0.2485 vs 0.3005) — the same ordering the
literature reports for input-dependent vs fixed decay rates. The
"mt_lnn_ad" arch (LiquidADRegressor, battery_soh_edge.py) folds the
learnable decay INTO the liquid bank: λ_t = g·exp(-Δt_t/τ_{p,s}) +
(1-g)·MLP([per-proto x-summary; Δt_t]), g a per-(proto, scale) sigmoid
gate. Its OWN pre-registered rules, evaluated on THIS domain:
  (R2' distribution shift) on the extrapolation tier
      (extrap_dt0.05to0.4_cv1.0), mt_lnn_ad beats gru_d AND lstm AND gru
      AND mt_lnn_dt with |t| >= 2;
  (R1' in-distribution) at >= 2 of the 3 cv=1 density tiers, mt_lnn_ad
      beats gru_d AND lstm AND gru with |t| >= 2.
Verdict mapping, fixed now (all four cases, no post-hoc reading):
R2'+R1' -> the mechanism line REOPENS as "adaptive multi-scale decay" and
enters the full three-domain suite; R2' only -> the single
"Δt-shift-robustness" selling point stands, the line is archived NARROWED
(no reopening); R1' only -> partial, capacity-suspect, no reopening;
neither -> together with mt_lnn_dt's double negative the decay direction is
PERMANENTLY archived — no third probe may be raised. Params reported
alongside; the probe must stay in the mt_lnn/gru parameter band (enforced
by tests/test_liquid_ad.py).

    python benchmarks/synth_ct_control.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore", message=".*d_proto.*multiple of 8.*")

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
                      burn_in=5.0, dt_norm=None):
    """Walk one trajectory, drawing an independent sampling realisation per
    window. Returns (inputs (n, n_samples, 3), targets (n,)).

    dt_norm is the DENOMINATOR of the Δt channel. It defaults to dt_mean
    (in-distribution tiers). The extrapolation tier passes the TRAIN mean
    for both splits, so the test windows carry Δt values ~8x outside the
    training range in the model's own input units — normalising by the test
    mean would silently wash the shift out.
    """
    dn = dt_mean if dt_norm is None else dt_norm
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
        Xs.append(np.concatenate([obs, (dt_ch / dn)[:, None]],
                                 axis=1).astype(np.float32))
        ys.append(np.float32(
            np.interp(ts[-1] + horizon, times, states[:, 0])))
        t0 = ts[-1] + horizon + dt_mean                      # past the target
    return Xs, ys


def build_dataset(args, dt_mean, dt_cv, rng, n_windows, dt_norm=None):
    """Fresh trajectories + sampling realisations until n_windows exist."""
    per_batch = max(8, n_windows // 6)
    allX, ally = [], []
    while sum(map(len, allX)) < n_windows:
        t, st = simulate_batch(args.mu, args.t_end, args.dt_fine,
                               per_batch, rng)
        for i in range(per_batch):
            xs, ys = windows_from_traj(t, st[i], dt_mean, dt_cv,
                                       args.n_samples, args.horizon, rng,
                                       dt_norm=dt_norm)
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


def standardize_sensors(Xtr, Xte):
    """Z-score the SENSOR channels only — the trailing Δt channel stays in
    its raw protocol units (dt / train-mean ≥ 0). Z-scoring elapsed time
    would create negative gaps and is semantically wrong (found by smoke)."""
    mu_f = Xtr[..., :-1].reshape(-1, Xtr.shape[-1] - 1).mean(0)
    sd_f = Xtr[..., :-1].reshape(-1, Xtr.shape[-1] - 1).std(0) + 1e-8

    def z(X):
        out = X.copy()
        out[..., :-1] = ((X[..., :-1] - mu_f) / sd_f).astype(np.float32)
        return out.astype(np.float32)

    return z(Xtr), z(Xte)


def _code_version():
    """Short git hash stamped into the JSON for provenance."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL, timeout=5).decode().strip()
    except Exception:
        return "unknown"


def _train_all(arch, Xtr, ytr_n, Xte, yte_n, args, seeds, ysd):
    """Run every seed with per-seed stdout progress + JSON trace.

    Returns (list of rmse in raw units, per-seed dict, param count, total s).
    A seed that diverges is named in the log, not silently dropped.
    """
    vals, per_seed = [], {}
    t_all = time.time()
    for s in seeds:
        t0 = time.time()
        v = run(arch, Xtr, ytr_n, Xte, yte_n, args, s)
        took = time.time() - t0
        if v is None:
            print(f"    {arch:<12} seed {s}: UNSTABLE (non-finite loss) "
                  f"after {took:.0f}s")
            continue
        r = v * ysd
        vals.append(r)
        per_seed[s] = round(r, 6)
        print(f"    {arch:<12} seed {s}: rmse {r:.4f} ({took:.0f}s)")
    pm = build(arch, args.d_model, args.n_layers, Xtr.shape[1])
    pm.inp = nn.Linear(Xtr.shape[-1], args.d_model)
    n_params = sum(p.numel() for p in pm.parameters())
    return vals, per_seed, n_params, time.time() - t_all


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs",
                    default="mt_lnn,mt_lnn_dt,mt_lnn_ad,lstm,gru,gru_d,transformer")
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
    ap.add_argument("--no-extrap", action="store_true",
                    help="skip the Δt distribution-shift tier (R2)")
    ap.add_argument("--extrap-train-mean", type=float, default=0.05)
    ap.add_argument("--extrap-test-mean", type=float, default=0.4)
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
          f"samples")
    print("pre-registered rule: domain won iff mt_lnn |t|>=2 vs gru_d AND "
          "lstm AND gru at the cv=1 tiers; claim needs >=2 of 3 domains")
    print("probe rules registered in the docstring BEFORE any run: "
          "mt_lnn_dt R1/R2, mt_lnn_ad R2'/R1' — thresholds not movable\n")

    results = {}
    for dt_mean in means:
        for dt_cv in cvs:
            rng = np.random.default_rng(777)          # same data for all archs
            Xtr, ytr = build_dataset(args, dt_mean, dt_cv, rng,
                                     args.train_windows)
            Xte, yte = build_dataset(args, dt_mean, dt_cv, rng,
                                     args.test_windows)
            Xtr, Xte = standardize_sensors(Xtr, Xte)
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
                vals, per_seed, n_params, took = _train_all(
                    arch, Xtr, ytr_n, Xte, yte_n, args, seeds, ysd)
                if not vals:
                    print(f"    {arch:<12} ALL SEEDS UNSTABLE")
                    continue
                a = np.array(vals)
                results[key][arch] = {"rmse_mean": float(a.mean()),
                                      "rmse_std": float(a.std(ddof=1)),
                                      "n": len(a), "params": n_params,
                                      "per_seed": per_seed,
                                      "train_seconds": round(took, 1)}
                print(f"    {arch:<12} RMSE {a.mean():.4f} ± "
                      f"{a.std(ddof=1):.4f}  ({n_params:,} params, "
                      f"{took:.0f}s total)")
            print()

    # ---- extrapolation tier (R2): train narrow Δt, test wide Δt ----------
    if not args.no_extrap:
        rng = np.random.default_rng(777)
        Xtr, ytr = build_dataset(args, args.extrap_train_mean, 1.0, rng,
                                 args.train_windows,
                                 dt_norm=args.extrap_train_mean)
        Xte, yte = build_dataset(args, args.extrap_test_mean, 1.0, rng,
                                 args.test_windows,
                                 dt_norm=args.extrap_train_mean)
        Xtr, Xte = standardize_sensors(Xtr, Xte)
        ymu, ysd = float(ytr.mean()), float(ytr.std() + 1e-8)
        ytr_n = ((ytr - ymu) / ysd).astype(np.float32)
        yte_n = ((yte - ymu) / ysd).astype(np.float32)
        base = float(np.sqrt(((yte - ytr.mean()) ** 2).mean()))
        key = (f"extrap_dt{args.extrap_train_mean}to{args.extrap_test_mean}"
               f"_cv1.0")
        print(f"EXTRAP tier | train Δt mean={args.extrap_train_mean} cv=1, "
              f"test Δt mean={args.extrap_test_mean} cv=1, Δt channel in "
              f"TRAIN units (mean-predictor RMSE {base:.4f})")
        results[key] = {"extrap": True,
                        "train_dt_mean": args.extrap_train_mean,
                        "test_dt_mean": args.extrap_test_mean,
                        "baseline_rmse": base}
        for arch in archs:
            vals, per_seed, n_params, took = _train_all(
                arch, Xtr, ytr_n, Xte, yte_n, args, seeds, ysd)
            if not vals:
                print(f"    {arch:<12} ALL SEEDS UNSTABLE")
                continue
            a = np.array(vals)
            results[key][arch] = {"rmse_mean": float(a.mean()),
                                  "rmse_std": float(a.std(ddof=1)),
                                  "n": len(a), "params": n_params,
                                  "per_seed": per_seed,
                                  "train_seconds": round(took, 1)}
            print(f"    {arch:<12} RMSE {a.mean():.4f} ± {a.std(ddof=1):.4f}  "
                  f"({n_params:,} params, {took:.0f}s total)")
        print()

    # pairwise Welch at every high-irregularity tier (cv=1, non-extrap)
    print("=" * 66)
    pairwise = {}

    def t_of(key, a, b):
        """Signed Welch t for 'a has lower RMSE than b' from the pair dict."""
        k = f"{key}|{a}|{b}" if a < b else f"{key}|{b}|{a}"
        if k not in pairwise:
            return None
        return pairwise[k] if a < b else -pairwise[k]

    hi_keys = [k for k in results
               if results[k].get("dt_cv") == 1.0 and not results[k].get("extrap")]
    for key in hi_keys:
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

    extrap_key = next((k for k in results if results[k].get("extrap")), None)
    if extrap_key:
        for a in archs:
            for b in archs:
                if a >= b or a not in results[extrap_key] \
                        or b not in results[extrap_key]:
                    continue
                ra, rb = results[extrap_key][a], results[extrap_key][b]
                se = math.sqrt(ra["rmse_std"] ** 2 / ra["n"]
                               + rb["rmse_std"] ** 2 / rb["n"])
                t = (rb["rmse_mean"] - ra["rmse_mean"]) / se if se else 0.0
                pairwise[f"{extrap_key}|{a}|{b}"] = round(t, 3)
        print(f"  extrapolation tier {extrap_key}: mt_lnn_dt vs each — " +
              ", ".join(f"{b}={t_of(extrap_key, 'mt_lnn_dt', b)}"
                        for b in ("gru_d", "lstm", "gru", "mt_lnn")
                        if t_of(extrap_key, "mt_lnn_dt", b) is not None))
        print(f"  extrapolation tier {extrap_key}: mt_lnn_ad vs each — " +
              ", ".join(f"{b}={t_of(extrap_key, 'mt_lnn_ad', b)}"
                        for b in ("gru_d", "lstm", "gru", "mt_lnn_dt")
                        if t_of(extrap_key, "mt_lnn_ad", b) is not None))

    # ---- pre-registered mechanism verdict (R1/R2, mapping fixed above) ---
    opponents = [o for o in ("gru_d", "lstm", "gru", "mt_lnn") if o in archs]
    r1_wins = sum(
        1 for k in hi_keys
        if opponents and all((t_of(k, "mt_lnn_dt", o) or 0.0) >= 2.0
                             for o in opponents))
    r1 = bool(opponents) and "mt_lnn_dt" in archs and r1_wins >= 2
    r2 = False
    if extrap_key and "mt_lnn_dt" in archs and opponents:
        r2 = all((t_of(extrap_key, "mt_lnn_dt", o) or 0.0) >= 2.0
                 for o in opponents)
    if r1 and r2:
        mech = "REVIVED — CT mechanism claim holds with evidence (R1+R2)"
    elif r2:
        mech = "PARTIAL — Δt-shift-robustness only (R2); narrower claim"
    elif r1:
        mech = "PARTIAL — in-distribution only (R1); capacity-suspect"
    elif "mt_lnn_dt" in archs:
        mech = ("CLOSED — mechanism null on both rules; record the negative "
                "and stop telling the CT story")
    else:
        mech = "N/A — mt_lnn_dt not in --archs"
    print(f"\nMECHANISM VERDICT (pre-registered R1/R2): {mech}")

    # ---- pre-registered adaptive-decay verdict (R2'/R1', mt_lnn_ad) ------
    ad_r2_opp = [o for o in ("gru_d", "lstm", "gru", "mt_lnn_dt") if o in archs]
    ad_r1_opp = [o for o in ("gru_d", "lstm", "gru") if o in archs]
    r1p_wins = sum(
        1 for k in hi_keys
        if ad_r1_opp and all((t_of(k, "mt_lnn_ad", o) or 0.0) >= 2.0
                             for o in ad_r1_opp))
    r1p = bool(ad_r1_opp) and "mt_lnn_ad" in archs and r1p_wins >= 2
    r2p = False
    if extrap_key and "mt_lnn_ad" in archs and ad_r2_opp:
        r2p = all((t_of(extrap_key, "mt_lnn_ad", o) or 0.0) >= 2.0
                  for o in ad_r2_opp)
    if r1p and r2p:
        mech_ad = ("REOPENED — adaptive multi-scale decay holds (R2'+R1'); "
                   "mechanism line reopens and enters the three-domain suite")
    elif r2p:
        mech_ad = ("NARROW — Δt-shift-robustness only (R2'); single selling "
                   "point stands, line archived narrowed, no reopening")
    elif r1p:
        mech_ad = ("PARTIAL — in-distribution only (R1'); capacity-suspect, "
                   "no reopening, no new probe")
    elif "mt_lnn_ad" in archs:
        mech_ad = ("ARCHIVED FOR GOOD — double negative with mt_lnn_dt; the "
                   "decay direction is permanently closed, no third probe")
    else:
        mech_ad = "N/A — mt_lnn_ad not in --archs"
    print(f"MECHANISM VERDICT (pre-registered R2'/R1', mt_lnn_ad): {mech_ad}")

    with open(args.out, "w") as f:
        json.dump({"system": "van_der_pol", "mu": args.mu, "seeds": len(seeds),
                   "horizon": args.horizon, "results": results,
                   "config": {"d_model": args.d_model,
                              "n_layers": args.n_layers,
                              "epochs": args.epochs, "batch": args.batch,
                              "lr": args.lr,
                              "train_windows": args.train_windows,
                              "test_windows": args.test_windows,
                              "n_samples": args.n_samples,
                              "window_len": 49,   # t0 + n_samples gaps
                              "dev": args.dev, "data_seed": 777,
                              "seed_list": seeds,
                              "code_version": _code_version(),
                              "started": time.strftime("%Y-%m-%d %H:%M:%S")},
                   "welch_pairwise_high_irregularity": pairwise,
                   "mechanism_probe": {"r1_in_distribution": r1,
                                       "r1_wins": r1_wins,
                                       "r2_distribution_shift": r2,
                                       "verdict": mech},
                   "mechanism_probe_ad": {
                       "r2_prime_distribution_shift": r2p,
                       "r1_prime_in_distribution": r1p,
                       "r1_prime_wins": r1p_wins,
                       "verdict": mech_ad},
                   "pre_registered_rule": "claim holds iff |t|>=2 vs gru_d "
                   "AND lstm AND gru at high-irregularity tiers on >=2 of 3 "
                   "domains; mt_lnn_dt mechanism rules R1/R2 and mt_lnn_ad "
                   "adaptive-decay rules R2'/R1' as in docstring"},
                  f, indent=2)
    print(f"\nresults -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
