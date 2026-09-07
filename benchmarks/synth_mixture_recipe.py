"""Δt-mixture training — REGISTERED recipe experiment (self-contained).

Context. The architecture mechanism line is closed at three levels (Δt
feature wiring; decay rate fixed→learnable, probes mt_lnn_dt/mt_lnn_ad;
attractor ablation — see HANDOFF §2.9 and synth_ct_control_ad.json). The
exploratory pilot (dt_mixture_pilot.json) then showed that training on a
MIXTURE of gap scales repairs most of the Δt-shift degradation for every
architecture and erases the architecture ranking (mt_lnn 0.452→0.057,
gru_d 0.251→0.071, all t≥4.8). This experiment upgrades that pilot to a
pre-registered, full-scale, self-contained two-arm comparison. It is a
TRAINING-RECIPE claim, not an architecture claim: no model code differs
between arms, only the training-data Δt distribution.

Design. Arms (identical total window count, identical eval sets):
  * narrow  — train on dt_mean=0.05 cv=1 only (the archived extrap arm);
  * mixture — train on an equal mixture of dt_mean {0.05, 0.15, 0.4} cv=1;
  Δt channel in NARROW units (÷0.05) for both arms; sensor/target
  standardisation frozen to the narrow arm's training statistics; the
  narrow arm is trimmed to exactly the mixture arm's size (the producer's
  build_dataset can return fewer windows than requested at sparse tiers —
  recorded, affects both arch-agnostic).
Eval tiers (fresh rng 999, identical for both arms):
  E1 in-dist   dt 0.05 cv=1;
  E2 shift     dt 0.4 cv=1  — the archived shift tier, scale INSIDE the
              mixture ("trained-scale interpolation");
  E3 true-OOD  dt 1.6 cv=1  — 4× beyond the mixture's largest scale.
              DESCRIPTIVE ONLY, no claim attaches to E3: it measures
              whether mixture training merely interpolates (and the
              TIDES-style extrapolation mechanism, arXiv:2605.09742,
              remains the open architecture gap) or extrapolates.

Pre-registered claims (fixed in this docstring BEFORE the run; the
runtime verdict below applies them exactly; thresholds not movable):
  R1 (repair): at E2, for EVERY arch, the mixture arm beats the narrow
      arm with Welch t >= 2 AND relative RMSE reduction >= 50%.
  R2 (bounded cost): at E1, for at least 6 of the 7 archs, mixture-arm
      RMSE <= narrow-arm RMSE + 0.03 absolute.
Verdict mapping, fixed now: R1+R2 -> the recipe claim HOLDS ("mixture Δt
  training is the default recipe for deployments with unknown sampling
  regimes" — recorded as a training-recipe result, NOT an architecture
  claim); R1 only -> PARTIAL (repair holds, per-arch in-dist cost must be
  reported alongside); R1 fails -> NEGATIVE (recipe does not transfer to
  full scale; the TIDES-style architecture line goes on the agenda).

    python benchmarks/synth_mixture_recipe.py \
        --out benchmarks/results/synth_mixture_recipe.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from benchmarks import synth_ct_control as scc
from benchmarks.battery_soh_edge import build


class _A:
    def __init__(self):
        self.mu, self.dt_fine, self.t_end = 2.0, 0.002, 120.0
        self.n_samples, self.horizon = 48, 0.5


def _code_version():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL, timeout=5).decode().strip()
    except Exception:
        return "unknown"


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
            if not torch.isfinite(loss):
                return None
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
    """t for 'a has lower RMSE than b' from per-seed lists."""
    a, b = np.asarray(a), np.asarray(b)
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((b.mean() - a.mean()) / se) if se else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs",
                    default="mt_lnn,mt_lnn_dt,mt_lnn_ad,lstm,gru,gru_d,"
                            "transformer")
    ap.add_argument("--mixture", default="0.05,0.15,0.4")
    ap.add_argument("--narrow-dt", type=float, default=0.05)
    ap.add_argument("--ood-dt", type=float, default=1.6)
    ap.add_argument("--train-windows", type=int, default=2500)
    ap.add_argument("--eval-windows", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--d_model", type=int, default=78)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--dev", default="cpu")
    ap.add_argument("--out",
                    default="benchmarks/results/synth_mixture_recipe.json")
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    archs = [a for a in args.archs.split(",") if a]
    mixes = [float(v) for v in args.mixture.split(",")]
    a = _A()

    # ---- data ----------------------------------------------------------
    print(f"Δt-mixture REGISTERED recipe experiment | arms narrow/"
          f"mixture{mixes} | eval E1 0.05 / E2 0.4 / E3 OOD {args.ood_dt} "
          f"| {len(archs)} archs x {len(seeds)} seeds | dev {args.dev}")
    print("pre-registered: R1 every-arch E2 t>=2 & >=50% reduction; "
          "R2 E1 cost <= +0.03 for >=6/7 archs; E3 descriptive only\n")
    t0 = time.time()
    rng = np.random.default_rng(777)
    Xn, yn = scc.build_dataset(a, args.narrow_dt, 1.0, rng,
                               args.train_windows, dt_norm=args.narrow_dt)
    parts = []
    n3 = args.train_windows // len(mixes)
    rngm = np.random.default_rng(778)
    for dm in mixes:
        Xp, yp = scc.build_dataset(a, dm, 1.0, rngm, n3,
                                   dt_norm=args.narrow_dt)
        parts.append((Xp, yp))
    Xm = np.concatenate([p[0] for p in parts])
    ym = np.concatenate([p[1] for p in parts])
    n_eq = min(len(Xn), len(Xm))                 # exact arm parity
    Xn, yn, Xm, ym = Xn[:n_eq], yn[:n_eq], Xm[:n_eq], ym[:n_eq]

    evals = {}
    for tag, dm in (("E1_dt0.05", 0.05), ("E2_dt0.40", 0.40),
                    (f"E3_ood_dt{args.ood_dt}", args.ood_dt)):
        rng_e = np.random.default_rng(999)
        Xe, ye = scc.build_dataset(a, dm, 1.0, rng_e, args.eval_windows,
                                   dt_norm=args.narrow_dt)
        evals[tag] = (Xe, ye)

    mu_f = Xn[..., :-1].reshape(-1, Xn.shape[-1] - 1).mean(0)
    sd_f = Xn[..., :-1].reshape(-1, Xn.shape[-1] - 1).std(0) + 1e-8

    def prep(X):
        out = X.copy()
        out[..., :-1] = ((X[..., :-1] - mu_f) / sd_f).astype(np.float32)
        return out.astype(np.float32)

    ymu, ysd = float(yn.mean()), float(yn.std() + 1e-8)
    arms = {"narrow": (prep(Xn), ((yn - ymu) / ysd).astype(np.float32)),
            "mixture": (prep(Xm), ((ym - ymu) / ysd).astype(np.float32))}
    ev = {k: (prep(X), y) for k, (X, y) in evals.items()}
    print(f"data built in {time.time() - t0:.0f}s | arm size {n_eq} "
          f"(mixture parts requested {n3}x{len(mixes)}) | eval sizes "
          + ", ".join(f"{k.split('_')[0]}:{len(v[0])}"
                      for k, v in ev.items()) + "\n")

    # ---- train ---------------------------------------------------------
    res = {}
    for arch in archs:
        res[arch] = {}
        for arm, (Xtr, ytr_n) in arms.items():
            per = {k: [] for k in ev}
            for s in seeds:
                t1 = time.time()
                m = train_model(arch, Xtr, ytr_n, args.d_model,
                                args.n_layers, args.epochs, args.lr,
                                args.batch, s)
                if m is None:
                    print(f"  {arch:12s} {arm:7s} seed {s}: UNSTABLE")
                    continue
                for k, (X, y) in ev.items():
                    per[k].append(rmse_raw(m, X, y, ymu, ysd))
                print(f"  {arch:12s} {arm:7s} seed {s}: "
                      + " | ".join(f"{k.split('_')[0]} {per[k][-1]:.4f}"
                                   for k in ev)
                      + f" ({time.time() - t1:.0f}s)")
            res[arch][arm] = {k: {"per_seed": [round(v, 6) for v in vs],
                                  "rmse_mean": round(float(np.mean(vs)), 6)
                                  if vs else None,
                                  "rmse_std": round(float(np.std(
                                      vs, ddof=1)), 6) if len(vs) > 1 else 0.0,
                                  "n": len(vs)}
                              for k, vs in per.items()}
        npar_m = build(arch, args.d_model, args.n_layers, Xtr.shape[1])
        npar_m.inp = nn.Linear(Xtr.shape[-1], args.d_model)
        res[arch]["params"] = sum(p.numel() for p in npar_m.parameters())
        print()

    # ---- pre-registered verdict ----------------------------------------
    r1_arch_ok, r2_ok_archs, e2_detail = [], [], {}
    for arch in archs:
        nar, mix = res[arch]["narrow"], res[arch]["mixture"]
        if nar["E2_dt0.40"]["rmse_mean"] is None \
                or mix["E2_dt0.40"]["rmse_mean"] is None:
            e2_detail[arch] = "missing"
            continue
        t_e2 = welch(mix["E2_dt0.40"]["per_seed"], nar["E2_dt0.40"]["per_seed"])
        red = 1.0 - mix["E2_dt0.40"]["rmse_mean"] / nar["E2_dt0.40"]["rmse_mean"]
        ok1 = bool(t_e2 >= 2.0 and red >= 0.5)
        r1_arch_ok.append(ok1)
        e2_detail[arch] = {"t": round(t_e2, 3), "reduction_pct":
                           round(red * 100, 1), "ok": ok1}
        d1 = mix["E1_dt0.05"]["rmse_mean"] - nar["E1_dt0.05"]["rmse_mean"]
        r2_ok_archs.append(bool(d1 <= 0.03))
    r1 = bool(r1_arch_ok) and all(r1_arch_ok)
    r2 = sum(r2_ok_archs) >= 6
    if r1 and r2:
        verdict = ("RECIPE HOLDS — mixture Δt training repairs the "
                   "trained-scale shift for every arch at bounded "
                   "in-dist cost; record as a TRAINING-RECIPE result, "
                   "not an architecture claim")
    elif r1:
        verdict = ("RECIPE PARTIAL — repair holds for every arch but "
                   "in-dist cost exceeds +0.03 for more than one arch; "
                   "report per-arch costs")
    else:
        verdict = ("RECIPE NEGATIVE — repair does not hold at full scale "
                   "for every arch; TIDES-style architecture line goes "
                   "on the agenda")
    print("=" * 66)
    for arch, d in e2_detail.items():
        if isinstance(d, dict):
            print(f"  E2 {arch:12s} t={d['t']:+.2f} reduction "
                  f"{d['reduction_pct']:.0f}% -> {'OK' if d['ok'] else 'FAIL'}")
    print(f"R1 (every-arch repair) = {r1}; R2 (bounded cost, "
          f"{sum(r2_ok_archs)}/7) = {r2}")
    print(f"E3 true-OOD (dt={args.ood_dt}, descriptive only):")
    for arch in archs:
        mix, nar = res[arch]["mixture"], res[arch]["narrow"]
        print(f"  {arch:12s} narrow {nar[f'E3_ood_dt{args.ood_dt}']['rmse_mean']:.4f}"
              f" -> mixture {mix[f'E3_ood_dt{args.ood_dt}']['rmse_mean']:.4f}")
    print(f"\nRECIPE VERDICT (pre-registered R1/R2): {verdict}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"system": "van_der_pol_mixture_recipe", "mu": 2.0,
                   "seeds": len(seeds),
                   "arms": {"narrow": f"dt{args.narrow_dt} cv1 n={n_eq}",
                            "mixture": f"{mixes} cv1 n={n_eq}"},
                   "results": res, "e2_detail": e2_detail,
                   "recipe_verdict": {"r1_repair": r1, "r2_bounded_cost": r2,
                                      "r2_ok_archs": int(sum(r2_ok_archs)),
                                      "verdict": verdict},
                   "config": {"epochs": args.epochs, "batch": args.batch,
                              "lr": args.lr, "d_model": args.d_model,
                              "n_layers": args.n_layers, "dev": args.dev,
                              "seed_list": seeds, "archs": archs,
                              "ood_dt": args.ood_dt,
                              "train_windows": args.train_windows,
                              "eval_windows": args.eval_windows,
                              "code_version": _code_version(),
                              "started": time.strftime(
                                  "%Y-%m-%d %H:%M:%S")},
                   "pre_registered_rule": "R1: every arch E2 t>=2 & >=50% "
                   "reduction; R2: E1 cost <= +0.03 for >=6/7 archs; E3 "
                   "descriptive; mapping as in docstring"}, f, indent=2)
    print(f"\nresults -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
