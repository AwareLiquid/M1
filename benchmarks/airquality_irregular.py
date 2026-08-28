"""Irregular-sampling robustness, domain 2 of 3: Beijing multi-site air
quality — real public data with NATURAL missingness (a different failure
process than the battery pilot's synthetic drops).

Data. UCI "Beijing Multi-Site Air-Quality Data" (public, no auth):
  https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip
12 monitoring stations, hourly, 2013-03-01..2017-02-28. Verification against
the published UCI page description, asserted by ``load_stations()``:
  * exactly 12 station files (page: "12 locations");
  * exactly 35,064 hourly rows per station (page: four years of hourly data:
    1461 days x 24 h, leap-year 2016 included);
  * pollutant columns contain real missing values (measured 2-6% per column,
    max outage run 343 h) — the page notes the data "contain missing values".

Task. Multi-step forecasting: given the past 48 observed hours of 11 numeric
channels (PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM;
categorical wind direction is dropped), predict PM2.5 and TEMP for the next
6 hours. ONE STATION is held out entirely (cross-station generalisation, the
same honesty rule as the battery pilot's held-out cell); remaining stations
train. Targets must be fully observed, inputs compact the same way as the
battery protocol: rows with any missing channel are DROPPED and the elapsed
hours become a Δt channel appended as the LAST input feature and supplied to
EVERY architecture (the fairness guardrail — the RNNs are not blind to gaps;
GRU-D additionally drives its trainable decay from that same Δt). A random
drop sweep on top of the natural missingness creates the sparse tiers.

Pre-registered judgement (fixed before any run, not movable after): the
robustness claim needs mt_lnn to beat gru_d AND lstm AND gru with Welch |t|
>= 2 at the sparsest tier on at least 2 of the 3 task domains (battery /
air quality / synthetic control). Anything less is recorded as a negative.

    python benchmarks/airquality_irregular.py --drops 0.0,0.3,0.6
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from benchmarks.battery_soh_edge import build

DATA_URL = ("https://archive.ics.uci.edu/static/public/501/"
            "beijing+multi+site+air+quality+data.zip")
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "beijing_air")
EXTRACTED = os.path.join(ROOT, "PRSA_Data_20130301-20170228")
STATIONS = ["Aotizhongxin", "Changping", "Dingling", "Dongsi", "Guanyuan",
            "Gucheng", "Huairou", "Nongzhanguan", "Shunyi", "Tiantan",
            "Wanliu", "Wanshouxigong"]
FEATURES = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES",
            "DEWP", "RAIN", "WSPM"]
TARGETS = ["PM2.5", "TEMP"]
N_STATIONS_EXPECTED = 12
N_ROWS_EXPECTED = 35064


# --------------------------------------------------------------------------
# data: download once, validate against the published description
# --------------------------------------------------------------------------

def ensure_data():
    """Download + extract if needed; return the station CSV directory."""
    if os.path.isdir(EXTRACTED) and len(os.listdir(EXTRACTED)) >= N_STATIONS_EXPECTED:
        return EXTRACTED
    os.makedirs(ROOT, exist_ok=True)
    zpath = os.path.join(ROOT, "prsa.zip")
    if not os.path.exists(zpath):
        print(f"downloading {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        inner = [n for n in z.namelist() if n.endswith(".zip")]
        for n in inner:
            z.extract(n, ROOT)
    with zipfile.ZipFile(os.path.join(ROOT, inner[0])) as z:
        z.extractall(ROOT)
    return EXTRACTED


def load_station(name):
    """One station -> (35,064 x 11 float array with NaN gaps, hour index).

    Validation: 12 files x 35,064 hourly rows must match the UCI page, and
    the pollutant columns must actually contain NaNs (the "missing values"
    the page warns about). Returns None rows as NaN.
    """
    path = os.path.join(EXTRACTED, f"PRSA_Data_{name}_20130301-20170228.csv")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} — run ensure_data() first")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != N_ROWS_EXPECTED:
        raise SystemExit(f"{name}: {len(rows)} rows != {N_ROWS_EXPECTED} "
                         f"published on the UCI page")
    def fv(r, k):
        v = r[k]
        return float("nan") if v in ("", "NA") else float(v)
    X = np.array([[fv(r, k) for k in FEATURES] for r in rows], dtype=np.float32)
    nan_frac = np.isnan(X[:, :6]).mean()
    if not (0.001 < nan_frac < 0.20):
        raise SystemExit(f"{name}: pollutant NaN fraction {nan_frac:.4f} "
                         f"outside the published 0-6%-per-column band")
    return X


# --------------------------------------------------------------------------
# windowing on the compacted stream (battery-protocol style)
# --------------------------------------------------------------------------

def make_windows(X, in_len, horizon, stride, min_in, rng, drop=0.0,
                 max_windows=None):
    """Sliding windows over the compacted stream.

    A window keeps only fully-observed input rows (natural missingness +
    optional extra random drop); Δt = elapsed hours since the previous KEPT
    row, appended as the last channel and normalised by the nominal window
    span. Targets require `horizon` consecutive observed rows after the last
    input row. Returns (inputs, targets) with targets (n, 2*horizon)
    [PM2.5 then TEMP, per step].
    """
    T = len(X)
    obs = ~np.isnan(X).any(axis=1)                    # fully-observed rows
    F1 = X.shape[1] + 1
    Xs, ys = [], []
    for s in range(0, T - in_len - horizon, stride):
        e = s + in_len
        if not obs[e:e + horizon].all():
            continue
        span = obs[s:e]
        idx = np.where(span)[0] + s
        if drop > 0:
            keep_n = max(min_in, int(round(len(idx) * (1.0 - drop))))
            if len(idx) > keep_n:
                idx = np.sort(rng.choice(idx, size=keep_n, replace=False))
        if len(idx) < min_in:
            continue
        dt = np.diff(idx, prepend=idx[0] - 1).astype(np.float32)
        # Left-pad to a FIXED length so every window has the same shape (the
        # regressors read out at the last step = last REAL sample). Pad rows
        # carry dt = 1.0 — the maximal normalised gap — so a decay-based
        # model treats them as "nothing happened", and zero features.
        w = np.zeros((in_len, F1), dtype=np.float32)
        w[:, -1] = 1.0
        w[-len(idx):, :-1] = X[idx]
        w[-len(idx):, -1] = dt / in_len
        Xs.append(w)
        tgt = X[e:e + horizon][:, [FEATURES.index(t) for t in TARGETS]]
        ys.append(tgt.reshape(-1))
    if not Xs:
        raise SystemExit("no usable windows")
    Xs, ys = np.stack(Xs), np.stack(ys)
    if max_windows and len(Xs) > max_windows:
        sel = rng.choice(len(Xs), size=max_windows, replace=False)
        Xs, ys = Xs[sel], ys[sel]
    return Xs.astype(np.float32), ys.astype(np.float32)


def run(arch, Xtr, ytr, Xte, yte, args, seed, n_feat, n_out):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = args.dev
    m = build(arch, args.d_model, args.n_layers, Xtr.shape[1]).to(dev)
    m.inp = nn.Linear(n_feat, args.d_model).to(dev)
    m.head = nn.Linear(args.d_model, n_out).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr)
    lossf = nn.MSELoss()
    Xtr_t = torch.from_numpy(Xtr).to(dev)
    ytr_t = torch.from_numpy(ytr).to(dev)
    Xte_t = torch.from_numpy(Xte).to(dev)
    m.train()
    n = len(Xtr_t)
    for _ in range(args.epochs):
        perm = torch.randperm(n, device=dev)
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
    return pred


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", default="mt_lnn,lstm,gru,gru_d,transformer")
    ap.add_argument("--test-station", default="Dingling")
    ap.add_argument("--in-len", type=int, default=48)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--stride", type=int, default=24)
    ap.add_argument("--min-in", type=int, default=16)
    ap.add_argument("--max-train-windows", type=int, default=3000)
    ap.add_argument("--max-test-windows", type=int, default=800)
    ap.add_argument("--drops", default="0.0,0.3,0.6")
    ap.add_argument("--d_model", type=int, default=78)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--dev", default=None)
    ap.add_argument("--out", default="benchmarks/results/airquality_irregular.json")
    args = ap.parse_args()

    if args.dev is None:
        args.dev = ("mps" if torch.backends.mps.is_available()
                    else "cuda" if torch.cuda.is_available() else "cpu")

    ensure_data()
    train_st = [s for s in STATIONS if s != args.test_station]
    print(f"stations: train={len(train_st)} test=[{args.test_station}] "
          f"(held out) | dev={args.dev}")
    Xs = [load_station(s) for s in train_st]
    Xte_raw = load_station(args.test_station)

    # standardise on TRAIN stations only (features AND targets)
    alltrain = np.concatenate(Xs)
    mu = np.nanmean(alltrain, axis=0)
    sd = np.nanstd(alltrain, axis=0) + 1e-8
    tcol = [FEATURES.index(t) for t in TARGETS]
    ymu, ysd = mu[tcol], sd[tcol]

    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    drops = [float(d) for d in args.drops.split(",") if d != ""]
    archs = [a for a in args.archs.split(",") if a]

    print(f"air-quality irregular sweep | held-out {args.test_station} | "
          f"{len(seeds)} seeds | Δt supplied to every arch | "
          f"task: {args.horizon}h {TARGETS} from {args.in_len}h history\n")

    H = args.horizon
    results = {}
    for drop in drops:
        rng = np.random.default_rng(12345)          # same pattern for all archs
        Xtr, ytr = [], []
        for X in Xs:
            x, y = make_windows((X - mu) / sd, args.in_len, H, args.stride,
                                args.min_in, rng, drop,
                                max_windows=args.max_train_windows // len(Xs) + 1)
            Xtr.append(x); ytr.append(y)
        Xtr = np.concatenate(Xtr)[:args.max_train_windows]
        ytr = np.concatenate(ytr)[:args.max_train_windows]
        Xte, yte = make_windows((Xte_raw - mu) / sd, args.in_len, H,
                                args.stride, args.min_in, rng, drop,
                                max_windows=args.max_test_windows)
        n_feat = Xtr.shape[-1]
        n_out = ytr.shape[-1]
        base = float(np.sqrt(((yte - 0.0) ** 2).mean()))
        print(f"drop={drop:.0%}  train {len(Xtr)} / test {len(Xte)} windows "
              f"(standardised-space baseline RMSE {base:.4f})")
        results[f"{drop}"] = {"train_windows": int(len(Xtr)),
                              "test_windows": int(len(Xte)),
                              "baseline_std_rmse": base}
        for arch in archs:
            vals = []
            for s in seeds:
                pred = run(arch, Xtr, ytr, Xte, yte, args, s, n_feat, n_out)
                if pred is None:
                    continue
                err = (pred - yte) * ysd.repeat(H)          # back to raw units
                # RMSE per target variable, averaged over horizon steps
                for ti, t in enumerate(TARGETS):
                    e = err[:, ti * H:(ti + 1) * H]
                    vals.append((t, float(np.sqrt((e ** 2).mean()))))
            if not vals:
                print(f"    {arch:<12} UNSTABLE")
                continue
            for t in TARGETS:
                a = np.array([v[1] for v in vals if v[0] == t])
                results[f"{drop}"].setdefault(arch, {})[t] = {
                    "rmse_mean": float(a.mean()),
                    "rmse_std": float(a.std(ddof=1)),
                    "n": int(len(a))}
                print(f"    {arch:<12} {t:<6} RMSE {a.mean():.3f} ± "
                      f"{a.std(ddof=1):.3f}")
        print()

    # pairwise Welch at the sparsest tier, per target
    dN = f"{drops[-1]}"
    pairwise = {}
    print("=" * 66)
    print(f"pairwise Welch t at drop={drops[-1]:.0%} "
          f"(positive = first arch lower RMSE)")
    for t in TARGETS:
        for a in archs:
            for b in archs:
                if a >= b or a not in results[dN] or b not in results[dN]:
                    continue
                ra, rb = results[dN][a][t], results[dN][b][t]
                se = math.sqrt(ra["rmse_std"] ** 2 / ra["n"]
                               + rb["rmse_std"] ** 2 / rb["n"])
                tv = (rb["rmse_mean"] - ra["rmse_mean"]) / se if se else 0.0
                pairwise[f"{t}|{a}|{b}"] = round(tv, 3)
                print(f"  {t:<6} {a} vs {b:<12} t={tv:+.2f} "
                      f"{'SIGNIFICANT' if abs(tv) > 2 else 'within noise'}")

    with open(args.out, "w") as f:
        json.dump({"test_station": args.test_station, "seeds": len(seeds),
                   "drops": drops, "targets": TARGETS, "results": results,
                   "welch_pairwise_sparsest": pairwise,
                   "pre_registered_rule": "claim holds iff |t|>=2 vs gru_d "
                   "AND lstm AND gru at the sparsest tier on >=2 of 3 domains"},
                  f, indent=2)
    print(f"\nresults -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
