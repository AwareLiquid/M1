"""A2 edge pilot: battery State-of-Health regression on real NASA cell data.

Why this task. The architecture's surviving claim is constant-memory streaming
inference, which only matters if it is useful on something a customer actually
deploys. Battery management is the cleanest such case: a BMS runs continuously
on a microcontroller, sees an unbounded stream of voltage/current/temperature,
and must estimate capacity fade — exactly the regime where an O(1) recurrent
state beats a growing window.

Data. NASA Ames PCoE Li-ion aging set (public, no auth):
  https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip
Cells B0005/6/7/18, 18650s cycled to failure at room temperature. Verified
against the published description: B0005 has 168 discharge cycles and fades
1.8565 -> 1.3251 Ah.

Protocol. Each discharge cycle is one sample: a (T, 3) series of
(voltage, current, temperature) -> the cycle's measured capacity in Ah.
We hold out an ENTIRE CELL for test rather than splitting cycles, because
splitting cycles inside one cell leaks the degradation trajectory and makes
every model look good. Cross-cell generalisation is the honest version and is
what a deployed BMS faces on a battery it has never seen.

Models share width (d_model) and depth (n_layers) so the comparison is about
the mixer rather than capacity. Parameter counts still differ by architecture
and are reported alongside every result — read them, do not assume parity:
  * mt_lnn   — the liquid core (MTLNNLayer), recurrent, O(1) state
  * lstm     — standard recurrent baseline
  * gru      — lighter recurrent baseline
  * transformer — attention encoder (O(T) KV during streaming)

Reported: test RMSE/MAE in Ah on the held-out cell, plus parameter count. This
script makes no claim about which wins — it prints what the run produces.

    py -3.11 benchmarks/battery_soh_edge.py --epochs 60
    py -3.11 benchmarks/battery_soh_edge.py --test-cell B0018 --archs mt_lnn,lstm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "nasa_battery", "extracted")
CELLS = ["B0005", "B0006", "B0007", "B0018"]
FEATURES = ["Voltage_measured", "Current_measured", "Temperature_measured"]


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load_cell(name, seq_len):
    """One sample per discharge cycle: (seq_len, 3) resampled series -> capacity."""
    import scipy.io as sio
    path = os.path.join(DATA_DIR, f"{name}.mat")
    if not os.path.exists(path):
        raise SystemExit(
            f"missing {path}\nDownload + extract the NASA set first:\n"
            f"  curl -L -o battery.zip 'https://phm-datasets.s3.amazonaws.com/"
            f"NASA/5.+Battery+Data+Set.zip'")
    mat = sio.loadmat(path, simplify_cells=True)[name]
    X, y = [], []
    for c in mat["cycle"]:
        if c["type"] != "discharge":
            continue
        d = c["data"]
        cap = np.ravel(d["Capacity"])
        if cap.size == 0 or not np.isfinite(cap[0]):
            continue
        cols = []
        for f in FEATURES:
            v = np.ravel(d[f]).astype(np.float64)
            if v.size < 2:
                cols = []
                break
            # Cycles differ in length (197 points is typical but not fixed);
            # resample each onto a common grid so a batch is well-defined. This
            # is a uniform re-grid, NOT interpolation of missing data.
            idx = np.linspace(0, v.size - 1, seq_len)
            cols.append(np.interp(idx, np.arange(v.size), v))
        if not cols:
            continue
        X.append(np.stack(cols, axis=-1))          # (seq_len, 3)
        y.append(float(cap[0]))
    if not X:
        raise SystemExit(f"{name}: no usable discharge cycles parsed")
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


# --------------------------------------------------------------------------
# models — all take (B, T, 3) and regress a scalar
# --------------------------------------------------------------------------

class LiquidRegressor(nn.Module):
    """MT-LNN liquid core as a sequence encoder + linear head."""

    def __init__(self, d_model=64, n_layers=2, n_feat=3):
        super().__init__()
        from mt_lnn.config import MTLNNConfig
        from mt_lnn.mt_lnn_layer import MTLNNLayer
        # d_model must divide by n_protofilaments for d_proto to be integral.
        cfg = MTLNNConfig(vocab_size=2, d_model=d_model, n_layers=n_layers,
                          n_heads=13, d_head=max(1, d_model // 13),
                          n_protofilaments=13, n_time_scales=5,
                          max_seq_len=4096, dropout=0.0, attention_dropout=0.0)
        self.inp = nn.Linear(n_feat, d_model)
        self.layers = nn.ModuleList([MTLNNLayer(cfg) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.inp(x)
        for layer in self.layers:
            out = layer(h)
            h = h + (out[0] if isinstance(out, tuple) else out)
        return self.head(self.norm(h)[:, -1]).squeeze(-1)


class RNNRegressor(nn.Module):
    def __init__(self, kind="lstm", d_model=64, n_layers=2, n_feat=3):
        super().__init__()
        rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.inp = nn.Linear(n_feat, d_model)
        self.rnn = rnn(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        h, _ = self.rnn(self.inp(x))
        return self.head(self.norm(h)[:, -1]).squeeze(-1)


class TransformerRegressor(nn.Module):
    def __init__(self, d_model=64, n_layers=2, n_feat=3, n_heads=None,
                 max_len=128):
        super().__init__()
        # d_model is chosen to divide by 13 for the liquid core (e.g. 65), which
        # is not divisible by a default 4 heads. Pick the largest divisor <= 8 so
        # the baseline still gets real multi-head attention at any d_model.
        if n_heads is None:
            n_heads = max(h for h in range(1, 9) if d_model % h == 0)
        self.n_heads = n_heads
        self.inp = nn.Linear(n_feat, d_model)
        # Size the positional table to the sequence we actually feed. A fixed
        # 4096-row table would add ~266K unused parameters at d_model=65 and
        # hand the attention baseline an 8x parameter advantage over the
        # recurrent models, which would make the comparison meaningless.
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        enc = nn.TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward=4 * d_model,
                                         batch_first=True, dropout=0.0,
                                         norm_first=True)
        self.enc = nn.TransformerEncoder(enc, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.inp(x)
        h = h + self.pos[:, :h.shape[1]]
        h = self.enc(h)
        return self.head(self.norm(h)[:, -1]).squeeze(-1)


def build(arch, d_model, n_layers, seq_len=128):
    if arch == "mt_lnn":
        return LiquidRegressor(d_model, n_layers)
    if arch in ("lstm", "gru"):
        return RNNRegressor(arch, d_model, n_layers)
    if arch == "transformer":
        return TransformerRegressor(d_model, n_layers, max_len=seq_len)
    raise SystemExit(f"unknown arch {arch}")


# --------------------------------------------------------------------------

def run(arch, Xtr, ytr, Xte, yte, args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = build(arch, args.d_model, args.n_layers, args.seq_len).to(dev)
    n_params = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr)
    lossf = nn.MSELoss()

    Xtr_t = torch.from_numpy(Xtr).to(dev)
    ytr_t = torch.from_numpy(ytr).to(dev)
    Xte_t = torch.from_numpy(Xte).to(dev)

    t0 = time.time()
    m.train()
    n = len(Xtr_t)
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad(set_to_none=True)
            loss = lossf(m(Xtr_t[idx]), ytr_t[idx])
            if not torch.isfinite(loss):
                return {"arch": arch, "seed": seed, "params": n_params,
                        "stable": False, "rmse": float("inf"), "mae": float("inf")}
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
    train_s = time.time() - t0

    m.eval()
    with torch.no_grad():
        pred = m(Xte_t).cpu().numpy()
    err = pred - yte
    return {"arch": arch, "seed": seed, "params": n_params, "stable": True,
            "rmse": float(np.sqrt((err ** 2).mean())),
            "mae": float(np.abs(err).mean()),
            "train_s": round(train_s, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", default="mt_lnn,lstm,gru,transformer")
    ap.add_argument("--test-cell", default="B0018",
                    help="cell held out entirely for test")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--d_model", type=int, default=65)   # 13*5, divides by 13
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="benchmarks/battery_soh_results.json")
    args = ap.parse_args()

    if args.test_cell not in CELLS:
        raise SystemExit(f"--test-cell must be one of {CELLS}")
    train_cells = [c for c in CELLS if c != args.test_cell]

    print(f"loading NASA cells: train={train_cells} test=[{args.test_cell}]")
    Xs, ys = [], []
    for c in train_cells:
        x, y = load_cell(c, args.seq_len)
        print(f"  {c}: {len(x)} discharge cycles, capacity "
              f"{y.max():.3f} -> {y.min():.3f} Ah")
        Xs.append(x); ys.append(y)
    Xtr, ytr = np.concatenate(Xs), np.concatenate(ys)
    Xte, yte = load_cell(args.test_cell, args.seq_len)
    print(f"  {args.test_cell}: {len(Xte)} cycles (HELD OUT), capacity "
          f"{yte.max():.3f} -> {yte.min():.3f} Ah")

    # Standardise features on TRAIN statistics only (no test leakage).
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-8
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    print(f"train samples {len(Xtr)}, test samples {len(Xte)}, "
          f"seq_len {args.seq_len}\n")

    # A constant predictor is the floor any model must beat; without it an
    # RMSE number is unreadable.
    base_rmse = float(np.sqrt(((yte - ytr.mean()) ** 2).mean()))
    print(f"baseline (predict train mean {ytr.mean():.3f} Ah): "
          f"RMSE {base_rmse:.4f} Ah\n")

    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    rows = []
    for arch in [a for a in args.archs.split(",") if a]:
        for sd_ in seeds:
            r = run(arch, Xtr, ytr, Xte, yte, args, sd_)
            rows.append(r)
            print(f"  [{arch:12s} seed {sd_}] params {r['params']:>7,} | "
                  f"RMSE {r['rmse']:.4f} | MAE {r['mae']:.4f} | "
                  f"{r.get('train_s','-')}s")

    print("\n" + "=" * 72)
    print(f"NASA battery SoH | held-out cell {args.test_cell} | "
          f"{len(seeds)} seeds | capacity in Ah")
    print(f"{'arch':<14}{'params':>9}  {'RMSE (mean±std)':>20}  {'MAE':>8}")
    summary = {}
    for arch in [a for a in args.archs.split(",") if a]:
        rs = [r for r in rows if r["arch"] == arch and r["stable"]]
        if not rs:
            print(f"{arch:<14}{'-':>9}  {'UNSTABLE':>20}")
            continue
        rm = np.array([r["rmse"] for r in rs])
        ma = np.array([r["mae"] for r in rs])
        summary[arch] = {"params": rs[0]["params"], "rmse_mean": float(rm.mean()),
                         "rmse_std": float(rm.std(ddof=1) if len(rm) > 1 else 0.0),
                         "mae_mean": float(ma.mean()), "n": len(rs)}
        print(f"{arch:<14}{rs[0]['params']:>9,}  "
              f"{rm.mean():>10.4f} ± {rm.std(ddof=1) if len(rm)>1 else 0:<7.4f}  "
              f"{ma.mean():>8.4f}")
    print(f"{'(mean baseline)':<14}{'-':>9}  {base_rmse:>10.4f}")
    print("=" * 72)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"test_cell": args.test_cell, "train_cells": train_cells,
                   "seq_len": args.seq_len, "d_model": args.d_model,
                   "n_layers": args.n_layers, "epochs": args.epochs,
                   "baseline_rmse": base_rmse, "summary": summary,
                   "runs": rows}, f, indent=2)
    print(f"\nresults -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
