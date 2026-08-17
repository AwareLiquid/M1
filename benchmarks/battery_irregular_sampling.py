"""A2 pilot, part 3: irregular sampling — the one axis where a liquid ODE
should structurally beat a discrete RNN.

Motivation. On the regularly-sampled battery task all four architectures are
statistically indistinguishable (see battery_soh_results_10seed.json: every
pairwise Welch |t| < 1 at n=10). Constant streaming memory is real but is a
property of any recurrent model, not of the liquid core specifically. So if the
liquid formulation has a deployable advantage, it has to show up where its
continuous-time assumption actually bites: sensor streams that are NOT sampled
on a fixed grid.

That is the realistic BMS regime — a controller wakes on events, drops samples
under load, and changes its rate with duty cycle. A discrete RNN sees only "next
step" and has no notion of how much time passed; a continuous-time model can in
principle integrate over the true gap.

Design, kept deliberately fair:
  * Irregularity is applied to TRAIN and TEST alike (a model is not asked to
    generalise to a regime it never saw).
  * Every architecture receives the elapsed time Δt as an EXTRA INPUT FEATURE.
    Without this the comparison would be rigged: the RNNs would be blind to the
    gaps by construction and would lose for a trivial reason rather than an
    architectural one.
  * We sweep the drop rate so the trend is visible, not a single point.
  * Same held-out-cell protocol and seed count as the regular-sampling run.

If the liquid core has no edge here either, that is the finding and it should be
reported as such.

    py -3.11 benchmarks/battery_irregular_sampling.py --drops 0.0,0.3,0.6
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

from benchmarks.battery_soh_edge import CELLS, build, load_cell


def irregular_subsample(X, drop, rng, min_keep=16):
    """Randomly drop timesteps and return (kept series + Δt channel).

    Each sample gets its own drop pattern — a fleet of controllers does not
    share a schedule. Δt is in units of original timesteps, normalised.
    """
    B, T, F = X.shape
    keep_n = max(min_keep, int(round(T * (1.0 - drop))))
    out = np.zeros((B, keep_n, F + 1), dtype=np.float32)
    for b in range(B):
        idx = np.sort(rng.choice(T, size=keep_n, replace=False))
        out[b, :, :F] = X[b, idx]
        dt = np.diff(idx, prepend=idx[0] - 1).astype(np.float32)
        out[b, :, F] = dt / T          # normalised elapsed time
    return out


def run(arch, Xtr, ytr, Xte, yte, args, seed, n_feat):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # n_feat is 4 here (3 sensors + Δt); build() defaults to 3, so patch the
    # input projection to match rather than silently truncating a channel.
    m = build(arch, args.d_model, args.n_layers, Xtr.shape[1]).to(dev)
    m.inp = nn.Linear(n_feat, args.d_model).to(dev)
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
    err = pred - yte
    return float(np.sqrt((err ** 2).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", default="mt_lnn,lstm,gru,transformer")
    ap.add_argument("--test-cell", default="B0018")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--drops", default="0.0,0.3,0.6",
                    help="fraction of timesteps dropped")
    ap.add_argument("--d_model", type=int, default=65)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--out", default="benchmarks/battery_irregular_results.json")
    args = ap.parse_args()

    train_cells = [c for c in CELLS if c != args.test_cell]
    Xs, ys = [], []
    for c in train_cells:
        x, y = load_cell(c, args.seq_len)
        Xs.append(x); ys.append(y)
    Xtr_raw, ytr = np.concatenate(Xs), np.concatenate(ys)
    Xte_raw, yte = load_cell(args.test_cell, args.seq_len)

    mu = Xtr_raw.reshape(-1, Xtr_raw.shape[-1]).mean(0)
    sd = Xtr_raw.reshape(-1, Xtr_raw.shape[-1]).std(0) + 1e-8
    Xtr_raw = (Xtr_raw - mu) / sd
    Xte_raw = (Xte_raw - mu) / sd

    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    drops = [float(d) for d in args.drops.split(",") if d != ""]
    archs = [a for a in args.archs.split(",") if a]

    print(f"irregular-sampling sweep | held-out {args.test_cell} | "
          f"{len(seeds)} seeds | Δt supplied to every arch\n")
    results = {}
    for drop in drops:
        rng = np.random.default_rng(12345)          # same pattern for all archs
        Xtr = irregular_subsample(Xtr_raw, drop, rng)
        Xte = irregular_subsample(Xte_raw, drop, rng)
        n_feat = Xtr.shape[-1]
        base = float(np.sqrt(((yte - ytr.mean()) ** 2).mean()))
        print(f"drop={drop:.0%}  kept {Xtr.shape[1]}/{args.seq_len} steps  "
              f"(baseline RMSE {base:.4f})")
        results[f"{drop}"] = {"kept_steps": int(Xtr.shape[1]), "baseline": base}
        for arch in archs:
            vals = [run(arch, Xtr, ytr, Xte, yte, args, s, n_feat) for s in seeds]
            vals = [v for v in vals if v is not None]
            if not vals:
                print(f"    {arch:<12} UNSTABLE")
                continue
            a = np.array(vals)
            results[f"{drop}"][arch] = {"rmse_mean": float(a.mean()),
                                        "rmse_std": float(a.std(ddof=1)),
                                        "n": len(a)}
            print(f"    {arch:<12} RMSE {a.mean():.4f} ± {a.std(ddof=1):.4f}")
        print()

    # Does the liquid core degrade more slowly than the discrete RNNs as the
    # sampling gets sparser? That, not the absolute number, is the claim.
    print("=" * 66)
    print("degradation from regular to sparsest sampling (lower = more robust)")
    d0, dN = f"{drops[0]}", f"{drops[-1]}"
    for arch in archs:
        if arch in results[d0] and arch in results[dN]:
            a0 = results[d0][arch]["rmse_mean"]
            aN = results[dN][arch]["rmse_mean"]
            print(f"  {arch:<12} {a0:.4f} -> {aN:.4f}   Δ {aN - a0:+.4f} "
                  f"({(aN/a0 - 1) * 100:+.1f}%)")

    if "mt_lnn" in results[dN]:
        mt = results[dN]["mt_lnn"]
        print(f"\nat drop={drops[-1]:.0%}, mt_lnn vs each (Welch t):")
        for arch in archs:
            if arch == "mt_lnn" or arch not in results[dN]:
                continue
            o = results[dN][arch]
            se = math.sqrt(mt["rmse_std"] ** 2 / mt["n"] + o["rmse_std"] ** 2 / o["n"])
            t = (o["rmse_mean"] - mt["rmse_mean"]) / se if se else 0.0
            print(f"  vs {arch:<12} t={t:+.2f}  "
                  f"{'SIGNIFICANT' if abs(t) > 2 else 'within noise'}")
    print("=" * 66)

    with open(args.out, "w") as f:
        json.dump({"test_cell": args.test_cell, "seeds": len(seeds),
                   "drops": drops, "results": results}, f, indent=2)
    print(f"\nresults -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
