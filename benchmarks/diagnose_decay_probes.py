"""Post-mortem diagnostics for the two CLOSED decay probes (mt_lnn_dt /
mt_lnn_ad) — why they lost the Δt-shift tier, and a cheap causal gate for
the one hypothesis the data still leaves open.

Findings this script reproduces (2026-08-29, after the ARCHIVED FOR GOOD
verdict in synth_ct_control_ad.json):

  1. ANATOMY of the trained hybrid probe (mt_lnn_ad): the per-(proto,
     scale) gates stay at their init (~0.5) after 12 epochs — the two
     decay paths never got differentiated by training; the MLP path IS
     live (λ_mlp correlates with Δt at ~−0.5) but rate-adaptivity alone
     bought nothing on the shift tier (ad ≡ dt, t=+0.00).
  2. WHAT GRU-D DOES INSTEAD: its learned per-unit rate γ(Δt_norm=8)≈0.25
     sends ~75% of the hidden state to the RUNNING MEAN hbar — a learned,
     data-dependent ANCHOR. Both liquid probes decay toward the current
     input coupling A_t (the bank's hard-wired fixed point); the anchor
     variable was never tested.
  3. CAUSAL GATE (the decision step for a possible anchor probe): train
     GRU-D with hbar ABLATED (hbar ≡ 0, rate and everything else intact).
     If shift robustness collapses, decay-toward-anchor is load-bearing
     and an "anchored bank" probe is justified as a NEW pre-registered
     family (attractor variable — the decay-RATE closure stands
     regardless). If robustness survives, the anchor is not the
     differentiator and the direction dies here, cheaply.

CPU-only, minutes. Not an adjudication: no threshold here can revive or
close any registered claim by itself; the gate only decides whether the
next pre-registered probe is worth building.

    python benchmarks/diagnose_decay_probes.py \
        --out benchmarks/results/diagnose_decay_probes.json
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
from benchmarks.battery_soh_edge import GRUDRegressor, build


class _Args:
    """Namespace carrying the producer's data defaults."""

    def __init__(self, n_windows):
        self.mu = 2.0
        self.dt_fine = 0.002
        self.t_end = 120.0
        self.n_samples = 48
        self.horizon = 0.5
        self.n_windows = n_windows


class GRUDNoAnchor(GRUDRegressor):
    """GRU-D with the running-mean anchor ablated: hbar ≡ 0.

    Identical learned per-unit decay RATE, gates, input decay — the state
    now decays toward the ORIGIN instead of a running mean. Single-variable
    causal test of the attractor."""

    def forward(self, x, dt=None, mask=None):
        if dt is None:
            dt = x[..., -1:]
        if mask is None:
            mask = torch.ones_like(dt)
        e = self.inp(torch.cat(
            [self._input_decay(x[..., :-1], dt, mask), dt], dim=-1))
        dec_in = torch.cat([dt, 1.0 - mask], dim=-1)
        B, T, D = e.shape
        seq = e
        for cell in self.cells:
            gamma = cell.decay_rate_seq(dec_in)
            x_proj = seq @ cell.Wx
            h = seq.new_zeros(B, D)
            out = []
            for t in range(T):
                h_dec = gamma[:, t] * h                  # hbar ablated: →0
                gates = x_proj[:, t] + h_dec @ cell.Uh + cell.bias
                z, r, _ = gates.chunk(3, dim=-1)
                z, r = torch.sigmoid(z), torch.sigmoid(r)
                n = torch.tanh(x_proj[:, t, 2 * D:]
                               + (r * h_dec) @ cell.Uh[:, 2 * D:]
                               + cell.bias[2 * D:])
                h = (1.0 - z) * h_dec + z * n
                out.append(h)
            seq = torch.stack(out, dim=1)
        return self.head(self.norm(seq[:, -1])).squeeze(-1)


def build_data(a, train_windows, test_windows):
    """Extrap-tier data exactly as the producer builds it (rng 777,
    train 0.05/cv1, test 0.4/cv1, Δt channel in TRAIN units) + a fresh
    in-distribution eval set."""
    rng = np.random.default_rng(777)
    Xtr, ytr = scc.build_dataset(a, 0.05, 1.0, rng, train_windows,
                                 dt_norm=0.05)
    Xte, yte = scc.build_dataset(a, 0.4, 1.0, rng, test_windows,
                                 dt_norm=0.05)
    Xtr, Xte = scc.standardize_sensors(Xtr, Xte)
    Xid, yid = scc.build_dataset(_Args(test_windows), 0.05, 1.0,
                                 np.random.default_rng(778), test_windows,
                                 dt_norm=0.05)
    _, Xid = scc.standardize_sensors(Xtr, Xid)
    return (Xtr, ytr, Xte, yte, Xid, yid)


def train(arch, Xtr, ytr, d_model, n_layers, epochs, lr, batch, seed,
          cls=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if cls is None:
        m = build(arch, d_model, n_layers, Xtr.shape[1])
    else:
        m = cls(d_model, n_layers, Xtr.shape[1])
    m.inp = nn.Linear(Xtr.shape[-1], d_model)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    Xt = torch.from_numpy(Xtr)
    yt = torch.from_numpy(((ytr - ytr.mean()) / (ytr.std() + 1e-8))
                          .astype(np.float32))
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


def rmse(m, X, y, ymu, ysd):
    m.eval()
    with torch.no_grad():
        p = m(torch.from_numpy(X)).numpy()
    return float(np.sqrt((((p - (y - ymu) / ysd) * ysd) ** 2).mean()))


def anatomy(ad, Xte, d_model):
    """Gate / two-path-λ / gamma readings of trained models."""
    lines = []
    with torch.no_grad():
        for li in range(2):
            g = torch.sigmoid(getattr(ad, f"gate_{li}"))
            lines.append(f"layer{li} gate g: mean {g.mean():.3f} "
                         f"[{g.min():.3f}, {g.max():.3f}] (init 0.5)")
        x = torch.from_numpy(Xte[:64])
        seq = ad.inp(x)
        dt = x[..., -1].clamp(min=0)
        li = 0
        tau = nn.functional.softplus(getattr(ad, f"log_tau_{li}")) + ad.tau_min
        ls = torch.exp(-dt[:, :, None, None] / tau[None, None])
        summ = getattr(ad, f"lam_proj_{li}")(seq)
        feats = torch.cat([summ, dt[..., None]], dim=-1)
        o = getattr(ad, f"mlp_lam_{li}")(feats)
        lm = torch.exp(-nn.functional.softplus(o)).reshape(
            *dt.shape, ad.P, ad.S)
        g = torch.sigmoid(getattr(ad, f"gate_{li}"))
        le = g[None, None] * ls + (1 - g[None, None]) * lm
        corr = float(np.corrcoef(lm.mean(dim=(2, 3)).ravel(),
                                 dt.ravel())[0, 1])
        lines.append(f"shift inputs (Δt_norm≈8): λ_struct {ls.mean():.3f} | "
                     f"λ_mlp {lm.mean():.3f} (corr with Δt {corr:+.2f}, "
                     f"std {float(lm.std()):.3f}) | λ_eff {le.mean():.3f}")
    return lines


def gamma_table(gd):
    lines, W, b = [], gd.cells[0].W_decay, gd.cells[0].b_decay
    lines.append(f"gru_d W_decay(dt): mean {float(W[0].mean()):.3f} "
                 f"max {float(W[0].max()):.3f} (init U(0, 0.5))")
    for dtn in (1.0, 8.0):
        gam = torch.exp(-torch.relu(dtn * W[0] + b))
        lines.append(f"  γ(Δt_norm={dtn:.0f}) = {float(gam.mean()):.3f} "
                     f"(→ {1 - float(gam.mean()):.0%} toward hbar)")
    return lines


def welch(a, b):
    se = math.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
    return (np.mean(b) - np.mean(a)) / se if se else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-windows", type=int, default=1200)
    ap.add_argument("--test-windows", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--dev", default="cpu")
    ap.add_argument("--out",
                    default="benchmarks/results/diagnose_decay_probes.json")
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    a = _Args(args.train_windows)
    Xtr, ytr, Xte, yte, Xid, yid = build_data(a, args.train_windows,
                                              args.test_windows)
    ymu, ysd = float(ytr.mean()), float(ytr.std() + 1e-8)
    print(f"extrap-tier replication | train {len(Xtr)} / shift-test "
          f"{len(Xte)} / in-dist-eval {len(Xid)} windows | "
          f"{args.epochs} epochs | dev {args.dev}")

    # ---- part 1: anatomy (single seed is enough — readings, not a claim)
    t0 = time.time()
    ad = train("mt_lnn_ad", Xtr, ytr, 78, 2, args.epochs, 3e-3, 32, 0)
    gd0 = train("gru_d", Xtr, ytr, 78, 2, args.epochs, 3e-3, 32, 0)
    print(f"anatomy trained in {time.time() - t0:.0f}s")
    anat = {"gate_and_lambda": anatomy(ad, Xte, 78),
            "gru_d_gamma": gamma_table(gd0)}
    for k, ls in anat.items():
        print(f"[{k}]")
        for l in ls:
            print("  " + l)

    # ---- part 2: causal ablation gate (all seeds) ------------------------
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    res = {}
    for name, cls in (("gru_d", None), ("gru_d_noanchor", GRUDNoAnchor)):
        vals_id, vals_sh = [], []
        for s in seeds:
            t0 = time.time()
            m = train(name, Xtr, ytr, 78, 2, args.epochs, 3e-3, 32, s,
                      cls=cls)
            ri = rmse(m, Xid, yid, ymu, ysd)
            rs = rmse(m, Xte, yte, ymu, ysd)
            vals_id.append(ri)
            vals_sh.append(rs)
            print(f"  {name:16s} seed {s}: in-dist {ri:.4f} | "
                  f"shift {rs:.4f} ({time.time() - t0:.0f}s)")
        res[name] = {"in_dist": [round(v, 4) for v in vals_id],
                     "shift": [round(v, 4) for v in vals_sh],
                     "in_dist_mean": round(float(np.mean(vals_id)), 4),
                     "shift_mean": round(float(np.mean(vals_sh)), 4)}
    t = welch(res["gru_d_noanchor"]["shift"], res["gru_d"]["shift"])
    # negative t = ablated model has HIGHER shift rmse → anchor load-bearing
    gate = "ANCHOR LOAD-BEARING — ablation significantly worsens the shift " \
           "tier (|t|>=2); the anchored-bank probe is justified" \
           if t <= -2.0 else \
           "ANCHOR NOT THE DIFFERENTIATOR — ablation within noise; the " \
           "attractor hypothesis dies here, do not build the probe"
    print(f"\nWelch t (noanchor vs full gru_d, shift tier) = {t:+.2f}")
    print(f"GATE: {gate}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"purpose": "post-mortem of the closed decay probes; "
                              "causal gate for the anchor hypothesis",
                   "config": {"train_windows": args.train_windows,
                              "test_windows": args.test_windows,
                              "epochs": args.epochs, "seeds": seeds,
                              "dev": args.dev, "data_seed": 777},
                   "anatomy": anat,
                   "ablation": res,
                   "welch_t_shift_noanchor_vs_full": round(t, 3),
                   "gate": gate}, f, indent=2)
    print(f"\nresults -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
