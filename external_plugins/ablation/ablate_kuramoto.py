"""ablate_kuramoto.py — REAL ablation of the Kuramoto plugin on a trained O1
MTLNNModel over held-out tokens.

Run (defaults to the m2_out_v7 173M checkpoint + its own validation.bin):
    python -m external_plugins.ablation.ablate_kuramoto
    python -m external_plugins.ablation.ablate_kuramoto \
        --ckpt kaggle_out/tiny/checkpoints/final.pt --n-seq 16 --seq-len 128

What it does
------------
1. Rebuilds MTLNNConfig from the checkpoint's saved config dict, loads the
   trained weights into a from-scratch MTLNNModel (O1), eval mode.
2. Computes BASELINE held-out perplexity (model's pure next-token CE).
3. OBSERVE: attaches Kuramoto (observe) + early-exit; recomputes PPL and asserts
   it is bit-identical to baseline (the non-intrusion contract on the REAL model)
   while collecting the order parameter R per layer and the settling-block index.
4. INTERVENE SWEEP: for a small grid of (kappa, alpha, n_steps), re-measures PPL
   and reports ΔPPL — the actual "does phase coupling help / hurt" signal.

All forwards run under torch.no_grad(); weights are never modified.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import warnings

import numpy as np
import torch

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
try:  # Windows consoles default to GBK; force UTF-8 so output never crashes.
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from mt_lnn.config import MTLNNConfig          # noqa: E402
from mt_lnn.model import MTLNNModel            # noqa: E402
from external_plugins import (                 # noqa: E402
    PluginRunner,
    KuramotoCoupling,
    SettlingEarlyExit,
)


def build_model(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_cfg = ck["config"]
    # keep only the init fields MTLNNConfig accepts (drop derived d_proto, etc.)
    init_fields = {f.name for f in dataclasses.fields(MTLNNConfig) if f.init}
    cfg_kwargs = {k: v for k, v in raw_cfg.items() if k in init_fields}
    cfg = MTLNNConfig(**cfg_kwargs)
    model = MTLNNModel(cfg).eval()
    missing, unexpected = model.load_state_dict(ck["model_state"], strict=False)
    if missing:
        print(f"[load] {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"[load] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    print(f"[load] {ckpt_path}  step={ck.get('step')}  "
          f"d_model={cfg.d_model} n_layers={cfg.n_layers} P={cfg.n_protofilaments}")
    return model, cfg


def load_batch(data_path: str, n_seq: int, seq_len: int, vocab_size: int, seed: int = 0):
    arr = np.memmap(data_path, dtype=np.uint16, mode="r")
    if len(arr) < n_seq * seq_len:
        raise ValueError(f"{data_path} has {len(arr)} tokens, need {n_seq * seq_len}")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(arr) - seq_len, size=n_seq)
    xs = np.stack([arr[s : s + seq_len].astype(np.int64) for s in starts])
    ids = torch.from_numpy(xs).clamp_(0, vocab_size - 1)
    return ids  # (n_seq, seq_len)


def random_R_baseline(P: int, n_draws: int = 20000, seed: int = 0) -> float:
    """Expected order parameter R for P i.i.d. uniform phases (Monte-Carlo).

    With P channels whose phases carry NO cross-channel structure, R is not 0 but
    a finite-size floor ~ sqrt(pi)/(2*sqrt(P)) in expectation. Any observed R must
    clear this floor to count as real binding rather than sampling noise.
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-math.pi, math.pi, size=(n_draws, P))
    z = np.exp(1j * theta).mean(axis=1)
    return float(np.abs(z).mean())


@torch.no_grad()
def perplexity(model, ids, chunk: int = 8) -> float:
    """Mean next-token CE over the batch → exp = PPL. Chunked to bound memory.

    Uses the model's standard training convention: pass labels == input_ids, and
    the model's internal logits[:, :-1] vs labels[:, 1:] shift yields the proper
    next-token cross-entropy (result['lm_loss'] is the pure CE, pre-aux).
    """
    tot_loss, tot_n = 0.0, 0
    for i in range(0, ids.shape[0], chunk):
        b_ids = ids[i : i + chunk]
        out = model(b_ids, labels=b_ids)
        n = b_ids.shape[0] * (b_ids.shape[1] - 1)
        tot_loss += float(out["lm_loss"]) * n
        tot_n += n
    return math.exp(tot_loss / max(1, tot_n))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="m2_out_v7/checkpoints/final.pt")
    ap.add_argument("--data", default="m2_out_v7/data/validation.bin")
    ap.add_argument("--n-seq", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model, cfg = build_model(args.ckpt)
    ids = load_batch(args.data, args.n_seq, args.seq_len, cfg.vocab_size, args.seed)
    print(f"[data] {args.data}  batch={tuple(ids.shape)}")

    base_ppl = perplexity(model, ids)
    print(f"\n[baseline] held-out PPL = {base_ppl:.4f}")

    # ---- OBSERVE: must be bit-identical; collect R + settling ------------- #
    kur = KuramotoCoupling(n_protofilaments=cfg.n_protofilaments, kappa=0.6,
                           n_steps=4, mode="observe")
    ee = SettlingEarlyExit(tol=0.05, relative=True)
    with PluginRunner(model, plugins=[kur, ee]):
        obs_ppl = perplexity(model, ids)
        diag = {"kuramoto": dict(kur.last_diagnostics),
                "early_exit": dict(ee.last_diagnostics)}
    identical = abs(obs_ppl - base_ppl) < 1e-9
    print(f"[observe ] PPL = {obs_ppl:.4f}   bit-identical={identical}")

    # De-anisotropy control: re-measure R after removing the cross-channel mean.
    kur_c = KuramotoCoupling(n_protofilaments=cfg.n_protofilaments, kappa=0.6,
                             n_steps=4, mode="observe", center=True)
    with PluginRunner(model, plugins=[kur_c]):
        perplexity(model, ids)
    R_raw = diag["kuramoto"].get("R_mean")
    R_centered = kur_c.last_diagnostics.get("R_mean")
    R_rand = random_R_baseline(cfg.n_protofilaments)
    print(f"[observe ] R_mean raw        = {R_raw:.4f}")
    print(f"[observe ] R_mean centered   = {R_centered:.4f}  (de-anisotropy)")
    print(f"[observe ] R random baseline = {R_rand:.4f}  (P={cfg.n_protofilaments} uniform phases)")
    verdict = ("REAL cross-channel structure" if R_centered > R_rand + 0.05
               else "likely ANISOTROPY artifact")
    print(f"[observe ] => {verdict}")
    R_layers = {k: round(v, 4) for k, v in diag["kuramoto"].items() if k.startswith("R_final/")}
    print(f"[observe ] R per layer (raw) = {R_layers}")
    print(f"[observe ] settle_block = {diag['early_exit'].get('settle_block')}"
          f"/{diag['early_exit'].get('n_blocks')}  "
          f"skippable_frac = {diag['early_exit'].get('skippable_frac')}")

    # ---- INTERVENE SWEEP -------------------------------------------------- #
    print("\n[intervene sweep]  (dPPL < 0 => phase coupling HELPS)")
    print(f"  {'kappa':>6} {'alpha':>6} {'steps':>6} {'lin':>4} "
          f"{'PPL':>10} {'dPPL':>10} {'d%':>8}")
    grid = [
        (0.6, 0.02, 4, True),
        (0.6, 0.05, 4, True),
        (0.6, 0.10, 4, True),
        (1.0, 0.05, 4, True),
        (0.6, 0.05, 4, False),
    ]
    for kappa, alpha, n_steps, lin in grid:
        kp = KuramotoCoupling(n_protofilaments=cfg.n_protofilaments, kappa=kappa,
                              n_steps=n_steps, dt=0.5, linear=lin,
                              mode="intervene", alpha=alpha)
        with PluginRunner(model, plugins=[kp]):
            ppl = perplexity(model, ids)
        d = ppl - base_ppl
        print(f"  {kappa:6.2f} {alpha:6.2f} {n_steps:6d} {str(lin):>4} "
              f"{ppl:10.4f} {d:+10.4f} {100*d/base_ppl:+7.2f}%")

    print("\n==================== CONTRACT ====================")
    ok = identical
    print("  observe bit-identical to baseline on REAL model:", ok)
    print("  (intervene rows above are the real ablation signal)")
    print("==================================================")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
