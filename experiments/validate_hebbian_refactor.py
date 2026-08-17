"""
experiments/validate_hebbian_refactor.py — Stage 2 end-to-end validation of the
refactored Hebbian branch (mechanism A) on a ~10M model, real WikiText-103.

GOAL (per the agreed Stage 2 spec): "把等价的赫布损失加入总 loss，实现梯度对齐、
5% 占比上限约束；在 10M 小模型上跑通训练，验证无 NaN、梯度稳定、门控生效。"
This is a STABILITY + MECHANISM check, not an effectiveness claim (that is Stage 3
ablation). It verifies:
  - training runs to completion with NaN_count == 0 (graceful + finite);
  - the dedicated LAVI gate is ACTIVE the whole run (gate_std > 0, not stuck);
  - the Hebbian gradient share stays <= grad_frac_cap (the hard 5% safety bound)
    at every recalibration -- measured, not assumed;
  - final val PPL vs an identical use_hebbian_refactor=OFF baseline (reported as
    informational; a real verdict needs the Stage 3 ablation).

GRADIENT ALIGNMENT (why two extra backward passes on recal steps):
  g_main is measured with the Hebbian term zeroed (alpha_eff=0 -> loss==pure LM);
  g_hebb_raw is measured by back-propagating raw_loss=-signal alone. recalibrate()
  then sets _scale so base_lr*_scale*g_hebb_raw <= cap*g_main. On normal steps
  only ONE backward runs (the real, capped step), so the amortised cost is small.

Run:  python -m experiments.validate_hebbian_refactor
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel
from train import BinDataset, evaluate

DATA_DIR = os.path.join(_ROOT, "kaggle_out", "tiny", "data")


def _grad_global_norm(model: MTLNNModel) -> float:
    tot = 0.0
    for p in model.parameters():
        if p.grad is not None:
            tot += float(p.grad.detach().pow(2).sum().item())
    return math.sqrt(tot)


def build_model(use_refactor: bool, vocab_size: int, *, d_model: int, n_layers: int,
                n_heads: int, seq_len: int, device: str, seed: int) -> MTLNNModel:
    torch.manual_seed(seed)
    cfg = MTLNNConfig(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        n_kv_heads=1, d_head=d_model // n_heads, max_seq_len=seq_len,
        gwtb_n_heads=1, dropout=0.0, attention_dropout=0.0,
        use_competitive_gwtb=False, use_world_model=False, use_predictive_coding=False,
        use_rhythm=False,                 # prove the gate lives WITHOUT rhythm
        use_hebbian=False,                # legacy path off
        use_hebbian_refactor=use_refactor,
    )
    return MTLNNModel(cfg).to(device)


def run(use_refactor: bool, *, steps: int, batch: int, seq_len: int, lr: float,
        d_model: int, n_layers: int, n_heads: int, recal_every: int,
        eval_batches: int, device: str, seed: int) -> Dict:
    np.random.seed(seed)
    vocab = json.load(open(os.path.join(DATA_DIR, "meta.json")))["vocab_size"]
    train_ds = BinDataset(os.path.join(DATA_DIR, "train.bin"), seq_len)
    val_ds = BinDataset(os.path.join(DATA_DIR, "validation.bin"), seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False,
                            drop_last=True, num_workers=0)

    model = build_model(use_refactor, vocab, d_model=d_model, n_layers=n_layers,
                        n_heads=n_heads, seq_len=seq_len, device=device, seed=seed)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            eps=1e-8, weight_decay=0.01)
    hp = model.hebbian_plasticity

    nan_count = 0
    gate_stds: List[float] = []
    grad_fracs: List[float] = []
    scales: List[float] = []
    last_grad_norm = 0.0

    model.train()
    step = 0
    t0 = time.time()
    while step < steps:
        for inp, lbl in train_loader:
            if step >= steps:
                break
            inp, lbl = inp.to(device), lbl.to(device)

            # ---- periodic gradient alignment (refactor only) ----
            if hp is not None and step % recal_every == 0:
                saved = hp._scale
                # (1) pure-LM gradient: zero the Hebbian term (alpha_eff=0).
                hp._scale = 0.0
                opt.zero_grad(set_to_none=True)
                out = model(inp, labels=lbl)
                out["loss"].backward()
                g_main = _grad_global_norm(model)
                # (2) raw Hebbian gradient: fresh forward, back-prop -signal only.
                opt.zero_grad(set_to_none=True)
                _ = model(inp, labels=lbl)
                raw = hp.raw_loss(model)
                g_hebb_raw = _grad_global_norm(model) if raw is None else None
                if raw is not None:
                    raw.backward()
                    g_hebb_raw = _grad_global_norm(model)
                opt.zero_grad(set_to_none=True)
                hp._scale = saved
                hp.recalibrate(g_main, g_hebb_raw)
                grad_fracs.append(hp.last_stats["grad_frac"])
                scales.append(hp._scale)

            # ---- real (capped) optimisation step ----
            opt.zero_grad(set_to_none=True)
            out = model(inp, labels=lbl)
            loss = out["loss"]
            if not torch.isfinite(loss):
                nan_count += 1
                opt.zero_grad(set_to_none=True)
                step += 1
                continue
            loss.backward()
            last_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
            opt.step()
            if hp is not None and hp.last_stats.get("gate_std") is not None:
                gate_stds.append(hp.last_stats["gate_std"])
            step += 1

    final_ppl = evaluate(model, val_loader, device, max_batches=eval_batches)
    return {
        "use_refactor": use_refactor,
        "n_params_M": round(n_params / 1e6, 2),
        "final_ppl": round(final_ppl, 4),
        "nan_count": nan_count,
        "final_grad_norm": round(last_grad_norm, 5),
        "gate_std_mean": round(float(np.mean(gate_stds)), 6) if gate_stds else None,
        "gate_std_min": round(float(np.min(gate_stds)), 6) if gate_stds else None,
        "grad_frac_max": round(float(np.max(grad_fracs)), 6) if grad_fracs else None,
        "grad_frac_mean": round(float(np.mean(grad_fracs)), 6) if grad_fracs else None,
        "scale_mean": round(float(np.mean(scales)), 4) if scales else None,
        "seconds": round(time.time() - t0, 1),
    }


def main(*, steps: int = 120, batch: int = 8, seq_len: int = 96, lr: float = 3e-4,
         d_model: int = 128, n_layers: int = 4, n_heads: int = 8,
         recal_every: int = 15, eval_batches: int = 40, seed: int = 0) -> Dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert os.path.exists(os.path.join(DATA_DIR, "train.bin")), f"missing {DATA_DIR}"
    common = dict(steps=steps, batch=batch, seq_len=seq_len, lr=lr, d_model=d_model,
                  n_layers=n_layers, n_heads=n_heads, recal_every=recal_every,
                  eval_batches=eval_batches, device=device, seed=seed)
    print("[off]  running baseline (use_hebbian_refactor=False) ...", flush=True)
    off = run(False, **common)
    print(f"[off]  ppl={off['final_ppl']:.3f} nan={off['nan_count']} "
          f"({off['n_params_M']}M, {off['seconds']:.0f}s)", flush=True)
    print("[on]   running refactor (use_hebbian_refactor=True) ...", flush=True)
    on = run(True, **common)
    print(f"[on]   ppl={on['final_ppl']:.3f} nan={on['nan_count']} "
          f"gate_std_min={on['gate_std_min']} grad_frac_max={on['grad_frac_max']} "
          f"scale_mean={on['scale_mean']} ({on['seconds']:.0f}s)", flush=True)

    cap = MTLNNConfig().hebbian_grad_frac_cap
    checks = {
        "no_nan": on["nan_count"] == 0 and off["nan_count"] == 0,
        "grad_finite": math.isfinite(on["final_grad_norm"]),
        "gate_active": (on["gate_std_min"] or 0.0) > 1e-4,
        "grad_frac_within_cap": (on["grad_frac_max"] or 0.0) <= cap + 1e-6,
    }
    report = {
        "config": dict(common, cap=cap),
        "off": off, "on": on,
        "delta_ppl_on_minus_off": round(on["final_ppl"] - off["final_ppl"], 4),
        "stage2_checks": checks,
        "stage2_pass": all(checks.values()),
    }
    _save(report)
    print("\n=== STAGE 2 VALIDATION ===")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  dPPL (on-off) = {report['delta_ppl_on_minus_off']:+.4f} "
          f"(informational; effectiveness verdict is Stage 3)")
    print(f"  STAGE 2 {'PASS' if report['stage2_pass'] else 'FAIL'}")
    return report


def _save(report: Dict) -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "report_validate_hebbian_refactor.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
