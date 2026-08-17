"""
experiments/ablation_hebbian_refactor.py — Stage 3 step 1: find the MAXIMUM SAFE
& EFFECTIVE Hebbian gradient share for the refactored branch (mechanism A).

FRAMING (per the agreed Stage 3 goal): 5% (grad_frac_cap) is a hard SAFETY RED
LINE, never a target. We sweep the decoupled base_lr -- which, through
recalibrate(), maps to a REALIZED Hebbian gradient share = min(base_lr * raw_ratio,
cap) -- and look for the largest realized share that is BOTH stable (no NaN, PPL
not blown up) AND effective (helps, or at least does not hurt, the metrics).
Stage 2 showed that at base_lr=1e-2 the share is ~3e-6 (utterly inert), so this
sweep pushes base_lr into the tens--hundreds to make the term actually contribute.

METRICS per setting (vs the use_hebbian_refactor=OFF baseline):
  - realized grad_frac (mean/max)  -- where on the 0..5% axis we actually landed
  - stability: NaN count, final grad norm finite, val PPL not diverged
  - standard held-out val PPL (cross-run; carries seed noise, reported with std)
  - LONG-CONTEXT segmented PPL: early-third vs late-third token PPL within the
    SAME long eval sequence. This is a PAIRED, lower-noise probe of within-context
    consistency -- the within-sequence lagged Hebbian term, if it does anything,
    should most plausibly help the LATE segment. We report late-PPL and the
    late-minus-early gap (smaller gap => better long-range consistency).

HONEST STANCE: at 10M params / ~150 CPU steps the cross-run PPL noise floor is
large (prior ablation saw +/-7 PPL). If a Hebbian effect is smaller than that, the
correct conclusion is "no detectable benefit at this scale", which is a valid
Stage 3 result. The paired late-vs-early gap is the more sensitive signal.

Run:  python -m experiments.ablation_hebbian_refactor
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel
from train import BinDataset, evaluate

DATA_DIR = os.path.join(_ROOT, "kaggle_out", "tiny", "data")

# base_lr sweep. None == use_hebbian_refactor OFF (baseline). The values push the
# realized gradient share from ~0 up toward (but never past) the 5% red line.
SETTINGS = [("off", None), ("blr30", 30.0), ("blr100", 100.0),
            ("blr300", 300.0), ("blr1000", 1000.0)]


def _grad_global_norm(model: MTLNNModel) -> float:
    tot = 0.0
    for p in model.parameters():
        if p.grad is not None:
            tot += float(p.grad.detach().pow(2).sum().item())
    return math.sqrt(tot)


def build_model(base_lr: Optional[float], vocab: int, *, d_model: int, n_layers: int,
                n_heads: int, seq_len: int, device: str, seed: int) -> MTLNNModel:
    torch.manual_seed(seed)
    cfg = MTLNNConfig(
        vocab_size=vocab, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        n_kv_heads=1, d_head=d_model // n_heads, max_seq_len=seq_len,
        gwtb_n_heads=1, dropout=0.0, attention_dropout=0.0,
        use_competitive_gwtb=False, use_world_model=False, use_predictive_coding=False,
        use_rhythm=False, use_hebbian=False,
        use_hebbian_refactor=(base_lr is not None),
        hebbian_base_lr=(base_lr if base_lr is not None else 1e-2),
        hebbian_grad_frac_cap=0.05,   # the hard red line, fixed for every setting
    )
    return MTLNNModel(cfg).to(device)


@torch.no_grad()
def segmented_ppl(model, loader, device, *, max_batches: int, n_seg: int = 3) -> Dict:
    """Per-position-bucket next-token PPL within long eval sequences.
    Returns early (first bucket) and late (last bucket) PPL + their gap."""
    model.eval()
    seg_loss = [0.0] * n_seg
    seg_cnt = [0] * n_seg
    for i, (inp, lbl) in enumerate(loader):
        if i >= max_batches:
            break
        inp, lbl = inp.to(device), lbl.to(device)
        logits = model(inp, labels=lbl)["logits"]          # (B,T,V)
        shift_logits = logits[:, :-1]                       # predict token t+1
        shift_labels = lbl[:, 1:]
        per_pos = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1), reduction="none",
        ).view(shift_labels.shape)                          # (B, T-1)
        T = per_pos.size(1)
        bounds = [round(k * T / n_seg) for k in range(n_seg + 1)]
        for s in range(n_seg):
            chunk = per_pos[:, bounds[s]:bounds[s + 1]]
            seg_loss[s] += float(chunk.sum())
            seg_cnt[s] += chunk.numel()
    model.train()
    seg_ppl = [math.exp(min(seg_loss[s] / max(seg_cnt[s], 1), 20.0)) for s in range(n_seg)]
    return {"early_ppl": round(seg_ppl[0], 4), "late_ppl": round(seg_ppl[-1], 4),
            "late_minus_early": round(seg_ppl[-1] - seg_ppl[0], 4)}


def run(base_lr: Optional[float], seed: int, *, steps: int, batch: int, seq_len: int,
        lr: float, d_model: int, n_layers: int, n_heads: int, recal_every: int,
        eval_batches: int, device: str) -> Dict:
    np.random.seed(seed)
    vocab = json.load(open(os.path.join(DATA_DIR, "meta.json")))["vocab_size"]
    train_ds = BinDataset(os.path.join(DATA_DIR, "train.bin"), seq_len)
    val_ds = BinDataset(os.path.join(DATA_DIR, "validation.bin"), seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False,
                            drop_last=True, num_workers=0)

    model = build_model(base_lr, vocab, d_model=d_model, n_layers=n_layers,
                        n_heads=n_heads, seq_len=seq_len, device=device, seed=seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            eps=1e-8, weight_decay=0.01)
    hp = model.hebbian_plasticity

    nan_count = 0
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
            if hp is not None and step % recal_every == 0:
                saved = hp._scale
                hp._scale = 0.0
                opt.zero_grad(set_to_none=True)
                model(inp, labels=lbl)["loss"].backward()
                g_main = _grad_global_norm(model)
                opt.zero_grad(set_to_none=True)
                _ = model(inp, labels=lbl)
                raw = hp.raw_loss(model)
                raw.backward()
                g_hebb_raw = _grad_global_norm(model)
                opt.zero_grad(set_to_none=True)
                hp._scale = saved
                hp.recalibrate(g_main, g_hebb_raw)
                grad_fracs.append(hp.last_stats["grad_frac"])
                scales.append(hp._scale)

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
            step += 1

    val_ppl = evaluate(model, val_loader, device, max_batches=eval_batches)
    seg = segmented_ppl(model, val_loader, device, max_batches=eval_batches)
    return {
        "val_ppl": round(val_ppl, 4),
        "nan_count": nan_count,
        "final_grad_norm": round(last_grad_norm, 5),
        "grad_frac_mean": round(float(np.mean(grad_fracs)), 7) if grad_fracs else 0.0,
        "grad_frac_max": round(float(np.max(grad_fracs)), 7) if grad_fracs else 0.0,
        "scale_mean": round(float(np.mean(scales)), 5) if scales else None,
        **seg,
        "seconds": round(time.time() - t0, 1),
    }


def main(*, seeds=(0, 1), steps: int = 150, batch: int = 8, seq_len: int = 192,
         lr: float = 3e-4, d_model: int = 128, n_layers: int = 4, n_heads: int = 8,
         recal_every: int = 15, eval_batches: int = 40) -> Dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert os.path.exists(os.path.join(DATA_DIR, "train.bin")), f"missing {DATA_DIR}"
    t0 = time.time()

    results: Dict[str, Dict] = {}
    for name, blr in SETTINGS:
        per_seed = []
        for s in seeds:
            r = run(blr, s, steps=steps, batch=batch, seq_len=seq_len, lr=lr,
                    d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                    recal_every=recal_every, eval_batches=eval_batches, device=device)
            per_seed.append(r)
            print(f"[{name:>7s} s{s}] frac={r['grad_frac_mean']:.2e} "
                  f"val_ppl={r['val_ppl']:.2f} late={r['late_ppl']:.2f} "
                  f"L-E={r['late_minus_early']:+.2f} nan={r['nan_count']} "
                  f"scale={r['scale_mean']} ({r['seconds']:.0f}s)", flush=True)

        def agg(key):
            xs = [r[key] for r in per_seed]
            return {"mean": statistics.fmean(xs),
                    "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0}
        results[name] = {
            "base_lr": blr,
            "grad_frac_mean": statistics.fmean(r["grad_frac_mean"] for r in per_seed),
            "grad_frac_max": max(r["grad_frac_max"] for r in per_seed),
            "scale_mean": (statistics.fmean(r["scale_mean"] for r in per_seed)
                           if per_seed[0]["scale_mean"] is not None else None),
            "nan_total": sum(r["nan_count"] for r in per_seed),
            "val_ppl": agg("val_ppl"),
            "late_ppl": agg("late_ppl"),
            "late_minus_early": agg("late_minus_early"),
            "per_seed": per_seed,
        }

    report = {
        "config": dict(seeds=list(seeds), steps=steps, batch=batch, seq_len=seq_len,
                       lr=lr, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                       recal_every=recal_every, eval_batches=eval_batches,
                       device=device, red_line_cap=0.05),
        "settings": results,
        "seconds": round(time.time() - t0, 1),
    }
    _save(report)
    _print(report)
    return report


def _save(report: Dict) -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "report_ablation_hebbian_refactor.json"), "w") as f:
        json.dump(report, f, indent=2)

    s = report["settings"]
    cfg = report["config"]
    off = s["off"]
    noise = max(v["val_ppl"]["std"] for v in s.values())
    lines = ["# Stage 3 step 1: max safe & effective Hebbian gradient share", ""]
    lines.append(f"Seeds `{cfg['seeds']}` | {cfg['steps']} steps | "
                 f"{cfg['d_model']}d x {cfg['n_layers']}L | seq {cfg['seq_len']} | "
                 f"real WikiText-103 | {cfg['device']} | red line cap=5% | "
                 f"runtime {report['seconds']}s")
    lines.append("")
    lines.append("5% is a hard SAFETY ceiling, not a target. base_lr is swept to "
                 "raise the REALIZED Hebbian gradient share; we look for the largest "
                 "share that stays stable AND helps (or does not hurt). late-PPL / "
                 "late-minus-early gap (paired, within-sequence) is the sensitive "
                 "long-context-consistency probe.")
    lines.append("")
    lines.append("| setting | base_lr | realized frac (mean/max) | scale | NaN | "
                 "val PPL (dvs off) | late PPL | late-early gap (dvs off) |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
    for name, _ in SETTINGS:
        v = s[name]
        dppl = "--" if name == "off" else f"{v['val_ppl']['mean']-off['val_ppl']['mean']:+.2f}"
        dgap = "--" if name == "off" else \
            f"{v['late_minus_early']['mean']-off['late_minus_early']['mean']:+.2f}"
        scl = "--" if v["scale_mean"] is None else f"{v['scale_mean']:.3f}"
        lines.append(
            f"| {name} | {v['base_lr']} | {v['grad_frac_mean']:.2e} / "
            f"{v['grad_frac_max']:.2e} | {scl} | {v['nan_total']} | "
            f"{v['val_ppl']['mean']:.2f} ({dppl}) | {v['late_ppl']['mean']:.2f} | "
            f"{v['late_minus_early']['mean']:+.2f} ({dgap}) |")
    lines.append("")
    lines.append(f"Cross-run val-PPL seed noise (max std) ~ {noise:.2f}. A val-PPL "
                 "delta smaller than this is NOT a detectable effect.")
    lines.append("")
    with open(os.path.join(here, "report_ablation_hebbian_refactor.md"), "w") as f:
        f.write("\n".join(lines))


def _print(report: Dict) -> None:
    s = report["settings"]
    off = s["off"]
    print("\n=== STAGE 3 STEP 1: base_lr / realized-fraction sweep ===")
    for name, _ in SETTINGS:
        v = s[name]
        dppl = 0.0 if name == "off" else v["val_ppl"]["mean"] - off["val_ppl"]["mean"]
        print(f"{name:>7s}  frac={v['grad_frac_mean']:.2e}(max {v['grad_frac_max']:.2e})  "
              f"nan={v['nan_total']}  val_ppl={v['val_ppl']['mean']:.2f}"
              f"+/-{v['val_ppl']['std']:.2f} (d{dppl:+.2f})  "
              f"late-early={v['late_minus_early']['mean']:+.2f}")


if __name__ == "__main__":
    main()
