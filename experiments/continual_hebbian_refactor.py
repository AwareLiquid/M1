"""
experiments/continual_hebbian_refactor.py -- Stage 3 step 2: does the refactored
Hebbian branch (mechanism A) act as a CONSOLIDATION prior that reduces catastrophic
forgetting in continual learning?

MOTIVATION
----------
Stage 3 step 1 (report_ablation_hebbian_refactor.md) showed that, on in-distribution
next-token PPL, every non-zero Hebbian gradient share mildly-to-clearly HURTS -- there
is no safe-and-effective ratio for the standard LM objective. But the user's framing
named a DIFFERENT setting where a Hebbian co-activation prior is biologically motivated:
CONTINUAL LEARNING. A Hebbian term that reinforces the co-activation structure already
present in the weights could, in principle, make those weights stickier and so reduce
how much training on a NEW task overwrites an OLD one. in-distribution PPL and
forgetting are different metrics; step 1's negative result does not settle this.

PROTOCOL (mirrors experiments/liquid_pc/continual.py, the established forgetting probe)
  per seed, per variant:
    build model -> train on TASK A -> measure A PPL (a_before)
                -> train on TASK B -> re-measure A PPL (a_after), measure B PPL (b_after)
    forgetting = a_after - a_before        (lower = less catastrophic forgetting)
    b_after must stay low, else the variant merely BLOCKED learning B (worthless).
  We ALSO record a held-out region C never trained on (in-domain generalization).

TASKS are disjoint, far-apart token regions of WikiText-103 train.bin (~471M tokens):
A from the head, B from the far middle. SAME domain (all English Wikipedia), so this is
a "different-data" continual shift, a relatively MILD test -- if forgetting is small for
everyone, the honest conclusion is "little forgetting to consolidate against at this
scale", which is itself a valid result. True CROSS-DOMAIN OOD needs a second corpus and
is deferred (documented in the report).

VARIANTS: off (use_hebbian_refactor=False) vs on at two base_lr settings spanning the
realized-share axis from step 1 (blr100 ~3.9%, blr300 ~4.7%, both inside the 5% red
line). The PAIRED per-seed forgetting delta (on minus off) is the sensitive signal.

Run:  python -m experiments.continual_hebbian_refactor
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
from torch.utils.data import DataLoader, Dataset

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel
from train import evaluate

DATA_DIR = os.path.join(_ROOT, "kaggle_out", "tiny", "data")

# (name, base_lr).  None == use_hebbian_refactor OFF (the baseline / control).
SETTINGS = [("off", None), ("blr100", 100.0), ("blr300", 300.0)]


class SliceDataset(Dataset):
    """Non-overlapping seq_len windows drawn from a FIXED [lo, hi) token region of a
    shared uint16 memmap. Lets us carve disjoint Task A / Task B / region C from one
    WikiText-103 train.bin without re-tokenising. Labels are input-aligned (the model
    shifts internally; see train.BinDataset)."""

    def __init__(self, data: np.memmap, lo: int, hi: int, seq_len: int):
        self.data = data
        self.lo = lo
        self.hi = min(hi, len(data))
        self.seq_len = seq_len
        self.n = max(1, (self.hi - self.lo - seq_len - 1) // seq_len)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        base = self.lo + idx * self.seq_len
        max_start = self.hi - self.seq_len - 1
        start = min(base + np.random.randint(0, self.seq_len), max_start)
        chunk = self.data[start: start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        return x, x.clone()


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
        hebbian_grad_frac_cap=0.05,   # hard red line, fixed for every setting
    )
    return MTLNNModel(cfg).to(device)


def train_phase(model, opt, hp, loader, *, steps: int, recal_every: int,
                device: str, grad_fracs: List[float]) -> int:
    """Train `model` for `steps` optimiser steps over `loader`, with the periodic
    two-backward Hebbian gradient recalibration (only when hp is not None). Appends
    the realized post-scale gradient fraction at each recalibration to `grad_fracs`
    (a normal training step's compute_signal overwrites last_stats, so we must read
    grad_frac AT the recal, not after). Returns the NaN count (loss non-finite)."""
    nan_count = 0
    model.train()
    step = 0
    while step < steps:
        for inp, lbl in loader:
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
                if raw is not None:
                    raw.backward()
                    g_hebb_raw = _grad_global_norm(model)
                    opt.zero_grad(set_to_none=True)
                    hp._scale = saved
                    hp.recalibrate(g_main, g_hebb_raw)
                    grad_fracs.append(hp.last_stats["grad_frac"])
                else:
                    hp._scale = saved

            opt.zero_grad(set_to_none=True)
            out = model(inp, labels=lbl)
            loss = out["loss"]
            if not torch.isfinite(loss):
                nan_count += 1
                opt.zero_grad(set_to_none=True)
                step += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
    return nan_count


def run(base_lr: Optional[float], seed: int, *, steps_a: int, steps_b: int, batch: int,
        seq_len: int, lr: float, d_model: int, n_layers: int, n_heads: int,
        recal_every: int, eval_batches: int, device: str,
        a_tok: int, b_off: int, task_tok: int, c_off: int) -> Dict:
    np.random.seed(seed)
    vocab = json.load(open(os.path.join(DATA_DIR, "meta.json")))["vocab_size"]
    data = np.memmap(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16, mode="r")

    # Disjoint regions. A = head; B = far middle; C = held-out (never trained).
    # Within A and B, reserve the FIRST eval_span tokens for held-out eval and train
    # on the rest, so a_before/a_after are measured on data A-training never saw.
    eval_span = max(seq_len * (eval_batches * batch + 2), seq_len * 64)
    a_eval = SliceDataset(data, 0, eval_span, seq_len)
    a_train = SliceDataset(data, eval_span, eval_span + task_tok, seq_len)
    b_eval = SliceDataset(data, b_off, b_off + eval_span, seq_len)
    b_train = SliceDataset(data, b_off + eval_span, b_off + eval_span + task_tok, seq_len)
    c_eval = SliceDataset(data, c_off, c_off + eval_span, seq_len)

    def dl(ds, shuffle):
        return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=True,
                          num_workers=0)

    model = build_model(base_lr, vocab, d_model=d_model, n_layers=n_layers,
                        n_heads=n_heads, seq_len=seq_len, device=device, seed=seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            eps=1e-8, weight_decay=0.01)
    hp = model.hebbian_plasticity

    t0 = time.time()
    grad_fracs: List[float] = []
    nan_a = train_phase(model, opt, hp, dl(a_train, True), steps=steps_a,
                        recal_every=recal_every, device=device, grad_fracs=grad_fracs)
    a_before = evaluate(model, dl(a_eval, False), device, max_batches=eval_batches)
    nan_b = train_phase(model, opt, hp, dl(b_train, True), steps=steps_b,
                        recal_every=recal_every, device=device, grad_fracs=grad_fracs)
    a_after = evaluate(model, dl(a_eval, False), device, max_batches=eval_batches)
    b_after = evaluate(model, dl(b_eval, False), device, max_batches=eval_batches)
    c_after = evaluate(model, dl(c_eval, False), device, max_batches=eval_batches)

    return {
        "a_before": round(a_before, 4),
        "a_after": round(a_after, 4),
        "forgetting": round(a_after - a_before, 4),
        "b_after": round(b_after, 4),
        "c_after": round(c_after, 4),
        "nan_count": nan_a + nan_b,
        "grad_frac_mean": round(float(np.mean(grad_fracs)), 7) if grad_fracs else 0.0,
        "grad_frac_max": round(float(np.max(grad_fracs)), 7) if grad_fracs else 0.0,
        "scale": round(hp._scale, 5) if hp else None,
        "seconds": round(time.time() - t0, 1),
    }


def _mean_std(xs: List[float]) -> Dict:
    return {"mean": statistics.fmean(xs),
            "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
            "vals": [round(v, 4) for v in xs]}


def main(*, seeds=(0, 1), steps_a: int = 120, steps_b: int = 120, batch: int = 8,
         seq_len: int = 128, lr: float = 3e-4, d_model: int = 128, n_layers: int = 4,
         n_heads: int = 8, recal_every: int = 15, eval_batches: int = 30,
         task_tok: int = 6_000_000, b_off: int = 200_000_000,
         c_off: int = 380_000_000) -> Dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert os.path.exists(os.path.join(DATA_DIR, "train.bin")), f"missing {DATA_DIR}"
    t0 = time.time()

    results: Dict[str, Dict] = {}
    for name, blr in SETTINGS:
        per_seed = []
        for s in seeds:
            r = run(blr, s, steps_a=steps_a, steps_b=steps_b, batch=batch,
                    seq_len=seq_len, lr=lr, d_model=d_model, n_layers=n_layers,
                    n_heads=n_heads, recal_every=recal_every, eval_batches=eval_batches,
                    device=device, a_tok=task_tok, b_off=b_off, task_tok=task_tok,
                    c_off=c_off)
            per_seed.append(r)
            print(f"[{name:>7s} s{s}] A {r['a_before']:.2f}->{r['a_after']:.2f} "
                  f"forget={r['forgetting']:+.2f} B={r['b_after']:.2f} "
                  f"C={r['c_after']:.2f} nan={r['nan_count']} "
                  f"frac={r['grad_frac_mean']:.2e} ({r['seconds']:.0f}s)", flush=True)
        results[name] = {
            "base_lr": blr,
            "forgetting": _mean_std([r["forgetting"] for r in per_seed]),
            "a_before": _mean_std([r["a_before"] for r in per_seed]),
            "a_after": _mean_std([r["a_after"] for r in per_seed]),
            "b_after": _mean_std([r["b_after"] for r in per_seed]),
            "c_after": _mean_std([r["c_after"] for r in per_seed]),
            "nan_total": sum(r["nan_count"] for r in per_seed),
            "grad_frac_mean": statistics.fmean(r["grad_frac_mean"] for r in per_seed),
            "per_seed": per_seed,
        }

    # Paired per-seed forgetting delta: on minus off (negative => Hebbian forgets LESS).
    off_forget = [r["forgetting"] for r in results["off"]["per_seed"]]
    paired = {}
    for name, _ in SETTINGS:
        if name == "off":
            continue
        on_forget = [r["forgetting"] for r in results[name]["per_seed"]]
        diffs = [on - off for on, off in zip(on_forget, off_forget)]
        paired[name] = {
            "mean_diff": round(statistics.fmean(diffs), 4),
            "on_lower_in": f"{sum(1 for d in diffs if d < 0)}/{len(diffs)}",
            "diffs": [round(d, 4) for d in diffs],
        }

    report = {
        "config": dict(seeds=list(seeds), steps_a=steps_a, steps_b=steps_b, batch=batch,
                       seq_len=seq_len, lr=lr, d_model=d_model, n_layers=n_layers,
                       n_heads=n_heads, recal_every=recal_every, eval_batches=eval_batches,
                       task_tok=task_tok, b_off=b_off, c_off=c_off, device=device,
                       red_line_cap=0.05),
        "settings": results,
        "paired_forgetting_on_minus_off": paired,
        "seconds": round(time.time() - t0, 1),
    }
    _save(report)
    _print(report)
    return report


def _save(report: Dict) -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "report_continual_hebbian_refactor.json"), "w") as f:
        json.dump(report, f, indent=2)

    s = report["settings"]
    cfg = report["config"]
    p = report["paired_forgetting_on_minus_off"]
    lines = ["# Stage 3 step 2: Hebbian refactor as a continual-learning "
             "consolidation prior", ""]
    lines.append(f"Seeds `{cfg['seeds']}` | train A {cfg['steps_a']} + B "
                 f"{cfg['steps_b']} steps | {cfg['d_model']}d x {cfg['n_layers']}L | "
                 f"seq {cfg['seq_len']} | disjoint WikiText-103 regions | {cfg['device']} "
                 f"| red line cap=5% | runtime {report['seconds']}s")
    lines.append("")
    lines.append("Forgetting probe: train A -> measure A PPL -> train B -> re-measure A. "
                 "`forgetting` = A_after - A_before (lower = less catastrophic "
                 "forgetting). `B_after` must stay low or the variant merely blocked "
                 "learning B. `C` is a held-out region never trained on. Tasks are "
                 "disjoint but SAME-domain WikiText slices (a mild shift); true "
                 "cross-domain OOD needs a second corpus and is deferred.")
    lines.append("")
    lines.append("| setting | base_lr | realized frac | A_before | A_after | "
                 "forgetting | B_after | C (held-out) | NaN |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, _ in SETTINGS:
        v = s[name]
        lines.append(
            f"| {name} | {v['base_lr']} | {v['grad_frac_mean']:.2e} | "
            f"{v['a_before']['mean']:.2f} | {v['a_after']['mean']:.2f} | "
            f"{v['forgetting']['mean']:+.2f} +/- {v['forgetting']['std']:.2f} | "
            f"{v['b_after']['mean']:.2f} | {v['c_after']['mean']:.2f} | "
            f"{v['nan_total']} |")
    lines.append("")
    lines.append("Paired (per-seed) forgetting delta, Hebbian ON minus OFF "
                 "(negative => Hebbian forgets LESS):")
    lines.append("")
    lines.append("| setting | mean diff | on-lower-in |")
    lines.append("|---|---:|---:|")
    for name in p:
        lines.append(f"| {name} | {p[name]['mean_diff']:+.4f} | "
                     f"{p[name]['on_lower_in']} |")
    lines.append("")
    with open(os.path.join(here, "report_continual_hebbian_refactor.md"), "w") as f:
        f.write("\n".join(lines))


def _print(report: Dict) -> None:
    s = report["settings"]
    p = report["paired_forgetting_on_minus_off"]
    print("\n=== STAGE 3 STEP 2: continual-learning forgetting ===")
    for name, _ in SETTINGS:
        v = s[name]
        print(f"{name:>7s}  forget={v['forgetting']['mean']:+.2f}"
              f"+/-{v['forgetting']['std']:.2f}  "
              f"B_after={v['b_after']['mean']:.2f}  C={v['c_after']['mean']:.2f}  "
              f"nan={v['nan_total']}")
    for name in p:
        print(f"  paired {name} on-off forgetting: {p[name]['mean_diff']:+.4f} "
              f"(on lower in {p[name]['on_lower_in']})")


if __name__ == "__main__":
    main()
