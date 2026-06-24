"""
experiments/ablation_full_modules.py -- do the self-developed modules, enabled
on the from-scratch MTLNNModel (O1 line), actually move the LM objective?

WHY THIS EXISTS
---------------
The decouple fix (plasticity.py, v2.2) and the full-stack smoke proved the
modules are jointly TRAINABLE (no dead params, graph intact, joint loop
converges). "Trainable" is necessary but NOT sufficient: a module can carry a
real gradient and still leave held-out perplexity unchanged. This script asks
the EFFECTIVENESS question the smoke test deliberately does not: enabling each
module -- and all of them together -- does it change held-out PPL beyond seed
noise, on REAL WikiText-103 (gpt2 tokens)?

ARMS (faithful to MTLNNConfig flags)
  baseline : every self-developed aux module OFF (pure LTC core + LM head)
  +hebbian : use_hebbian=True (gate ON; now decoupled from rhythm)
  +pc      : use_predictive_coding=True
  +wm      : use_world_model=True, world_model_warmup_steps=0
  +gwtb    : use_competitive_gwtb=True (+ ortho penalty)
  all      : all four ON simultaneously (the joint claim)

PER ARM, PER SEED we report:
  - final held-out PPL via train.evaluate() -- PURE next-token CE (out["lm_loss"]),
    NOT the aux-contaminated training objective. This is the honest LM metric.
  - dPPL vs baseline (negative = the module(s) helped)
  - NaN step count (stability / collapse check)
  - which aux-loss terms fire and their mean magnitude (sanity: the module is
    actually contributing a term, not silently inert)
  - runtime

HONEST SCOPE
  Defaults here are a SMALL, fast CPU screen (early signal), NOT a pretrain. A
  within-seed-noise delta is the expected prior at this scale (Exp5: Hebbian
  inert; the adapter aux A/B was within noise too). The point is to MEASURE on
  the native O1 stack with all modules genuinely wired, and to report the verdict
  the data supports -- not to manufacture a win. For a real verdict, raise
  steps / d_model / n_layers on a GPU (see GPU invocation in main()).

Run (CPU smoke):  python -m experiments.ablation_full_modules
Run (GPU, fuller): python -c "import experiments.ablation_full_modules as a; \
    a.main(seeds=(0,1,2), steps=2000, d_model=256, n_layers=4, n_heads=8, batch=16)"
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel
from train import BinDataset, evaluate  # faithful reuse of the real reader + eval

# Tokenised gpt2 bin corpus. Overridable via env so the Kaggle kernel can point
# this at /kaggle/working/data (real WikiText-103) without editing the script.
DATA_DIR = os.environ.get(
    "MTLNN_ABLATION_DATA", os.path.join(_ROOT, "kaggle_out", "tiny", "data")
)

# Which aux-loss keys each arm is expected to emit (used only for the sanity
# "did the term actually fire" report, not for scoring).
ARM_AUX_KEYS = {
    "baseline": [],
    "+hebbian": ["hebbian_loss"],
    "+pc":      ["pred_loss"],
    "+wm":      ["world_model_loss"],
    "+gwtb":    ["ortho_penalty"],
    "all":      ["hebbian_loss", "pred_loss", "world_model_loss", "ortho_penalty"],
}
ARMS = list(ARM_AUX_KEYS.keys())


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_cfg(arm: str, vocab_size: int, *, d_model: int, n_layers: int,
              n_heads: int, seq_len: int) -> MTLNNConfig:
    """Translate an arm name into MTLNNConfig flags. Only the named module(s) are
    ON; everything else stays at the baseline OFF state so each delta is
    attributable to exactly that module (or, for 'all', their conjunction)."""
    on = lambda *names: arm in ("all", *names)
    return MTLNNConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=1,
        d_head=d_model // n_heads,
        max_seq_len=seq_len,
        dropout=0.0,
        attention_dropout=0.0,
        gwtb_n_heads=1,
        # --- competitive GWT-B workspace + ortho penalty ---
        use_competitive_gwtb=on("+gwtb"),
        n_competitive_bids=3,
        competitive_score_noise=0.5,
        competitive_ortho_weight=0.01,
        # --- Hebbian plasticity (decoupled gate, see plasticity.py v2.2) ---
        use_hebbian=on("+hebbian"),
        hebbian_lr=1e-2,
        hebbian_lavi_gate=True,
        # --- predictive coding ---
        use_predictive_coding=on("+pc"),
        # --- world model (no warmup so it contributes a loss immediately) ---
        use_world_model=on("+wm"),
        world_model_warmup_steps=0,
        world_model_loss_weight=0.1,
        # rhythm stays OFF for every arm -- Hebbian no longer needs it.
        use_rhythm=False,
        global_rhythm=False,
        dynamic_scale_gates=True,
    )


def run_one(arm: str, seed: int, *, steps: int, batch: int, seq_len: int,
            lr: float, d_model: int, n_layers: int, n_heads: int,
            eval_batches: int, device: str) -> Dict:
    set_seed(seed)
    train_ds = BinDataset(os.path.join(DATA_DIR, "train.bin"), seq_len)
    val_ds = BinDataset(os.path.join(DATA_DIR, "validation.bin"), seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False,
                            drop_last=True, num_workers=0)

    vocab_size = json.load(open(os.path.join(DATA_DIR, "meta.json")))["vocab_size"]
    cfg = build_cfg(arm, vocab_size, d_model=d_model, n_layers=n_layers,
                    n_heads=n_heads, seq_len=seq_len)
    model = MTLNNModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            eps=1e-8, weight_decay=0.01)

    aux_keys = ARM_AUX_KEYS[arm]
    aux_mag: Dict[str, List[float]] = {k: [] for k in aux_keys}
    nan_count = 0

    model.train()
    step = 0
    t0 = time.time()
    while step < steps:
        for inp, lbl in train_loader:
            if step >= steps:
                break
            inp = inp.to(device)
            lbl = lbl.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(inp, labels=lbl)
            loss = out["loss"]
            # Record aux-term magnitudes (sanity: the module actually fired).
            for k in aux_keys:
                if k in out and out[k] is not None:
                    aux_mag[k].append(abs(float(out[k].detach().item())))
            if not torch.isfinite(loss):
                nan_count += 1
                step += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

    final_ppl = evaluate(model, val_loader, device, max_batches=eval_batches)
    return {
        "final_ppl": round(final_ppl, 4),
        "nan_count": nan_count,
        "aux_mag_mean": {k: (statistics.fmean(v) if v else 0.0)
                         for k, v in aux_mag.items()},
        "seconds": round(time.time() - t0, 1),
    }


def main(*, seeds: Tuple[int, ...] = (0, 1), steps: int = 200, batch: int = 8,
         seq_len: int = 64, lr: float = 3e-4, d_model: int = 128,
         n_layers: int = 2, n_heads: int = 8, eval_batches: int = 30) -> Dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert os.path.exists(os.path.join(DATA_DIR, "train.bin")), \
        f"missing data at {DATA_DIR}"
    t0 = time.time()

    results: Dict[str, Dict] = {}
    for arm in ARMS:
        per_seed = []
        for s in seeds:
            r = run_one(arm, s, steps=steps, batch=batch, seq_len=seq_len, lr=lr,
                        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                        eval_batches=eval_batches, device=device)
            per_seed.append(r)
            aux_str = " ".join(f"{k}={v:.2e}" for k, v in r["aux_mag_mean"].items())
            print(f"[{arm:>8s} seed {s}] ppl={r['final_ppl']:.3f} "
                  f"nan={r['nan_count']} {aux_str} ({r['seconds']:.0f}s)", flush=True)
        ppls = [r["final_ppl"] for r in per_seed]
        results[arm] = {
            "ppl_mean": statistics.fmean(ppls),
            "ppl_std": statistics.pstdev(ppls) if len(ppls) > 1 else 0.0,
            "ppl_vals": ppls,
            "nan_total": sum(r["nan_count"] for r in per_seed),
            "aux_mag_mean": {
                k: statistics.fmean(r["aux_mag_mean"].get(k, 0.0) for r in per_seed)
                for k in ARM_AUX_KEYS[arm]
            },
            "per_seed": per_seed,
        }

    report = {
        "config": dict(seeds=list(seeds), steps=steps, batch=batch, seq_len=seq_len,
                       lr=lr, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                       eval_batches=eval_batches, device=device, data=DATA_DIR),
        "arms": results,
        "seconds": round(time.time() - t0, 1),
    }
    _save(report)
    _print(report)
    return report


def _save(report: Dict) -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "report_ablation_full_modules.json"), "w") as f:
        json.dump(report, f, indent=2)

    cfg = report["config"]
    a = report["arms"]
    base_ppl = a["baseline"]["ppl_mean"]
    noise = max(a[arm]["ppl_std"] for arm in ARMS)
    lines = ["# Full-module ablation: do the self-developed modules move PPL?", ""]
    lines.append(f"Seeds `{cfg['seeds']}` | {cfg['steps']} steps | "
                 f"model {cfg['d_model']}d x {cfg['n_layers']}L x {cfg['n_heads']}H | "
                 f"seq {cfg['seq_len']} batch {cfg['batch']} | real WikiText-103 "
                 f"(gpt2) | {cfg['device']} | runtime {report['seconds']}s")
    lines.append("")
    lines.append("Held-out PPL is PURE next-token CE (`out['lm_loss']`), not the "
                 "aux-contaminated training objective. Each arm turns on exactly "
                 "one module (baseline = all OFF; `all` = all four ON). `dPPL` "
                 "negative = the module(s) helped.")
    lines.append("")
    lines.append("| arm | val PPL (mean+/-std) | dPPL vs baseline | NaN | aux terms (mean |.|) |")
    lines.append("|---|---:|---:|---:|---|")
    for arm in ARMS:
        v = a[arm]
        dppl = v["ppl_mean"] - base_ppl
        dstr = "--" if arm == "baseline" else f"{dppl:+.3f}"
        aux_str = ", ".join(f"{k}={val:.1e}" for k, val in v["aux_mag_mean"].items()) or "--"
        lines.append(f"| {arm} | {v['ppl_mean']:.3f} +/- {v['ppl_std']:.3f} | "
                     f"{dstr} | {v['nan_total']} | {aux_str} |")
    lines.append("")
    # Honest verdict, computed not asserted.
    best_arm, best_d = None, 0.0
    for arm in ARMS:
        if arm == "baseline":
            continue
        d = a[arm]["ppl_mean"] - base_ppl
        if d < best_d:
            best_d, best_arm = d, arm
    if best_arm is not None and abs(best_d) > 2 * noise:
        verdict = (f"`{best_arm}` improves PPL by {best_d:+.3f} vs baseline, beyond "
                   f"the ~{noise:.3f} seed noise -> a real effect at this scale; "
                   f"worth scaling up to confirm.")
    else:
        verdict = (f"no arm improves PPL beyond the ~{noise:.3f} seed noise "
                   f"(best = {best_arm} at {best_d:+.3f}). At this small CPU scale "
                   f"the modules are within-noise -- the expected prior. This "
                   f"measures; it does not prove inefficacy. Scale up on GPU "
                   f"(steps/d_model/n_layers) before drawing a final verdict.")
    lines.append(f"**Read:** {verdict}")
    lines.append("")
    with open(os.path.join(here, "report_ablation_full_modules.md"), "w") as f:
        f.write("\n".join(lines))


def _print(report: Dict) -> None:
    a = report["arms"]
    base_ppl = a["baseline"]["ppl_mean"]
    print("\n=== FULL-MODULE ABLATION ===")
    for arm in ARMS:
        v = a[arm]
        dppl = v["ppl_mean"] - base_ppl
        dstr = "  (ref)" if arm == "baseline" else f"  dPPL={dppl:+.3f}"
        print(f"{arm:>8s}  PPL={v['ppl_mean']:.3f}+/-{v['ppl_std']:.3f}{dstr}  "
              f"NaN={v['nan_total']}")


def _env_main() -> Dict:
    """Entry point that pulls knobs from env vars (AB_*), so a Kaggle kernel can
    scale the run up without editing this file. Falls back to the small CPU-smoke
    defaults when an env var is unset."""
    def _i(name: str, default: int) -> int:
        return int(os.environ.get(name, default))

    def _f(name: str, default: float) -> float:
        return float(os.environ.get(name, default))

    seeds = tuple(int(x) for x in os.environ.get("AB_SEEDS", "0,1").split(",") if x != "")
    return main(
        seeds=seeds,
        steps=_i("AB_STEPS", 200),
        batch=_i("AB_BATCH", 8),
        seq_len=_i("AB_SEQ_LEN", 64),
        lr=_f("AB_LR", 3e-4),
        d_model=_i("AB_D_MODEL", 128),
        n_layers=_i("AB_N_LAYERS", 2),
        n_heads=_i("AB_N_HEADS", 8),
        eval_batches=_i("AB_EVAL_BATCHES", 30),
    )


if __name__ == "__main__":
    _env_main()
