"""
experiments/continual_crossdomain_liquid.py
===========================================
The Gemini-differentiator validation, done honestly.

CLAIM UNDER TEST
----------------
M1's pitch against a frozen-weights frontier model (Gemini 3.1) is NOT raw
perplexity parity -- it is the orthogonal axis a frozen model structurally
cannot occupy: *continual learning without catastrophic forgetting*. The
concrete, falsifiable version of that claim is:

    Adding M1's liquid core (competitive GWT + predictive world model +
    Hebbian consolidation + predictive coding) to a backbone REDUCES how much
    training on a new domain B overwrites a previously learned domain A,
    relative to the SAME backbone with the liquid core OFF (plain SGD).

This script is the controlled test of that single sentence.

WHY THIS IS STRONGER THAN THE EARLIER PROBE
-------------------------------------------
experiments/continual_hebbian_refactor.py (Stage 3 step 2) was:
  * TINY (d_model=128, 4 layers),
  * SAME-domain (disjoint slices of ONE corpus -- a mild shift), and
  * tested only the `hebbian_refactor` term, not the full liquid core.
It returned a NEGATIVE result. This script fixes all three:
  * LM-scale backbone (d_model=512, 6 layers, 8 heads by default),
  * TRUE cross-domain (domain A and domain B are DIFFERENT corpora, e.g.
    WikiText-103 encyclopedic English vs TinyStories simple narrative),
  * the FULL liquid core ON vs OFF.

PROTOCOL  (per arm, per seed)
-----------------------------
  build model from scratch (fixed seed)
  train on A for Sa steps
    -> A_before = held-out PPL on A    (clean: A's validation split)
    -> B_before = held-out PPL on B    (cross-domain gap, never trained on B yet)
  train on B for Sb steps
    -> A_after  = held-out PPL on A    (forgetting = A_after - A_before)
    -> B_after  = held-out PPL on B    (learning  = B_before - B_after)

  forgetting (lower = better retention). B_after MUST drop vs B_before, else the
  arm merely FAILED to learn B and any "low forgetting" is meaningless.

ARMS
----
  dense  : liquid core OFF -- the ablation control (plain Transformer-equivalent
           on the SAME backbone).  This is the honest scientific control: it
           isolates the liquid core's contribution. It is NOT a foreign network,
           so param counts differ from `liquid` (the liquid modules ARE the extra
           params); both arms' param counts are reported for transparency.
  liquid : full M1 liquid core ON (competitive_gwtb + world_model + hebbian +
           predictive_coding), mirroring the M2 pretrain config.

PERPLEXITY is measured with the PURE next-token cross-entropy (train.evaluate ->
out["lm_loss"]), never the aux-contaminated training objective, so dense and
liquid are compared on the identical, honest LM metric.

VERDICT is pre-registered and reported regardless of sign:
  SUPPORTED        if paired (liquid - dense) forgetting < 0 in a MAJORITY of
                   seeds AND liquid still learns B (B_after < B_before).
  NOT-SUPPORTED    otherwise. A negative result is a valid, publishable finding
                   and is reported plainly -- no p-hacking, no goal-post moving.

Run (Kaggle, true cross-domain):
  python -m experiments.continual_crossdomain_liquid \
      --data_a data_wiki --data_b data_tiny --steps_a 800 --steps_b 800 \
      --d_model 512 --n_layers 6 --n_heads 8 --seeds 0,1,2

Run (local CPU smoke test, single corpus carved into disjoint slices):
  python -m experiments.continual_crossdomain_liquid --smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel
from train import evaluate

# Arm -> (liquid core on?, EWC anti-forgetting on?). `consolidation` is a DENSE
# backbone plus EWC, so consolidation-vs-dense cleanly isolates the mechanism.
ARM_SPEC = {
    "dense": dict(liquid=False, ewc=False),
    "liquid": dict(liquid=True, ewc=False),
    "consolidation": dict(liquid=False, ewc=True),
}
ARMS = ("dense", "liquid", "consolidation")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class BinSlice(Dataset):
    """Random seq_len windows from a [lo, hi) token region of a uint16 memmap.

    Used both for a domain's full train split (lo=0, hi=len) and for carving two
    disjoint slices out of ONE corpus in --smoke mode. Labels are input-aligned;
    the model shifts internally (see train.BinDataset)."""

    def __init__(self, data: np.memmap, lo: int, hi: int, seq_len: int):
        self.data = data
        self.lo = max(0, lo)
        self.hi = min(hi, len(data))
        self.seq_len = seq_len
        self.n = max(1, (self.hi - self.lo - seq_len - 1) // seq_len)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        max_start = self.hi - self.seq_len - 1
        base = self.lo + idx * self.seq_len
        start = min(base + int(np.random.randint(0, self.seq_len)), max_start)
        start = max(self.lo, start)
        chunk = self.data[start: start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        return x, x.clone()


class Domain:
    """A named training domain: a train memmap slice + a held-out eval memmap."""

    def __init__(self, name: str, data_dir: str, seq_len: int,
                 train_lo_frac: float = 0.0, train_hi_frac: float = 1.0,
                 eval_split: str = "validation"):
        self.name = name
        meta_path = os.path.join(data_dir, "meta.json")
        self.meta = json.load(open(meta_path))
        self.vocab = self.meta["vocab_size"]
        train = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
        lo = int(len(train) * train_lo_frac)
        hi = int(len(train) * train_hi_frac)
        self.train = BinSlice(train, lo, hi, seq_len)

        eval_path = os.path.join(data_dir, f"{eval_split}.bin")
        if os.path.exists(eval_path):
            ev = np.memmap(eval_path, dtype=np.uint16, mode="r")
            self.eval = BinSlice(ev, 0, len(ev), seq_len)
        else:
            # No held-out split: reserve a disjoint tail of the train region.
            mid = lo + int((hi - lo) * 0.9)
            self.eval = BinSlice(train, mid, hi, seq_len)


def dl(ds: Dataset, batch: int, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=True, num_workers=0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(liquid: bool, vocab: int, *, d_model: int, n_layers: int,
                n_heads: int, seq_len: int, device: str, seed: int) -> MTLNNModel:
    """liquid=True  -> full M1 liquid core ON (mirrors the M2 pretrain config).
    liquid=False -> the ablation control: every bio-inspired module OFF, a plain
    attention LM on the same backbone."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    common = dict(
        vocab_size=vocab, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        n_kv_heads=1, d_head=d_model // n_heads, max_seq_len=seq_len,
        dropout=0.0, attention_dropout=0.0, tie_embeddings=True,
    )
    if liquid:
        cfg = MTLNNConfig(
            **common,
            use_competitive_gwtb=True, n_competitive_bids=3,
            use_world_model=True, world_model_loss_weight=0.01,
            use_hebbian=True, hebbian_lr=1e-4,
            use_predictive_coding=True, predictive_loss_weight=0.05,
            use_graceful_degradation=True,
        )
    else:
        cfg = MTLNNConfig(
            **common,
            use_competitive_gwtb=False, use_world_model=False,
            use_hebbian=False, use_hebbian_refactor=False,
            use_predictive_coding=False, use_rhythm=False,
        )
    return MTLNNModel(cfg).to(device)


def n_params(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


# ---------------------------------------------------------------------------
# Anti-forgetting mechanism: Elastic Weight Consolidation (EWC)
# ---------------------------------------------------------------------------
class EWC:
    """Elastic Weight Consolidation -- the *real* reconsolidation mechanism.

    Catastrophic forgetting happens because plain SGD on task B is free to move
    every weight, including the ones that encode task A. EWC (Kirkpatrick et al.
    2017) makes the weights that mattered for A *stiff* during B:

      1. After training on A, SNAPSHOT the params (the anchor theta_A).
      2. Estimate the diagonal of the Fisher information F by accumulating the
         squared gradient of A's log-likelihood (= pure next-token CE) over a few
         A batches. F_i is large for params whose perturbation hurts A's loss.
      3. During B, add a quadratic penalty   (lambda/2) * sum_i F_i (theta_i - theta_A_i)^2
         so SGD pays a price proportional to F_i for drifting away from A's
         solution -- important-for-A params are anchored, free params adapt to B.

    This is a TRAINING-PROCEDURE mechanism, architecture-agnostic: it wraps the
    SAME backbone, so `consolidation` vs `dense` is a clean ablation that isolates
    EWC's contribution exactly the way `liquid` vs `dense` isolates the core.

    Fisher is estimated from out["loss"]; on a dense backbone (no aux terms) that
    IS the pure cross-entropy, so the importance estimate is uncontaminated."""

    def __init__(self, model: torch.nn.Module, loader: DataLoader, device: str,
                 n_batches: int, lam: float):
        self.lam = float(lam)
        self.anchor = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = {n: torch.zeros_like(p) for n, p in self.anchor.items()}
        was_training = model.training
        model.eval()  # freeze stochasticity; grads still flow
        seen = 0
        for inp, lbl in loader:
            if seen >= n_batches:
                break
            inp, lbl = inp.to(device), lbl.to(device)
            model.zero_grad(set_to_none=True)
            loss = model(inp, labels=lbl)["loss"]
            if not torch.isfinite(loss):
                continue
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None and n in self.fisher:
                    self.fisher[n] += p.grad.detach() ** 2
            seen += 1
        denom = max(seen, 1)
        for n in self.fisher:
            self.fisher[n] /= denom
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()

    def penalty(self, model: torch.nn.Module) -> torch.Tensor:
        dev = next(model.parameters()).device
        loss = torch.zeros((), device=dev)
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss = loss + (self.fisher[n] * (p - self.anchor[n]) ** 2).sum()
        return 0.5 * self.lam * loss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_phase(model, opt, loader, *, steps: int, warmup: int, lr: float,
                device: str, ewc: "Optional[EWC]" = None) -> int:
    """Train for `steps` optimiser steps on out["loss"] (CE + aux for the liquid
    arm, pure CE for dense). If `ewc` is given, its quadratic anchor penalty is
    added (active only in phase B). Linear warmup then constant LR. Returns NaN
    count."""
    model.train()
    nan_count = 0
    step = 0
    while step < steps:
        for inp, lbl in loader:
            if step >= steps:
                break
            inp, lbl = inp.to(device), lbl.to(device)
            cur_lr = lr * min(1.0, (step + 1) / max(1, warmup))
            for g in opt.param_groups:
                g["lr"] = cur_lr
            opt.zero_grad(set_to_none=True)
            loss = model(inp, labels=lbl)["loss"]
            if ewc is not None:
                loss = loss + ewc.penalty(model)
            if not torch.isfinite(loss):
                nan_count += 1
                step += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
    return nan_count


def run_arm(arm: str, seed: int, dom_a: Domain, dom_b: Domain, *,
            steps_a: int, steps_b: int, batch: int, seq_len: int, lr: float,
            warmup: int, d_model: int, n_layers: int, n_heads: int,
            eval_batches: int, device: str, ewc_lambda: float,
            fisher_batches: int) -> Dict:
    spec = ARM_SPEC[arm]
    np.random.seed(seed)
    model = build_model(spec["liquid"], dom_a.vocab, d_model=d_model,
                        n_layers=n_layers, n_heads=n_heads, seq_len=seq_len,
                        device=device, seed=seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            eps=1e-8, weight_decay=0.01)
    t0 = time.time()

    nan_a = train_phase(model, opt, dl(dom_a.train, batch, True), steps=steps_a,
                        warmup=warmup, lr=lr, device=device)
    a_before = evaluate(model, dl(dom_a.eval, batch, False), device, max_batches=eval_batches)
    b_before = evaluate(model, dl(dom_b.eval, batch, False), device, max_batches=eval_batches)

    # Anti-forgetting: snapshot A's solution + Fisher importance, then anchor B.
    ewc = None
    if spec["ewc"] and ewc_lambda > 0:
        ewc = EWC(model, dl(dom_a.train, batch, True), device,
                  n_batches=fisher_batches, lam=ewc_lambda)

    nan_b = train_phase(model, opt, dl(dom_b.train, batch, True), steps=steps_b,
                        warmup=warmup, lr=lr, device=device, ewc=ewc)
    a_after = evaluate(model, dl(dom_a.eval, batch, False), device, max_batches=eval_batches)
    b_after = evaluate(model, dl(dom_b.eval, batch, False), device, max_batches=eval_batches)

    return {
        "n_params_M": round(n_params(model), 2),
        "a_before": round(a_before, 4),
        "a_after": round(a_after, 4),
        "forgetting": round(a_after - a_before, 4),
        "b_before": round(b_before, 4),
        "b_after": round(b_after, 4),
        "b_learned": round(b_before - b_after, 4),
        "nan_count": nan_a + nan_b,
        "seconds": round(time.time() - t0, 1),
    }


def _mean_std(xs: List[float]) -> Dict:
    return {"mean": round(statistics.fmean(xs), 4),
            "std": round(statistics.pstdev(xs) if len(xs) > 1 else 0.0, 4),
            "vals": [round(v, 4) for v in xs]}


def _decide_verdict(paired: Dict, treatment_learns: bool, arms: Dict,
                    treatment: str, catastrophic_ratio: float = 5.0) -> Dict:
    """Honest verdict, robust to seed noise and to absolute catastrophe.

    A favourable MEAN direction is necessary but NOT sufficient. We additionally
    require the effect to clear the seed-to-seed variance (signal-to-noise ratio
    SNR = |mean_diff| / std_diff >= 1) and to hold in EVERY seed -- because with a
    handful of seeds, a sign-flipping effect whose std exceeds its mean is
    indistinguishable from noise. Separately, we flag whether forgetting is
    CATASTROPHIC for both the control and the treatment (A's PPL blows up >
    catastrophic_ratio x): if so, the mechanism has at best *reduced the damage*,
    not *enabled continual learning*, so the claim is not met in absolute terms.

      SUPPORTED      robust relative reduction AND A stays usable for treatment.
      WEAK-TREND     mean favours treatment but noise-dominated, OR both still
                     catastrophically forget (mechanism Built, effect a Target).
      NOT-SUPPORTED  no favourable direction.
    """
    md, sd = paired["mean_diff"], paired["std_diff"]
    n_lower, n_total = (int(x) for x in paired["treatment_lower_in"].split("/"))
    snr = abs(md) / (sd + 1e-9)

    def catastrophic(arm: str) -> bool:
        a = arms[arm]
        return a["a_after"]["mean"] > catastrophic_ratio * a["a_before"]["mean"]

    both_catastrophic = catastrophic("dense") and catastrophic(treatment)
    robust = (md < 0) and (snr >= 1.0) and (n_lower == n_total) and treatment_learns

    if robust and not both_catastrophic:
        verdict = "SUPPORTED"
    elif (md < 0) and (n_lower > n_total / 2) and treatment_learns:
        verdict = "WEAK-TREND"
    else:
        verdict = "NOT-SUPPORTED"
    return {"verdict": verdict, "effect_snr": round(snr, 3),
            "both_arms_catastrophic": both_catastrophic}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(args) -> Dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_len = args.seq_len
    seeds = tuple(int(s) for s in str(args.seeds).split(",") if s != "")
    arms = tuple(a for a in str(args.arms).split(",") if a in ARM_SPEC)
    treatment = args.treatment if args.treatment in arms else arms[-1]
    assert "dense" in arms, "dense control arm is required"
    assert treatment != "dense", "treatment must be a non-control arm"

    if args.smoke:
        # Single-corpus disjoint slices: validates the harness, NOT the claim.
        a_dir = b_dir = args.data_a
        dom_a = Domain("A_slice", a_dir, seq_len, 0.0, 0.45)
        dom_b = Domain("B_slice", b_dir, seq_len, 0.50, 0.95)
    else:
        dom_a = Domain(args.name_a, args.data_a, seq_len)
        dom_b = Domain(args.name_b, args.data_b, seq_len)
    assert dom_a.vocab == dom_b.vocab, "domains must share a tokenizer/vocab"

    print(f"[setup] device={device} | A={dom_a.name}({len(dom_a.train)} win) "
          f"B={dom_b.name}({len(dom_b.train)} win) | seq={seq_len} batch={args.batch} "
          f"| steps A{args.steps_a}/B{args.steps_b} | seeds={seeds} "
          f"| {args.d_model}d x {args.n_layers}L x {args.n_heads}H", flush=True)

    results: Dict[str, Dict] = {}
    for arm in arms:
        per_seed = []
        for s in seeds:
            r = run_arm(arm, s, dom_a, dom_b, steps_a=args.steps_a,
                        steps_b=args.steps_b, batch=args.batch, seq_len=seq_len,
                        lr=args.lr, warmup=args.warmup, d_model=args.d_model,
                        n_layers=args.n_layers, n_heads=args.n_heads,
                        eval_batches=args.eval_batches, device=device,
                        ewc_lambda=args.ewc_lambda, fisher_batches=args.fisher_batches)
            per_seed.append(r)
            print(f"[{arm:>6s} s{s}] {r['n_params_M']:.1f}M | "
                  f"A {r['a_before']:.2f}->{r['a_after']:.2f} "
                  f"forget={r['forgetting']:+.2f} | "
                  f"B {r['b_before']:.2f}->{r['b_after']:.2f} "
                  f"learned={r['b_learned']:+.2f} | nan={r['nan_count']} "
                  f"({r['seconds']:.0f}s)", flush=True)
        results[arm] = {
            "n_params_M": per_seed[0]["n_params_M"],
            "forgetting": _mean_std([r["forgetting"] for r in per_seed]),
            "a_before": _mean_std([r["a_before"] for r in per_seed]),
            "a_after": _mean_std([r["a_after"] for r in per_seed]),
            "b_before": _mean_std([r["b_before"] for r in per_seed]),
            "b_after": _mean_std([r["b_after"] for r in per_seed]),
            "b_learned": _mean_std([r["b_learned"] for r in per_seed]),
            "nan_total": sum(r["nan_count"] for r in per_seed),
            "per_seed": per_seed,
        }

    # Paired per-seed forgetting delta vs the dense control (negative => the arm
    # forgets LESS than dense). Computed for every non-control arm; the headline
    # verdict is decided on the `treatment` arm.
    dense_f = [r["forgetting"] for r in results["dense"]["per_seed"]]

    def _paired(arm: str) -> Dict:
        arm_f = [r["forgetting"] for r in results[arm]["per_seed"]]
        diffs = [t - dn for t, dn in zip(arm_f, dense_f)]
        n_lower = sum(1 for d in diffs if d < 0)
        return {
            "mean_diff": round(statistics.fmean(diffs), 4),
            "std_diff": round(statistics.pstdev(diffs) if len(diffs) > 1 else 0.0, 4),
            "treatment_lower_in": f"{n_lower}/{len(diffs)}",
            "diffs": [round(d, 4) for d in diffs],
        }

    paired_all = {arm: _paired(arm) for arm in arms if arm != "dense"}
    paired = paired_all[treatment]
    treatment_learns = results[treatment]["b_learned"]["mean"] > 0
    verdict_info = _decide_verdict(paired, treatment_learns, results, treatment)

    report = {
        "config": dict(seeds=list(seeds), steps_a=args.steps_a, steps_b=args.steps_b,
                       batch=args.batch, seq_len=seq_len, lr=args.lr,
                       warmup=args.warmup, d_model=args.d_model, n_layers=args.n_layers,
                       n_heads=args.n_heads, eval_batches=args.eval_batches,
                       domain_a=dom_a.name, domain_b=dom_b.name, device=device,
                       arms=list(arms), treatment=treatment,
                       ewc_lambda=args.ewc_lambda, fisher_batches=args.fisher_batches,
                       smoke=bool(args.smoke)),
        "arms": results,
        "treatment": treatment,
        "paired_forgetting_vs_dense": paired_all,
        "paired_forgetting_treatment_minus_dense": paired,
        "treatment_learns_b": treatment_learns,
        **verdict_info,
        "verdict": verdict_info["verdict"],
    }
    _save(report, args.out)
    _print(report)
    return report


def _save(report: Dict, out_prefix: str) -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(here, out_prefix)
    with open(base + ".json", "w") as f:
        json.dump(report, f, indent=2)

    c, a = report["config"], report["arms"]
    arms = c.get("arms", list(ARMS))
    treatment = report["treatment"]
    p = report["paired_forgetting_treatment_minus_dense"]
    L = [f"# Cross-domain continual learning: does the `{treatment}` mechanism reduce forgetting?",
         "",
         f"True cross-domain forgetting probe. Train on **A={c['domain_a']}**, then on "
         f"**B={c['domain_b']}**; `forgetting` = A_after_PPL - A_before_PPL (lower = "
         f"better retention), measured on each domain's held-out split with PURE "
         f"next-token cross-entropy. `b_learned` = B_before - B_after must be > 0 or "
         f"the arm failed to learn B.",
         "",
         f"Arms: `dense` (no anti-forgetting control), `liquid` (full liquid core), "
         f"`consolidation` (dense backbone + EWC: Fisher-weighted anchor to A, "
         f"lambda={c.get('ewc_lambda')}, fisher_batches={c.get('fisher_batches')}). "
         f"Headline treatment = **{treatment}** (paired vs the dense control).",
         "",
         f"Seeds `{c['seeds']}` | train A {c['steps_a']} + B {c['steps_b']} steps | "
         f"{c['d_model']}d x {c['n_layers']}L x {c['n_heads']}H | seq {c['seq_len']} | "
         f"batch {c['batch']} | lr {c['lr']} | {c['device']}"
         + ("  **(SMOKE: single-corpus slices, harness check only)**" if c["smoke"] else ""),
         "",
         "| arm | params | A_before | A_after | forgetting | B_before | B_after | "
         "B_learned | NaN |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in arms:
        v = a[arm]
        L.append(
            f"| {arm} | {v['n_params_M']:.1f}M | {v['a_before']['mean']:.2f} | "
            f"{v['a_after']['mean']:.2f} | "
            f"{v['forgetting']['mean']:+.2f} +/- {v['forgetting']['std']:.2f} | "
            f"{v['b_before']['mean']:.2f} | {v['b_after']['mean']:.2f} | "
            f"{v['b_learned']['mean']:+.2f} | {v['nan_total']} |")
    L += ["",
          f"**Paired (per-seed) forgetting delta, {treatment} - dense:** "
          f"{p['mean_diff']:+.4f} +/- {p['std_diff']:.4f}  "
          f"(SNR={report.get('effect_snr')}; {treatment} forgets less in "
          f"{p['treatment_lower_in']} seeds; per-seed diffs {p['diffs']})"]
    other = [arm for arm in arms if arm not in ("dense", treatment)]
    for arm in other:
        q = report["paired_forgetting_vs_dense"][arm]
        L.append(f"**(also) {arm} - dense:** {q['mean_diff']:+.4f} +/- {q['std_diff']:.4f} "
                 f"({arm} lower in {q['treatment_lower_in']}; diffs {q['diffs']})")
    L += ["",
          f"**{treatment} learns B:** {report['treatment_learns_b']}  |  "
          f"**Both control & treatment catastrophically forget A:** "
          f"{report.get('both_arms_catastrophic')}",
          "",
          f"## Verdict: {report['verdict']}",
          "",
          "Grading: SUPPORTED requires a relative forgetting reduction that clears "
          "seed variance (SNR>=1, treatment lower in EVERY seed) AND leaves A usable. "
          "WEAK-TREND = the mean favours the treatment but the effect is "
          "noise-dominated and/or both arms still forget A catastrophically -- the "
          "anti-forgetting mechanism is BUILT but its effectiveness is a TARGET, "
          "not yet validated. The verdict is reported as-is regardless of sign."]
    with open(base + ".md", "w") as f:
        f.write("\n".join(L))


def _print(report: Dict) -> None:
    p = report["paired_forgetting_treatment_minus_dense"]
    treatment = report["treatment"]
    arms = report["config"].get("arms", list(ARMS))
    print(f"\n=== CROSS-DOMAIN CONTINUAL: {treatment} vs dense ===")
    for arm in arms:
        v = report["arms"][arm]
        print(f"{arm:>14s} ({v['n_params_M']:.1f}M)  forget="
              f"{v['forgetting']['mean']:+.2f}+/-{v['forgetting']['std']:.2f}  "
              f"B_learned={v['b_learned']['mean']:+.2f}  nan={v['nan_total']}")
    print(f"  paired {treatment}-dense forgetting: {p['mean_diff']:+.4f} "
          f"({treatment} lower in {p['treatment_lower_in']})")
    print(f"  VERDICT: {report['verdict']}")


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_a", default=os.path.join(_ROOT, "kaggle_out", "tiny", "data"))
    ap.add_argument("--data_b", default=os.path.join(_ROOT, "kaggle_out", "tiny", "data"))
    ap.add_argument("--name_a", default="wikitext103")
    ap.add_argument("--name_b", default="tinystories")
    ap.add_argument("--steps_a", type=int, default=800)
    ap.add_argument("--steps_b", type=int, default=800)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--eval_batches", type=int, default=50)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--arms", default="dense,liquid,consolidation",
                    help="comma list from {dense,liquid,consolidation}; dense required")
    ap.add_argument("--treatment", default="consolidation",
                    help="non-control arm the headline verdict is decided on")
    ap.add_argument("--ewc_lambda", type=float, default=5000.0,
                    help="EWC anchor strength for the consolidation arm (0 disables)")
    ap.add_argument("--fisher_batches", type=int, default=50,
                    help="number of A batches used to estimate the diagonal Fisher")
    ap.add_argument("--out", default="report_continual_crossdomain_liquid")
    ap.add_argument("--smoke", action="store_true",
                    help="single-corpus disjoint slices + tiny config: harness check only")
    return ap


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    if args.smoke:
        # Override to a tiny, CPU-friendly config that exercises every code path.
        args.steps_a = args.steps_a if args.steps_a != 800 else 4
        args.steps_b = args.steps_b if args.steps_b != 800 else 4
        args.batch = 4
        args.seq_len = 64
        args.d_model = 128
        args.n_layers = 2
        args.n_heads = 4
        args.eval_batches = 3
        args.warmup = 2
        args.seeds = "0,1"
        args.fisher_batches = 3
        if args.ewc_lambda == 5000.0:
            args.ewc_lambda = 1000.0
    main(args)
