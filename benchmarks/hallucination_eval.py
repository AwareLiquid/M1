"""Hallucination self-detection — does the metacognitive signal actually work?

The deliberation router (Module E) flags a token for self-critique when its
ENTROPY is high / self-consistency is low — the claim being that low
confidence marks likely-wrong (hallucinated) output. This benchmark tests
that claim honestly and WITHOUT an external judge, on TruthfulQA MC1 (a set
where the plausible-but-false answer is the trap).

Method (selective prediction / metacognition):
  1. Score each answer choice by length-normalised log-likelihood; argmax =
     the model's answer. Correct iff it is the MC1 gold choice -> raw accuracy.
  2. Confidence signals the router keys on: ENTROPY over the choice
     distribution, and MARGIN (top1 - top2 log-prob).
  3. The real question: does that signal SEPARATE right from wrong? Report
     AUROC(−entropy vs correct) — 0.5 = the signal is useless (the metacog
     claim fails), >0.5 = it genuinely detects its own errors. Plus a
     risk-coverage point: accuracy on the most-confident 50% of questions
     (if the model can abstain on its low-confidence answers, does accuracy
     on the rest rise?).

Compares base vs base+adapter. HONEST FRAMING: this tests the model's
UNCERTAINTY signal as a hallucination detector — the actual signal the router
gates on. It does NOT test whether the generative self-critique VOTE rewrites
answers (that needs a truthfulness judge; follow-on). A null result (AUROC
≈0.5) is a valid, useful finding: it would say the self-detection gate does
not separate hallucinations and the effort should go to retrieval-grounding.

Usage:
    python benchmarks/hallucination_eval.py --ckpt ""        # base
    python benchmarks/hallucination_eval.py --ckpt checkpoints/llama_mt_adapter/llama_mt_adapter_v2s_003000.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import datasets as _datasets  # noqa: F401  (Windows DLL-order guard)

from benchmarks.capability_eval import build_model


def auroc(scores, labels):
    """AUROC via the Mann–Whitney U statistic (no sklearn dep). scores: higher
    = more confident-correct predicted; labels: 1 correct, 0 wrong."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_pos = sum(ranks[i] for i in range(len(scores)) if labels[i] == 1)
    u = rank_pos - len(pos) * (len(pos) + 1) / 2.0
    return u / (len(pos) * len(neg))


@torch.no_grad()
def score_choice(model, tok, device, prompt, choice):
    """Length-normalised log-likelihood of `choice` continuing `prompt`.

    Robustness (both were review-confirmed hazards):
      * The continuation tokens are tokenized SEPARATELY and concatenated as
        IDs — never assume tok(prompt) is a byte-prefix of tok(prompt+choice)
        (BPE can merge across the boundary, which would misalign the scored
        span and silently corrupt every number).
      * reset the adapter's streaming state before each choice, so each choice
        is scored from ZERO recurrent state (matching the stateless base);
        without it, build_model turns streaming ON and choice N's score is
        contaminated by choice N-1's carried (F,z)/h state.
    """
    from mt_lnn.llama_adapter import reset_adapter_streams
    reset_adapter_streams(model)
    ctx = tok(prompt, return_tensors="pt").input_ids.to(device)
    cont = tok(" " + choice, add_special_tokens=False,
               return_tensors="pt").input_ids.to(device)
    if cont.shape[1] == 0:
        return -1e9
    full = torch.cat([ctx, cont], dim=1)
    out = model(full)
    logits = out.logits[:, ctx.shape[1] - 1: full.shape[1] - 1, :]
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_lp = logp.gather(-1, cont.unsqueeze(-1)).squeeze(-1)
    return (tok_lp.sum() / cont.shape[1]).item()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=0, help="0 = all questions")
    ap.add_argument("--out_dir", default="benchmarks/hallucination_out")
    args = ap.parse_args()

    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model, tag = build_model(args.model, args.ckpt, device, dtype)
    tag = args.tag or tag
    model.eval()

    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    correct, entropies, margins, labels = [], [], [], []
    for i, ex in enumerate(ds):
        q = ex["question"]
        choices = ex["mc1_targets"]["choices"]
        gold = ex["mc1_targets"]["labels"].index(1)   # MC1: exactly one correct
        prompt = f"Q: {q}\nA:"
        scores = [score_choice(model, tok, device, prompt, c) for c in choices]
        pred = int(max(range(len(scores)), key=lambda k: scores[k]))
        is_correct = int(pred == gold)
        correct.append(is_correct)
        # confidence signals over the choice distribution
        sc = torch.tensor(scores)
        probs = torch.softmax(sc, dim=0)
        ent = -(probs * torch.log(probs.clamp_min(1e-9))).sum().item()
        top2 = torch.topk(sc, min(2, len(scores))).values
        margin = (top2[0] - top2[1]).item() if len(scores) > 1 else 0.0
        entropies.append(ent)
        margins.append(margin)
        labels.append(is_correct)
        if (i + 1) % 50 == 0:
            print(f"  [{tag}] {i+1}/{len(ds)} acc {sum(correct)/len(correct):.3f}",
                  flush=True)

    acc = sum(correct) / len(correct)
    # metacognition: does LOW entropy / HIGH margin predict CORRECT?
    auc_ent = auroc([-e for e in entropies], labels)   # higher confidence = -entropy
    auc_margin = auroc(margins, labels)
    # risk-coverage: accuracy on the most-confident (lowest-entropy) 50%
    n = len(entropies)
    order = sorted(range(n), key=lambda i: entropies[i])   # ascending entropy = most confident first
    half = order[: n // 2]
    acc_conf50 = sum(correct[i] for i in half) / max(len(half), 1)

    res = {"tag": tag, "n": n, "accuracy": round(acc, 4),
           "auroc_entropy": round(auc_ent, 4), "auroc_margin": round(auc_margin, 4),
           "acc_confident50": round(acc_conf50, 4)}
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(args.out_dir, f"halluc_{tag}.json"), "w"),
              indent=2)

    print("\n" + "=" * 60, flush=True)
    print(f"HALLUCINATION SELF-DETECTION | TruthfulQA MC1 | {tag}", flush=True)
    print("=" * 60, flush=True)
    print(f"  raw accuracy            {acc:.3f}  (n={n})", flush=True)
    print(f"  AUROC(confidence→correct, entropy) {auc_ent:.3f}  "
          f"(0.5 = signal useless; >0.5 = detects own errors)", flush=True)
    print(f"  AUROC(margin)           {auc_margin:.3f}", flush=True)
    print(f"  accuracy on most-confident 50%     {acc_conf50:.3f}  "
          f"(vs {acc:.3f} overall — rise = selective prediction works)", flush=True)


if __name__ == "__main__":
    main()
