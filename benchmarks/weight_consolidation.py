"""Weight consolidation — the true episodic->semantic migration (T2->T3 skeleton).

The last structural gap in the memory system (see the memory-system audit):
every existing piece keeps episodic bindings in the fast-weight (T1) or an
external store; NOTHING migrates a binding into the slow WEIGHTS (T3-semantic).
This is that migration, framed honestly by what CAN migrate.

WHAT CAN vs CANNOT migrate. A one-shot RANDOM binding (K1->V1, different every
trial) must NOT go into weights — that would be memorizing noise, and it is
exactly what the episodic fast-weight is for. Only a RECURRING knowledge set
(the same K->V facts seen across many "days") should consolidate into weights
and become semantic. That distinction IS complementary-learning-systems:
hippocampus for one-shot arbitrary, cortex for repeated/interleaved.

PROTOCOL (CLS-style sleep consolidation, fast-weight ABLATED so recall can
ONLY come from weights):
  1. Fix a dictionary of `n_facts` (key_token -> value_token) pairs (disjoint
     vocab ranges), the same every step — the "semantic facts".
  2. Consolidate: fine-tune the SLOW weights (MT adapter + LoRA; base frozen)
     to map each ISOLATED key->value (sequence [k_i, v_i], loss at k_i->v_i),
     with the fast-weight scaled to ZERO so nothing leaks through the episodic
     channel. Interleave 1:`interleave` with base WikiText-2 LM batches — the
     CLS rehearsal that bounds catastrophic forgetting.
  3. Eval, fast-weight still ablated:
     - facts recall: present [k_i] alone, does the model produce v_i from
       WEIGHTS? (migration succeeded)
     - control recall: held-out RANDOM keys -> should be ~chance (weights did
       not memorize unseen bindings — the honest negative that proves recall is
       real memorization, not leakage)
     - base PPL on WikiText-2 vs before consolidation (catastrophic forgetting)

A skeleton: it runs end-to-end and reports the three numbers; the science is
whether facts-recall rises while control stays chance and base PPL holds. GPU
for a real result; CPU-smokeable.

Usage:
    python benchmarks/weight_consolidation.py --n_facts 32 --consolidate_steps 2000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from benchmarks.attribution_ablation import build_chunks
from benchmarks.cross_window_recall import setup
from mt_lnn.llama_adapter import _iter_all_adapters, reset_adapter_streams


def ablate_fast_weight(model, freeze: bool = True):
    """Zero every adapter's fw_scale so the fast-weight (episodic T1) channel
    contributes nothing — recall must then come from the SLOW weights."""
    n = 0
    for a in _iter_all_adapters(model):
        if getattr(a, "fw_scale", None) is not None:
            a.fw_scale.data.zero_()
            a.fw_scale.requires_grad_(not freeze)
            n += 1
    return n


def make_facts(n_facts, key_lo, key_hi, val_lo, val_hi, seed):
    """A FIXED dictionary: n_facts unique keys -> values. Same every call for a
    given seed (the recurring 'semantic' set)."""
    g = torch.Generator().manual_seed(seed)
    keys = torch.randperm(key_hi - key_lo, generator=g)[:n_facts] + key_lo
    vals = torch.randint(val_lo, val_hi, (n_facts,), generator=g)
    return keys, vals   # (n_facts,), (n_facts,)


@torch.no_grad()
def eval_facts_recall(model, keys, vals, device):
    """Present each key ALONE; is the top next-token the paired value? (pure
    weight memory — no context, fast-weight ablated)."""
    model.eval()
    reset_adapter_streams(model)
    ids = keys.view(-1, 1).to(device)                 # (n_facts, 1)
    with torch.amp.autocast("cuda", enabled=device == "cuda",
                            dtype=torch.bfloat16 if device == "cuda" else torch.float32):
        logits = model(input_ids=ids).logits[:, -1, :]  # (n_facts, V)
    pred = logits.argmax(-1).cpu()
    return (pred == vals).float().mean().item()


@torch.no_grad()
def eval_ppl(model, chunks, device, dtype, max_chunks=100):
    model.eval()
    reset_adapter_streams(model)
    nll, tok = 0.0, 0
    for i in range(0, min(len(chunks), max_chunks)):
        ids = chunks[i:i + 1].to(device)
        with torch.amp.autocast("cuda", enabled=device == "cuda", dtype=dtype):
            out = model(input_ids=ids, labels=ids)
        n = ids.shape[1] - 1
        nll += out.loss.float().item() * n; tok += n
    return math.exp(nll / tok) if tok else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--config", default="mt_v2s")
    ap.add_argument("--n_facts", type=int, default=32)
    ap.add_argument("--consolidate_steps", type=int, default=2000)
    ap.add_argument("--interleave", type=int, default=1,
                    help="base-LM batches per fact batch (CLS rehearsal; 0 = no "
                         "anti-forgetting, expect base PPL to spike)")
    ap.add_argument("--fact_batch", type=int, default=16)
    ap.add_argument("--lm_seq", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--key_lo", type=int, default=5000)
    ap.add_argument("--key_hi", type=int, default=6000)
    ap.add_argument("--val_lo", type=int, default=7000)
    ap.add_argument("--val_hi", type=int, default=8000)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wikitext", default="wikitext-2-raw-v1")
    ap.add_argument("--out_dir", default="benchmarks/weight_consolidation_out")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported()
             else (torch.float16 if device == "cuda" else torch.float32))
    chance = 1.0 / (args.val_hi - args.val_lo)
    print(f"device={device} config={args.config} n_facts={args.n_facts} "
          f"chance={chance:.4f}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    # attribution.build_chunks uses WikiText-2-raw-v1 (its hardcoded set).
    lm = build_chunks(tok, "train", args.lm_seq)
    lm_test = build_chunks(tok, "test", args.lm_seq)

    torch.manual_seed(args.seed)
    m = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    m.config.use_cache = False
    m, _ = setup(args.config, m, args.lora_r, args.lora_alpha)
    n_abl = ablate_fast_weight(m, freeze=True)      # episodic channel OFF
    m.to(device)
    print(f"ablated fast-weight on {n_abl} adapters (recall must come from weights)",
          flush=True)

    keys, vals = make_facts(args.n_facts, args.key_lo, args.key_hi,
                            args.val_lo, args.val_hi, args.seed)
    ctrl_keys, ctrl_vals = make_facts(args.n_facts, args.key_lo, args.key_hi,
                                      args.val_lo, args.val_hi, args.seed + 999)

    ppl_before = eval_ppl(m, lm_test, device, dtype)
    facts_before = eval_facts_recall(m, keys, vals, device)
    print(f"BEFORE: facts_recall {facts_before:.3f} | base_ppl {ppl_before:.2f}",
          flush=True)

    # --- consolidate: fine-tune slow weights on the fixed dictionary,
    #     interleaved with base LM (CLS). fast-weight stays ablated.
    m.train()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=args.lr)
    scaler = (torch.amp.GradScaler("cuda")
              if device == "cuda" and dtype == torch.float16 else None)
    g = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    for step in range(1, args.consolidate_steps + 1):
        reset_adapter_streams(m)
        # fact batch: isolated [k_i, v_i] sequences, loss at k_i -> v_i
        sel = torch.randint(0, args.n_facts, (args.fact_batch,), generator=g)
        seq = torch.stack([keys[sel], vals[sel]], dim=1).to(device)   # (B,2)
        with torch.amp.autocast("cuda", enabled=device == "cuda", dtype=dtype):
            logits = m(input_ids=seq).logits[:, 0, :]                 # predict tok 1
            loss = torch.nn.functional.cross_entropy(logits, seq[:, 1])
        (scaler.scale(loss).backward() if scaler else loss.backward())
        # interleaved base-LM rehearsal (CLS anti-forgetting)
        for _ in range(args.interleave):
            idx = int(torch.randint(0, len(lm), (1,), generator=g))
            ids = lm[idx:idx + 1].to(device)
            reset_adapter_streams(m)
            with torch.amp.autocast("cuda", enabled=device == "cuda", dtype=dtype):
                lm_loss = m(input_ids=ids, labels=ids).loss
            (scaler.scale(lm_loss).backward() if scaler else lm_loss.backward())
        if scaler:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
        (scaler.step(opt), scaler.update()) if scaler else opt.step()
        opt.zero_grad(set_to_none=True)
        if step % args.log_every == 0:
            fr = eval_facts_recall(m, keys, vals, device); m.train()
            dt = max(time.time() - t0, 1e-3)
            print(f"[consolidate] {step}/{args.consolidate_steps} | fact_loss "
                  f"{loss.item():.4f} | facts_recall {fr:.3f} | "
                  f"{args.log_every/dt:.2f} it/s", flush=True)
            t0 = time.time()

    facts_after = eval_facts_recall(m, keys, vals, device)
    ctrl_after = eval_facts_recall(m, ctrl_keys, ctrl_vals, device)
    ppl_after = eval_ppl(m, lm_test, device, dtype)

    res = {"config": args.config, "n_facts": args.n_facts, "chance": chance,
           "interleave": args.interleave,
           "facts_recall_before": round(facts_before, 4),
           "facts_recall_after": round(facts_after, 4),
           "control_recall_after": round(ctrl_after, 4),
           "base_ppl_before": round(ppl_before, 3),
           "base_ppl_after": round(ppl_after, 3),
           "base_ppl_delta": round(ppl_after - ppl_before, 3)}
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(args.out_dir,
              f"consolidate_{args.config}_{args.n_facts}facts.json"), "w"), indent=2)

    print("\n" + "=" * 62, flush=True)
    print("WEIGHT CONSOLIDATION (episodic->semantic migration into weights)", flush=True)
    print("=" * 62, flush=True)
    print(f"  facts recall  : {facts_before:.3f} -> {facts_after:.3f}   "
          f"(migrated into weights if this rises; chance {chance:.4f})", flush=True)
    print(f"  control recall: {ctrl_after:.3f}   "
          f"(held-out random keys — must stay ~chance)", flush=True)
    print(f"  base PPL      : {ppl_before:.2f} -> {ppl_after:.2f}   "
          f"(Δ {ppl_after - ppl_before:+.2f} — catastrophic forgetting if it spikes)",
          flush=True)
    ok = (facts_after > 0.5 and ctrl_after < 5 * chance
          and (ppl_after - ppl_before) < 0.15 * ppl_before)
    print(f"  verdict: {'MIGRATION OK' if ok else 'CHECK'}", flush=True)


if __name__ == "__main__":
    main()
