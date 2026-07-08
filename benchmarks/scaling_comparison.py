"""Scaling comparison at ~125M — the external review's remaining validation.

Two questions the review said still gate the project, at 10x the 48M O1 scale:

  1. STABILITY. "Does an ODE/liquid-recurrent net even TRAIN STABLY when scaled
     100x, or does it diverge?" -> `--mode train`: build ~125M, train N steps
     on real text, assert no NaN/inf, report the loss trajectory + val PPL vs a
     matched-width vanilla Transformer and a matched-width LTC-LNN.

  2. THE ACTUAL EDGE. The recurrent architecture's thesis is O(1) inference
     state / no KV cache -> memory that does NOT grow with context. -> `--mode
     profile`: measure PEAK memory + throughput as SEQUENCE LENGTH grows for
     each arch. Transformer attention memory grows with T; the MT-LNN recurrent
     mixer should stay far flatter. This is the honest differentiator and is
     cheap to measure (a few fwd+bwd passes, no full training).

Matched WIDTH/DEPTH (d_model=832, 12 layers — the MTLNNConfig default, whose
comment already targets ~125M) isolates the token-mixer; actual param counts
per arch are reported (they differ by the mixer). Lean MT-LNN core trunk only
(the module switch-matrix showed the optional bio modules are PPL-neutral).

Platform-neutral, resume-safe (per-arch/mode JSON). Colab/Kaggle:
    !python benchmarks/scaling_comparison.py --mode profile
    !python benchmarks/scaling_comparison.py --mode train --steps 2000
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

import datasets as _datasets  # noqa: F401  (Windows DLL-order guard)

ARCHS = ["transformer", "lnn", "mt_lnn"]


def count_params(m):
    """Uniform total-parameter count across archs — the models' own
    get_num_params() disagree (baselines exclude embeddings, MTLNNModel
    includes them), which would make the comparison apples-to-oranges."""
    return sum(p.numel() for p in m.parameters())


def build(arch, d_model, n_layers, vocab, seq_len, device, dtype):
    from benchmarks.baselines import (BaselineConfig, SimpleCausalLNN,
                                       SimpleCausalTransformer)

    if arch in ("transformer", "lnn"):
        cfg = BaselineConfig(vocab_size=vocab, max_seq_len=seq_len,
                             d_model=d_model, n_layers=n_layers,
                             n_heads=13, d_ff=4 * d_model, dropout=0.0)
        m = (SimpleCausalTransformer(cfg) if arch == "transformer"
             else SimpleCausalLNN(cfg))
    else:
        from mt_lnn.config import MTLNNConfig
        from mt_lnn.model import MTLNNModel
        cfg = MTLNNConfig(
            vocab_size=vocab, max_seq_len=seq_len,
            d_model=d_model, n_layers=n_layers, n_heads=13, n_kv_heads=1,
            d_head=d_model // 13, dropout=0.0, attention_dropout=0.0,
            # Tie embeddings to match the baselines (BaselineConfig ties by
            # default) — otherwise MT-LNN's untied 2x(vocab x d_model) inflates
            # its total by ~84M at 832-wide and the comparison confounds the
            # mixer cost with an embedding-tying choice.
            tie_embeddings=True,
            # Lean core trunk: the switch-matrix showed the optional modules are
            # PPL-neutral and cost throughput, so the fair MT-LNN is core-only.
            use_predictive_coding=False, use_competitive_gwtb=False,
            use_world_model=False, use_hebbian=False, use_rhythm=False,
        )
        m = MTLNNModel(cfg)
    return m.to(device=device, dtype=dtype)


def profile_arch(arch, args, device, dtype):
    """Peak memory + throughput vs sequence length — the O(1) demonstration."""
    rows = []
    for T in [int(x) for x in args.profile_lens.split(",")]:
        try:
            m = build(arch, args.d_model, args.n_layers, args.vocab, T, device, dtype)
            n_params = count_params(m)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
            ids = torch.randint(0, args.vocab, (args.profile_batch, T), device=device)
            if device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            for _ in range(args.profile_iters):
                opt.zero_grad(set_to_none=True)
                out = m(ids, labels=ids)
                out["loss"].backward()
                opt.step()
            if device == "cuda":
                torch.cuda.synchronize()
            dt = (time.time() - t0) / args.profile_iters
            peak_mb = (torch.cuda.max_memory_allocated() / 2**20
                       if device == "cuda" else float("nan"))
            toks = args.profile_batch * T
            rows.append({"seq_len": T, "params": n_params,
                         "peak_mb": round(peak_mb, 1), "step_s": round(dt, 4),
                         "tok_s": round(toks / dt, 1)})
            print(f"  [{arch:11s}] T={T:5d} | {n_params/1e6:.1f}M | "
                  f"peak {peak_mb:7.1f} MB | {toks/dt:8.0f} tok/s", flush=True)
            del m, opt, out, ids
            if device == "cuda":
                torch.cuda.empty_cache()
        except RuntimeError as e:
            oom = "out of memory" in str(e).lower()
            rows.append({"seq_len": T, "error": "OOM" if oom else str(e)[:80]})
            print(f"  [{arch:11s}] T={T:5d} | {'OOM' if oom else 'ERR'}", flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()
            if oom:
                break
    return rows


def build_chunks(tok, split, seq_len, wikitext="wikitext-103-raw-v1"):
    from datasets import load_dataset
    ds = load_dataset("wikitext", wikitext, split=split)
    texts = [t for t in ds["text"] if t]
    ids = []
    for i in range(0, len(texts), 1000):
        for row in tok(texts[i:i + 1000])["input_ids"]:
            ids.extend(row)
            ids.append(tok.eos_token_id)
    n = (len(ids) // seq_len) * seq_len
    return torch.tensor([ids[i:i + seq_len] for i in range(0, n, seq_len)],
                        dtype=torch.long)


def train_arch(arch, args, device, dtype):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    train_c = build_chunks(tok, "train", args.seq_len, args.wikitext)
    test_c = build_chunks(tok, "test", args.seq_len, args.wikitext)
    g = torch.Generator().manual_seed(0)
    order = torch.randperm(len(train_c), generator=g)

    m = build(arch, args.d_model, args.n_layers, args.vocab, args.seq_len, device, dtype)
    n_params = count_params(m)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, betas=(0.9, 0.95))
    scaler = (torch.amp.GradScaler("cuda")
              if device == "cuda" and dtype == torch.float16 else None)
    m.train()
    stable, last, t0, step = True, float("nan"), time.time(), 0
    while step < args.steps:
        for idx in order:
            if step >= args.steps:
                break
            ids = train_c[int(idx):int(idx) + args.batch].to(device)
            if ids.shape[0] < 1:
                continue
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=dtype, enabled=device == "cuda"):
                out = m(ids, labels=ids)
                loss = out["loss"]
            if not torch.isfinite(loss):
                stable = False
                print(f"  [{arch}] NON-FINITE loss at step {step} — UNSTABLE", flush=True)
                break
            (scaler.scale(loss).backward() if scaler else loss.backward())
            if scaler:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            (scaler.step(opt), scaler.update()) if scaler else opt.step()
            last = loss.item()
            step += 1
            if step % args.log_every == 0:
                dt = max(time.time() - t0, 1e-3)
                print(f"  [{arch:11s}] {step}/{args.steps} loss {last:.4f} "
                      f"| {args.log_every*args.batch*args.seq_len/dt:.0f} tok/s", flush=True)
                t0 = time.time()
        if not stable:
            break

    # held-out PPL
    m.eval()
    nll, ntok = 0.0, 0
    with torch.no_grad():
        for i in range(0, min(len(test_c), args.eval_chunks or len(test_c)), args.batch):
            ids = test_c[i:i + args.batch].to(device)
            with torch.amp.autocast("cuda", dtype=dtype, enabled=device == "cuda"):
                out = m(ids, labels=ids)
            n = ids.shape[0] * (ids.shape[1] - 1)
            nll += out["loss"].float().item() * n
            ntok += n
    ppl = math.exp(nll / ntok) if ntok else float("nan")
    return {"arch": arch, "params": n_params, "stable": stable,
            "final_loss": last, "val_ppl": ppl, "steps": step}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["profile", "train"], default="profile")
    ap.add_argument("--archs", default="all")
    ap.add_argument("--d_model", type=int, default=832)   # 13*64, MTLNNConfig default
    ap.add_argument("--n_layers", type=int, default=12)   # ~125M
    ap.add_argument("--vocab", type=int, default=50257)   # gpt2
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--eval_chunks", type=int, default=200)
    ap.add_argument("--wikitext", default="wikitext-103-raw-v1",
                    help="wikitext-2-raw-v1 for a cheap smoke")
    ap.add_argument("--profile_lens", default="512,1024,2048,4096")
    ap.add_argument("--profile_batch", type=int, default=2)
    ap.add_argument("--profile_iters", type=int, default=3)
    ap.add_argument("--out_dir", default="benchmarks/scaling_out")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported()
             else (torch.float16 if device == "cuda" else torch.float32))
    archs = ARCHS if args.archs == "all" else [a for a in args.archs.split(",") if a in ARCHS]
    print(f"device={device} dtype={dtype} mode={args.mode} "
          f"d_model={args.d_model} n_layers={args.n_layers} archs={archs}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    results = {}
    for arch in archs:
        path = os.path.join(args.out_dir, f"{args.mode}_{arch}.json")
        if os.path.exists(path):
            results[arch] = json.load(open(path)); print(f"[skip] {arch}", flush=True); continue
        print(f"\n=== {arch} ({args.mode}) ===", flush=True)
        r = (profile_arch(arch, args, device, dtype) if args.mode == "profile"
             else train_arch(arch, args, device, dtype))
        results[arch] = r
        json.dump(r, open(path, "w"), indent=2)

    print("\n" + "=" * 66, flush=True)
    if args.mode == "profile":
        print(f"SCALING PROFILE | peak MB (lower+flatter = better) | "
              f"d_model={args.d_model} x {args.n_layers}L", flush=True)
        lens = [int(x) for x in args.profile_lens.split(",")]
        hdr = "arch         " + "".join(f"T={T:<10}" for T in lens)
        print(hdr, flush=True)
        for arch in archs:
            cells = {row["seq_len"]: row for row in results[arch]}
            line = f"{arch:<12} "
            for T in lens:
                r = cells.get(T, {})
                line += (f"{r['peak_mb']:>5.0f}MB " if "peak_mb" in r
                         else f"{r.get('error','-'):>7} ")[:11]
            print(line, flush=True)
    else:
        print(f"SCALING TRAIN | WikiText-103 | {args.steps} steps | "
              f"d_model={args.d_model} x {args.n_layers}L", flush=True)
        print("(embeddings tied+identical across archs, so the param DIFFERENCE "
              "is purely mixer cost)", flush=True)
        print(f"{'arch':<12} {'params':>12} {'stable':>7} {'val_ppl':>9} {'final_loss':>11}", flush=True)
        for arch in archs:
            r = results[arch]
            print(f"{arch:<12} {r['params']:>12,} {str(r['stable']):>7} "
                  f"{r['val_ppl']:>9.2f} {r['final_loss']:>11.4f}", flush=True)


if __name__ == "__main__":
    main()
