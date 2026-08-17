"""Attribution ablation (canonical, platform-agnostic).

LoRA-only vs MT-v1 vs MT-v2 (each +/- LoRA) on a frozen base, at IDENTICAL
data / step budget / optimizer, reporting held-out WikiText-2 test perplexity.
This is the experiment every MT-LNN claim rests on: does the adapter improve
PPL beyond plain LoRA, and at what parameter cost?

Configs
-------
  baseline    frozen base, no training (reference PPL)
  lora_only   LoRA r=8 on q,k,v,o
  mt_only     v1 MT adapters every 4th layer          (~62.8M on TinyLlama)
  mt_lora     v1 + LoRA (the shipped 003000 recipe)   (~65.1M)
  mt_v2_only  v2 adapters (mt_lnn_v2), FastWeight ON  (~8.4M)
  mt_v2_lora  v2 + LoRA
All see literally the same token tensors in the same order (fixed seed).

Known traps handled here (hard-won, do not remove):
  * enable_input_require_grads(): with a frozen base + gradient checkpointing,
    adapter-only configs otherwise receive ZERO gradients (silent no-learn).
  * PEFT freezes non-LoRA params: MT params are re-armed after get_peft_model.
  * bf16 autocast, no GradScaler (GradScaler rejects bf16).

Usage (any machine with a CUDA GPU):
    python benchmarks/attribution_ablation.py --configs baseline,lora_only
    python benchmarks/attribution_ablation.py --configs all --steps 1000
Writes one JSON per config to --out_dir (skips configs whose JSON exists, so
it is resume-safe and shardable across GPUs/sessions).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

# Allow `python benchmarks/attribution_ablation.py` from anywhere: put the
# repo root (parent of this file's dir) ahead of the script dir on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# IMPORT ORDER MATTERS ON WINDOWS: pyarrow (via datasets) must load its DLLs
# BEFORE transformers' Rust tokenizers, or load_dataset segfaults (exit 139).
# Keep this import at module level even though it looks hoistable into
# build_chunks — that deferred form is exactly what crashed.
import datasets as _datasets  # noqa: F401  (DLL-order guard)


CONFIG_NAMES = ["baseline", "lora_only", "mt_only", "mt_lora",
                "mt_v2_only", "mt_v2_lora", "mt_v2s_only", "mt_v2s_lora",
                "mt_v2_delta"]


def build_chunks(tok, split: str, seq_len: int) -> torch.Tensor:
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = [t for t in ds["text"] if t]
    ids: list = []
    # Batched tokenizer calls: ~50x fewer Python<->Rust crossings than per-row
    # (per-row calls segfault the fast tokenizer on Windows at this volume).
    for i in range(0, len(texts), 1000):
        for row in tok(texts[i: i + 1000], add_special_tokens=False)["input_ids"]:
            ids.extend(row)
            ids.append(tok.eos_token_id)
    n = (len(ids) // seq_len) * seq_len
    return torch.tensor(
        [ids[i: i + seq_len] for i in range(0, n, seq_len)], dtype=torch.long
    )


def make_base(model_name: str, dtype):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    m.config.use_cache = False
    m.gradient_checkpointing_enable()
    # Frozen base + checkpointing => checkpointed segments see no-grad inputs
    # and adapters silently get None gradients. PEFT does this for LoRA;
    # adapter-only configs need it explicitly.
    m.enable_input_require_grads()
    return m


def setup(cfg: str, m, lora_r: int, lora_alpha: int):
    """Attach the adapter under test. Returns (model, n_rearmed)."""
    from mt_lnn.llama_adapter import attach_mt_adapters, iter_mt_adapter_parameters

    def add_lora(model):
        from peft import LoraConfig, get_peft_model
        return get_peft_model(model, LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ))

    def rearm(model, it):
        n = 0
        for p in it(model):
            if not p.requires_grad:
                p.requires_grad = True
            n += p.numel()
        return n

    if cfg == "baseline":
        for p in m.parameters():
            p.requires_grad = False
        return m, 0
    if cfg == "lora_only":
        return add_lora(m), 0
    if cfg == "mt_only":
        attach_mt_adapters(m, every=4, n_protofilaments=13, n_time_scales=5,
                           map_hidden_dim=64, dropout=0.0, init_scale=1e-3,
                           use_scan=True)
        return m, 0
    if cfg == "mt_lora":
        attach_mt_adapters(m, every=4, n_protofilaments=13, n_time_scales=5,
                           map_hidden_dim=64, dropout=0.0, init_scale=1e-3,
                           use_scan=True)
        m = add_lora(m)
        return m, rearm(m, iter_mt_adapter_parameters)
    if cfg in ("mt_v2_only", "mt_v2s_only", "mt_v2_delta"):
        from mt_lnn.mt_lnn_v2 import attach_mt_v2_adapters
        # mt_v2_delta = gradient-as-memory fast-weight write (DeltaNet/Titans);
        # candidate for the distributed-context the outer product can't encode.
        rule = "delta" if cfg == "mt_v2_delta" else "outer"
        attach_mt_v2_adapters(m, every=4, selective_decay=cfg.startswith("mt_v2s"),
                              fast_weight_rule=rule)
        return m, 0
    if cfg in ("mt_v2_lora", "mt_v2s_lora"):
        from mt_lnn.mt_lnn_v2 import (attach_mt_v2_adapters,
                                      iter_mt_v2_adapter_parameters)
        attach_mt_v2_adapters(m, every=4, selective_decay=cfg.startswith("mt_v2s"))
        m = add_lora(m)
        return m, rearm(m, iter_mt_v2_adapter_parameters)
    raise ValueError(f"unknown config: {cfg}")


@torch.no_grad()
def eval_ppl(m, test_chunks, device, dtype, batch: int = 1) -> float:
    m.eval()
    total_nll, total_tok = 0.0, 0
    for i in range(0, len(test_chunks), batch):
        ids = test_chunks[i: i + batch].to(device)
        with torch.amp.autocast("cuda", dtype=dtype, enabled=device == "cuda"):
            out = m(input_ids=ids, labels=ids)
        n = ids.shape[0] * (ids.shape[1] - 1)
        total_nll += out.loss.float().item() * n
        total_tok += n
    return math.exp(total_nll / total_tok)


def run_config(cfg: str, args, tok, train_chunks, test_chunks,
               train_order, device, dtype) -> dict:
    from mt_lnn.llama_adapter import count_trainable_parameters

    torch.manual_seed(args.seed)
    m = make_base(args.model, dtype)
    m, n_rearmed = setup(cfg, m, args.lora_r, args.lora_alpha)
    m.to(device)
    trainable = count_trainable_parameters(m)
    total = sum(p.numel() for p in m.parameters())
    print(f"\n=== {cfg} ===  trainable {trainable:,} / {total:,} "
          f"({100 * trainable / total:.3f}%)  re-armed {n_rearmed:,}", flush=True)

    if cfg == "baseline" or trainable == 0:
        ppl = eval_ppl(m, test_chunks, device, dtype, args.batch)
        print(f"[{cfg}] test PPL (no training): {ppl:.3f}", flush=True)
        del m
        torch.cuda.empty_cache()
        return {"config": cfg, "trainable": trainable, "total": total,
                "final_loss": None, "test_ppl": ppl}

    m.train()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    # T4/P100 have no bf16 -> we train in fp16 autocast, which NEEDS loss
    # scaling or small adapter gradients underflow to zero. (GradScaler
    # rejects bf16, hence the conditional.)
    scaler = (torch.amp.GradScaler("cuda")
              if device == "cuda" and dtype == torch.float16 else None)
    step, last_loss, t0 = 0, float("nan"), time.time()
    opt.zero_grad(set_to_none=True)
    while step < args.steps:
        for idx in train_order:
            if step >= args.steps:
                break
            i = int(idx)
            ids = train_chunks[i: i + 1].to(device)
            with torch.amp.autocast("cuda", dtype=dtype, enabled=device == "cuda"):
                out = m(input_ids=ids, labels=ids)
                loss = out.loss / args.grad_accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            last_loss = out.loss.detach().float().item()
            if (step + 1) % args.grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in m.parameters() if p.requires_grad], args.grad_clip)
                if scaler is not None:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                dt = max(time.time() - t0, 1e-3)
                print(f"[{cfg}] step {step:5d}/{args.steps} | loss {last_loss:.4f}"
                      f" | {args.log_every / dt:.2f} it/s", flush=True)
                t0 = time.time()

    ppl = eval_ppl(m, test_chunks, device, dtype, args.batch)
    print(f"[{cfg}] final train loss {last_loss:.4f} | test PPL {ppl:.3f}",
          flush=True)
    if getattr(args, "save_ckpt", False):
        os.makedirs(args.out_dir, exist_ok=True)
        ck_path = os.path.join(args.out_dir, f"adapter_{cfg}_{args.steps}steps.pt")
        torch.save({
            "config": cfg, "steps": args.steps, "model": args.model,
            "test_ppl": ppl,
            "state_dict": {k: v.cpu() for k, v in m.state_dict().items()
                           if "mt_adapter" in k or "lora_" in k},
        }, ck_path)
        print(f"[{cfg}] saved adapter checkpoint {ck_path}", flush=True)
    del m, opt
    torch.cuda.empty_cache()
    return {"config": cfg, "trainable": trainable, "total": total,
            "final_loss": last_loss, "test_ppl": ppl}


def print_table(results, steps):
    base_ppl = next((r["test_ppl"] for r in results if r["config"] == "baseline"),
                    None)
    print("\n" + "=" * 78, flush=True)
    print(f"ATTRIBUTION ABLATION | WikiText-2 | {steps} steps", flush=True)
    print("=" * 78, flush=True)
    hdr = (f"{'config':<12} {'trainable':>12} {'%params':>8} {'test PPL':>9} "
           f"{'vs base':>8} {'dPPL/1M':>9}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for r in results:
        pct = 100 * r["trainable"] / r["total"]
        if base_ppl and r["trainable"] > 0:
            vs = f"{100 * (r['test_ppl'] - base_ppl) / base_ppl:+.1f}%"
            gpm = f"{(base_ppl - r['test_ppl']) / (r['trainable'] / 1e6):+.2f}"
        else:
            vs, gpm = "--", "--"
        print(f"{r['config']:<12} {r['trainable']:>12,} {pct:>7.3f}% "
              f"{r['test_ppl']:>9.3f} {vs:>8} {gpm:>9}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--configs", default="all",
                    help=f"comma list or 'all' ({','.join(CONFIG_NAMES)})")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--max_eval_chunks", type=int, default=0,
                    help="0 = whole test split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="benchmarks/attribution_out")
    ap.add_argument("--save_ckpt", action="store_true",
                    help="save each trained config's adapter/LoRA tensors to "
                         "out_dir (reusable by length_streaming_eval.py --ckpt)")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    wanted = (CONFIG_NAMES if args.configs.strip() == "all"
              else [c.strip() for c in args.configs.split(",") if c.strip()])
    for c in wanted:
        if c not in CONFIG_NAMES:
            raise SystemExit(f"unknown config '{c}'; choose from {CONFIG_NAMES}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported()
             else (torch.float16 if device == "cuda" else torch.float32))
    print(f"device={device} dtype={dtype} configs={wanted}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("Tokenizing WikiText-2 ...", flush=True)
    train_chunks = build_chunks(tok, "train", args.seq_len)
    test_chunks = build_chunks(tok, "test", args.seq_len)
    if args.max_eval_chunks and args.max_eval_chunks < len(test_chunks):
        test_chunks = test_chunks[: args.max_eval_chunks]
    print(f"train chunks: {len(train_chunks)}  test chunks: {len(test_chunks)}",
          flush=True)

    g = torch.Generator().manual_seed(args.seed)
    train_order = torch.randperm(len(train_chunks), generator=g)

    os.makedirs(args.out_dir, exist_ok=True)
    results = []
    for cfg in wanted:
        path = os.path.join(args.out_dir, f"attr_{args.steps}steps_{cfg}.json")
        if os.path.exists(path):                       # resume-safe sharding
            with open(path) as f:
                results.append(json.load(f))
            print(f"[skip] {cfg}: {path} exists", flush=True)
            continue
        r = run_config(cfg, args, tok, train_chunks, test_chunks,
                       train_order, device, dtype)
        r["meta"] = {"model": args.model, "steps": args.steps,
                     "seq_len": args.seq_len, "grad_accum": args.grad_accum,
                     "lr": args.lr, "lora_r": args.lora_r, "seed": args.seed}
        with open(path, "w") as f:
            json.dump(r, f, indent=2)
        print(f"saved {path}", flush=True)
        results.append(r)

    print_table(results, args.steps)


if __name__ == "__main__":
    main()
