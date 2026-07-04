"""Out-of-window memory on real language — roadmap item #2.

Question: when attention is TRUNCATED (KV cache dropped every `chunk`
tokens), does the MT adapters' streaming state recover any of the lost
context on real text (WikiText-2 test), measured in perplexity?

Three protocols over identical token windows:
  full               one forward per window (attention sees everything) —
                     reference upper bound, evaluated at several lengths.
  chunked_stateless  window fed in `chunk`-token pieces, KV dropped between
                     pieces, no adapter state (the old serving behaviour).
  chunked_streaming  same pieces, KV still dropped, but adapter streaming
                     state carries across pieces — the ONLY added channel.

The decisive number is chunked_streaming vs chunked_stateless: identical
scored positions, identical attention blindness; any PPL gap is pure
out-of-window memory. (`full` scores L-1 tokens per window while the
chunked protocols score (C-1)*L/C — noted, not compared directly.)

A model must first be TRAINED with state (or at least trained at all) for
this to be meaningful; use --train_steps to reuse the attribution trainer
inline, or --ckpt to load a saved adapter checkpoint.

Usage:
    python benchmarks/length_streaming_eval.py --config mt_v2_lora \
        --train_steps 1000 --window 2048 --chunk 512
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from benchmarks.attribution_ablation import (  # noqa: E402  (DLL-order safe)
    CONFIG_NAMES,
    build_chunks,
    make_base,
    setup,
)


@torch.no_grad()
def protocol_ppl(model, windows: torch.Tensor, chunk: int, streaming: bool,
                 device, dtype, batch: int = 1) -> float:
    """Mean PPL over `windows` fed in `chunk`-token pieces (KV dropped
    between pieces). streaming=True lets adapter state carry across pieces."""
    from mt_lnn.llama_adapter import (reset_adapter_streams,
                                      set_adapter_streaming)
    set_adapter_streaming(model, streaming)
    model.eval()
    L = windows.shape[1]
    total_nll, total_tok = 0.0, 0
    for i in range(0, len(windows), batch):
        w = windows[i: i + batch].to(device)
        reset_adapter_streams(model)
        for s in range(0, L, chunk):
            ids = w[:, s: s + chunk]
            if ids.shape[1] < 2:
                continue
            with torch.amp.autocast("cuda", dtype=dtype,
                                    enabled=device == "cuda"):
                out = model(input_ids=ids, labels=ids)
            n = ids.shape[0] * (ids.shape[1] - 1)
            total_nll += out.loss.float().item() * n
            total_tok += n
    set_adapter_streaming(model, False)
    return math.exp(total_nll / total_tok)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--config", default="mt_v2_lora", choices=CONFIG_NAMES)
    ap.add_argument("--train_steps", type=int, default=0,
                    help="train the adapter first (attribution trainer, "
                         "seq_len 512); 0 = evaluate as-attached")
    ap.add_argument("--ckpt", default="",
                    help="load a saved adapter checkpoint instead of training")
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--full_lengths", default="512,1024,2048")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--max_windows", type=int, default=0, help="0 = all")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seq_len", type=int, default=512)   # training seq_len
    ap.add_argument("--out_dir", default="benchmarks/length_out")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported()
             else (torch.float16 if device == "cuda" else torch.float32))
    print(f"device={device} dtype={dtype} config={args.config}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- model: train inline (reusing the attribution trainer) or load ckpt
    if args.train_steps > 0:
        train_chunks = build_chunks(tok, "train", args.seq_len)
        g = torch.Generator().manual_seed(args.seed)
        order = torch.randperm(len(train_chunks), generator=g)
        # run_config trains AND evals@512; we reuse its trained model by
        # replicating its training here instead (keep it simple: train via
        # run_config-like loop). To avoid drift we call run_config's pieces:
        torch.manual_seed(args.seed)
        m = make_base(args.model, dtype)
        m, _ = setup(args.config, m, args.lora_r, args.lora_alpha)
        m.to(device)
        import time as _t
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                                lr=args.lr, weight_decay=args.weight_decay)
        scaler = (torch.amp.GradScaler("cuda")
                  if device == "cuda" and dtype == torch.float16 else None)
        m.train()
        step, t0 = 0, _t.time()
        opt.zero_grad(set_to_none=True)
        while step < args.train_steps:
            for idx in order:
                if step >= args.train_steps:
                    break
                ids = train_chunks[int(idx): int(idx) + 1].to(device)
                with torch.amp.autocast("cuda", dtype=dtype,
                                        enabled=device == "cuda"):
                    out = m(input_ids=ids, labels=ids)
                    loss = out.loss / args.grad_accum
                (scaler.scale(loss) if scaler else loss).backward()
                if (step + 1) % args.grad_accum == 0:
                    if scaler:
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in m.parameters() if p.requires_grad],
                        args.grad_clip)
                    if scaler:
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0:
                    dt = max(_t.time() - t0, 1e-3)
                    print(f"[train] {step}/{args.train_steps} loss "
                          f"{out.loss.item():.4f} | {args.log_every/dt:.2f} it/s",
                          flush=True)
                    t0 = _t.time()
        del opt
    else:
        torch.manual_seed(args.seed)
        m = make_base(args.model, dtype)
        m, _ = setup(args.config, m, args.lora_r, args.lora_alpha)
        if args.ckpt:
            sd = torch.load(args.ckpt, map_location="cpu",
                            weights_only=False).get("state_dict", {})
            missing, unexpected = m.load_state_dict(sd, strict=False)
            print(f"[ckpt] loaded {len(sd) - len(unexpected)}/{len(sd)} "
                  f"tensors (unexpected {len(unexpected)})", flush=True)
        m.to(device)
    # Gradient checkpointing must be OFF for streaming eval correctness
    # (make_base enables it for training; disable before stateful evals).
    m.gradient_checkpointing_disable()

    # --- evaluation windows: contiguous test-set stream, length `window`
    test_long = build_chunks(tok, "test", args.window)
    if args.max_windows and args.max_windows < len(test_long):
        test_long = test_long[: args.max_windows]
    print(f"eval windows: {len(test_long)} x {args.window}", flush=True)

    results = {"config": args.config, "window": args.window,
               "chunk": args.chunk, "train_steps": args.train_steps,
               "protocols": {}}

    for L in [int(x) for x in args.full_lengths.split(",") if x]:
        wins = build_chunks(tok, "test", L)
        if args.max_windows:
            wins = wins[: args.max_windows * (args.window // L)]
        ppl = protocol_ppl(m, wins, chunk=L, streaming=False,
                           device=device, dtype=dtype, batch=args.batch)
        results["protocols"][f"full@{L}"] = ppl
        print(f"full@{L:<5}              PPL {ppl:.3f}", flush=True)

    ppl_sl = protocol_ppl(m, test_long, chunk=args.chunk, streaming=False,
                          device=device, dtype=dtype, batch=args.batch)
    results["protocols"]["chunked_stateless"] = ppl_sl
    print(f"chunked_stateless@{args.chunk}   PPL {ppl_sl:.3f}", flush=True)

    ppl_st = protocol_ppl(m, test_long, chunk=args.chunk, streaming=True,
                          device=device, dtype=dtype, batch=args.batch)
    results["protocols"]["chunked_streaming"] = ppl_st
    print(f"chunked_streaming@{args.chunk}   PPL {ppl_st:.3f}", flush=True)

    gap = ppl_sl - ppl_st
    rel = 100 * gap / ppl_sl if ppl_sl else 0.0
    results["out_of_window_gain_ppl"] = gap
    print(f"\nOUT-OF-WINDOW MEMORY GAIN: {gap:+.3f} PPL ({rel:+.2f}%) "
          f"[stateless {ppl_sl:.3f} -> streaming {ppl_st:.3f}]", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(
        args.out_dir,
        f"length_{args.config}_{args.train_steps}steps_w{args.window}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
