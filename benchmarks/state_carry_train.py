"""State-carry (TBPTT) LM training — make the liquid state earn its keep.

The length_streaming_eval null result showed WHY standard training leaves the
state empty across chunks: windowed LM training never rewards carrying
information over a KV boundary. This trainer creates that pressure directly:

  * Each training example is a WINDOW of `window` tokens split into
    `chunk`-token pieces.
  * Pieces are fed sequentially with the KV cache DROPPED between them
    (attention cannot see previous pieces) but adapter streaming state
    CARRIED — with gradients THROUGH the carried state
    (stream_in_training + stream_detach=False), so the write pathway learns
    from the read pathway's success on later pieces.
  * Loss: mean CE over all pieces. Piece 1 is ordinary LM; pieces 2+ are
    only improvable through the state channel.

--carry_state false trains the identical chunked protocol WITHOUT state
(the control: same data truncation, no channel).

Evaluation reuses length_streaming_eval's protocols; the headline is
chunked_streaming PPL (state-carry-trained) vs chunked_stateless, against
the full@2048 upper bound. Gradient checkpointing stays OFF (mutable
stream state + backward replay is a silent corruption).

Usage:
    python benchmarks/state_carry_train.py --config mt_v2s_lora \
        --steps 1000 --window 1024 --chunk 512
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

from benchmarks.attribution_ablation import (  # noqa: E402 (DLL-order safe)
    CONFIG_NAMES,
    build_chunks,
    setup,
)
from benchmarks.length_streaming_eval import protocol_ppl  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--config", default="mt_v2s_lora", choices=CONFIG_NAMES)
    ap.add_argument("--carry_state", default="true", choices=["true", "false"])
    ap.add_argument("--steps", type=int, default=1000,
                    help="window-steps (each = window/chunk chunk-forwards)")
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    # Between the LM 2e-4 and the from-scratch-behaviour 1e-3 (recall lesson).
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--state_scale_init", type=float, default=0.1,
                    help="recall lesson: 1e-3 residual gates bootstrap the "
                         "state path too slowly; 0 leaves checkpoint init")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_window", type=int, default=2048)
    ap.add_argument("--max_eval_windows", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="benchmarks/state_carry_out")
    args = ap.parse_args()
    carry = args.carry_state == "true"

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mt_lnn.llama_adapter import (_iter_all_adapters,
                                      count_trainable_parameters,
                                      reset_adapter_streams)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported()
             else (torch.float16 if device == "cuda" else torch.float32))
    print(f"device={device} dtype={dtype} config={args.config} carry={carry}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("Tokenizing WikiText-2 ...", flush=True)
    train_wins = build_chunks(tok, "train", args.window)
    g = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(train_wins), generator=g)
    print(f"train windows: {len(train_wins)} x {args.window}", flush=True)

    torch.manual_seed(args.seed)
    m = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    m.config.use_cache = False
    # NO gradient checkpointing: backward-time replay re-reads mutated stream
    # state. Window activations at batch 1 fit without it.
    m, _ = setup(args.config, m, args.lora_r, args.lora_alpha)
    adapters = list(_iter_all_adapters(m))
    if args.state_scale_init > 0:
        for a in adapters:
            a.scale.data.fill_(args.state_scale_init)
            if getattr(a, "fw_scale", None) is not None:
                a.fw_scale.data.fill_(args.state_scale_init)
    if carry:
        for a in adapters:
            a.stream_enabled = True
            a.stream_in_training = True
            a.stream_detach = False
            a.reset_stream()
    m.to(device)
    trainable = count_trainable_parameters(m)
    print(f"trainable {trainable:,} | adapters {len(adapters)}", flush=True)

    m.train()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                            lr=args.lr)
    scaler = (torch.amp.GradScaler("cuda")
              if device == "cuda" and dtype == torch.float16 else None)
    step, t0, ema = 0, time.time(), None
    opt.zero_grad(set_to_none=True)
    while step < args.steps:
        for idx in order:
            if step >= args.steps:
                break
            w = train_wins[int(idx): int(idx) + args.batch].to(device)
            reset_adapter_streams(m)
            losses = []
            with torch.amp.autocast("cuda", dtype=dtype,
                                    enabled=device == "cuda"):
                for s in range(0, args.window, args.chunk):
                    ids = w[:, s: s + args.chunk]
                    out = m(input_ids=ids, labels=ids)
                    losses.append(out.loss)
                loss = torch.stack(losses).mean() / args.grad_accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            cur = loss.item() * args.grad_accum
            ema = cur if ema is None else 0.95 * ema + 0.05 * cur
            if (step + 1) % args.grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in m.parameters() if p.requires_grad],
                    args.grad_clip)
                if scaler is not None:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                dt = max(time.time() - t0, 1e-3)
                print(f"[{args.config}|carry={carry}] step {step:5d}/"
                      f"{args.steps} | loss(ema) {ema:.4f} | "
                      f"{args.log_every / dt:.2f} it/s", flush=True)
                t0 = time.time()

    # --- evaluation: reuse the length_streaming protocols -------------------
    for a in adapters:
        a.stream_in_training = False
        a.stream_detach = True
        a.reset_stream()
    test_long = build_chunks(tok, "test", args.eval_window)
    if args.max_eval_windows and args.max_eval_windows < len(test_long):
        test_long = test_long[: args.max_eval_windows]
    res = {"config": args.config, "carry_state": carry, "steps": args.steps,
           "window": args.window, "chunk": args.chunk, "lr": args.lr,
           "trainable": trainable, "protocols": {}}
    for L in sorted({args.chunk, args.eval_window}):
        wins = build_chunks(tok, "test", L)
        if args.max_eval_windows:
            wins = wins[: max(1, args.max_eval_windows
                              * max(1, args.eval_window // L))]
        ppl = protocol_ppl(m, wins, chunk=L, streaming=False,
                           device=device, dtype=dtype, batch=args.batch)
        res["protocols"][f"full@{L}"] = ppl
        print(f"full@{L:<5}              PPL {ppl:.3f}", flush=True)
    ppl_sl = protocol_ppl(m, test_long, chunk=args.chunk, streaming=False,
                          device=device, dtype=dtype, batch=args.batch)
    res["protocols"]["chunked_stateless"] = ppl_sl
    print(f"chunked_stateless@{args.chunk}   PPL {ppl_sl:.3f}", flush=True)
    ppl_st = protocol_ppl(m, test_long, chunk=args.chunk, streaming=True,
                          device=device, dtype=dtype, batch=args.batch)
    res["protocols"]["chunked_streaming"] = ppl_st
    print(f"chunked_streaming@{args.chunk}   PPL {ppl_st:.3f}", flush=True)
    gain = ppl_sl - ppl_st
    res["out_of_window_gain_ppl"] = gain
    print(f"\nOUT-OF-WINDOW GAIN after state-carry training: {gain:+.3f} PPL "
          f"[stateless {ppl_sl:.3f} -> streaming {ppl_st:.3f}]", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir,
                        f"carry_{args.config}_{args.carry_state}_{args.steps}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
