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

ARCHS = ["transformer", "modern_transformer", "lnn", "mt_lnn", "mamba",
         "mt_lnn_mtp"]


# Mamba config, set from CLI in main() so build() (which only sees d_model/
# n_layers, the SHARED width for the matched-width archs) can size Mamba to its
# own natural ~130M shape independently. Default = standard Mamba-130m.
_MAMBA = {"hidden": 768, "layers": 24}

# MTP ablation knobs, set from CLI in main(). The "mt_lnn_mtp" arch is IDENTICAL
# to "mt_lnn" except it turns on the multi-token-prediction lookahead heads + aux
# loss. Because the MTP heads are registered LAST in MTLNNModel.__init__ (after
# the whole trunk + lm_head), the shared trunk's init RNG draws are unchanged, so
# at a matched seed the two variants start from a byte-identical trunk and differ
# ONLY by the MTP aux gradient — a clean controlled A/B for "does MTP-as-
# regularizer lower val PPL on the proven core?".
_MTP = {"k": 3, "weight": 0.1}


def _checkpoint_path(args, arch, seed):
    return os.path.join(args.out_dir, "checkpoints", f"train_{arch}_s{seed}.pt")


def _save_checkpoint(path, arch, seed, m, opt, scaler, step, cursor, last,
                     stable):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "arch": arch,
        "seed": seed,
        "step": step,
        "cursor": cursor,
        "last": last,
        "stable": stable,
        "model": m.state_dict(),
        "optim": opt.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng_all": (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None),
    }
    tmp = f"{path}.tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def _load_checkpoint(path, m, opt, scaler, device):
    ckpt = torch.load(path, map_location=device)
    m.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optim"])
    if scaler and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if ckpt.get("torch_rng") is not None:
        torch.set_rng_state(ckpt["torch_rng"].cpu())
    if device == "cuda" and ckpt.get("cuda_rng_all") is not None:
        cuda_rng_all = [s.detach().cpu() for s in ckpt["cuda_rng_all"]]
        if len(cuda_rng_all) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_rng_all)
        else:
            print("  [checkpoint] CUDA RNG device count mismatch; "
                  "skipping CUDA RNG restore", flush=True)
    return ckpt


def count_params(m):
    """Uniform total-parameter count across archs — the models' own
    get_num_params() disagree (baselines exclude embeddings, MTLNNModel
    includes them), which would make the comparison apples-to-oranges."""
    return sum(p.numel() for p in m.parameters())


def build(arch, d_model, n_layers, vocab, seq_len, device, dtype,
           selective_decay=False, signed_decay=False):
    from benchmarks.baselines import (BaselineConfig, SimpleCausalLNN,
                                       SimpleCausalTransformer,
                                       ModernCausalTransformer)

    if arch == "mamba":
        # The reviewer's named SSM baseline. Standard Mamba-130m config
        # (matched ~scale, not matched width — Mamba's layer is ~half a
        # transformer layer, so it uses 2x depth). HF falls back to a correct
        # sequential impl when mamba-ssm's CUDA kernel is absent (slower, but
        # the PPL comparison is unaffected).
        from transformers import MambaConfig, MambaForCausalLM
        cfg = MambaConfig(vocab_size=vocab, hidden_size=_MAMBA["hidden"],
                          num_hidden_layers=_MAMBA["layers"], state_size=16,
                          # mamba.py pscan backend: parallel-scan in pure
                          # PyTorch — needed on Windows where mamba-ssm's CUDA
                          # kernel can't build; sequential fallback is ~50x
                          # slower and unusable for 2000-step runs.
                          use_mambapy=True)
        return MambaForCausalLM(cfg).to(device=device, dtype=dtype)
    if arch in ("transformer", "modern_transformer", "lnn"):
        # SwiGLU uses three projections (gate/up/down). An 8/3 expansion keeps
        # its FFN parameter budget close to a 4x GELU FFN while giving the
        # baseline the modern RoPE/RMSNorm/SwiGLU recipe reviewers expect.
        d_ff = int(round((8 * d_model / 3) / 256) * 256)
        cfg = BaselineConfig(vocab_size=vocab, max_seq_len=seq_len,
                             d_model=d_model, n_layers=n_layers,
                             n_heads=13,
                             d_ff=(d_ff if arch == "modern_transformer"
                                   else 4 * d_model),
                             dropout=0.0)
        if arch == "transformer":
            m = SimpleCausalTransformer(cfg)
        elif arch == "modern_transformer":
            m = ModernCausalTransformer(cfg)
        else:
            m = SimpleCausalLNN(cfg)
    else:
        # "mt_lnn" = lean core; "mt_lnn_mtp" = lean core + MTP regularizer (the
        # only difference is the aux heads/loss — see _MTP note above).
        mtp_on = arch == "mt_lnn_mtp"
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
            # MTP ablation: the aux heads only when arch == "mt_lnn_mtp".
            use_mtp_heads=mtp_on,
            mtp_lookahead=_MTP["k"],
            mtp_loss_weight=(_MTP["weight"] if mtp_on else 0.0),
            # Parity-capable transition parameterisations (2026-08-05): the
            # liquid core's escape route from TC^0 — input-dependent λ_t.
            # Defaults OFF = bit-identical to every historical run.
            selective_decay=selective_decay,
            signed_decay=signed_decay,
        )
        m = MTLNNModel(cfg)
    return m.to(device=device, dtype=dtype)


def decode_state_profile(args, device, dtype):
    """The REAL O(1) test: bytes of CARRIED STATE needed to continue generation
    vs context length. An attention model must keep a KV cache that grows O(T);
    an attention-free recurrent model (ARR) keeps a fixed (F,z)+h state, O(1).

    llama  : matched-size HF Llama — KV-cache bytes are exact/analytic
             (2 * n_layers * n_kv_heads * d_head * T * dtype_bytes).
    arr    : convert_to_arr of the same Llama — state measured EMPIRICALLY by
             priming T tokens (streaming, O(1) memory) and summing the snapshot
             tensors, to prove the state size does NOT grow with T.

    This is the honest home of the O(1) claim — NOT training-time memory (see
    --mode profile, where MT-LNN has no advantage because full-sequence
    forward+backward materialises the whole scan)."""
    from transformers import LlamaConfig, LlamaForCausalLM

    from mt_lnn.arr import convert_to_arr
    from mt_lnn.llama_adapter import (reset_adapter_streams,
                                      set_adapter_streaming,
                                      snapshot_adapter_streams)

    n_kv, d_head, L = 1, args.d_model // 13, args.n_layers   # GQA=1 (native's choice)
    bytes_per = 2                                             # bf16/fp16
    lens = [int(x) for x in args.profile_lens.split(",")]

    def _snap_bytes(snap):
        tot = 0
        for k, e in snap.items():
            if k == "_schema":
                continue
            for t in ([e["h"]] + (e["fw"] or [])):
                if t is not None:
                    tot += t.numel() * t.element_size()
        return tot

    torch.manual_seed(0)
    lcfg = LlamaConfig(vocab_size=args.vocab, hidden_size=args.d_model,
                       num_hidden_layers=L, num_attention_heads=13,
                       num_key_value_heads=n_kv,
                       intermediate_size=4 * args.d_model,
                       max_position_embeddings=max(lens) + 8)
    arr = LlamaForCausalLM(lcfg)
    convert_to_arr(arr, d_proto=args.d_model // 13, fast_weight_dim=64)
    # Move AFTER conversion: convert_to_arr creates the mixers fresh (on CPU),
    # so the whole model must be placed on the device once they exist.
    arr = arr.to(device=device, dtype=dtype)
    arr.eval()                        # streaming is gated off in train() mode
    set_adapter_streaming(arr, True)

    rows = []
    for T in lens:
        kv_mb = 2 * L * n_kv * d_head * T * bytes_per / 2**20   # analytic, exact
        reset_adapter_streams(arr)
        with torch.no_grad():                                   # prime in chunks: O(1) mem
            for s in range(0, T, 512):
                arr(input_ids=torch.randint(0, args.vocab,
                    (1, min(512, T - s)), device=device), use_cache=False)
        arr_mb = _snap_bytes(snapshot_adapter_streams(arr)) / 2**20
        rows.append({"seq_len": T, "llama_kv_mb": round(kv_mb, 3),
                     "arr_state_mb": round(arr_mb, 3)})
        print(f"  T={T:6d} | llama KV {kv_mb:8.3f} MB (O(T)) | "
              f"arr state {arr_mb:7.3f} MB (O(1))", flush=True)
    return {"rows": rows, "n_layers": L, "n_kv_heads": n_kv, "d_head": d_head}


def profile_arch(arch, args, device, dtype):
    """TRAINING memory + throughput vs sequence length (full fwd+bwd). NOTE:
    this is NOT the O(1) test — the parallel scan materialises the whole
    sequence, so MT-LNN has no memory advantage here (see --mode decode)."""
    rows = []
    for T in [int(x) for x in args.profile_lens.split(",")]:
        try:
            m = build(arch, args.d_model, args.n_layers, args.vocab, T, device, dtype,
                      selective_decay=args.selective_decay,
                      signed_decay=args.signed_decay)
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


def build_chunks(tok, split, seq_len, wikitext="wikitext-103-raw-v1",
                 max_tokens=None):
    from datasets import load_dataset

    import numpy as np

    # datasets>=3 removed script-based datasets; the canonical hub id is now
    # "Salesforce/wikitext" and the bare "wikitext" name only resolves from a
    # pre-existing local cache (fresh machines - e.g. Kaggle kernels - crash
    # with HfUriError). Try the canonical id first, fall back for old caches.
    try:
        ds = load_dataset("Salesforce/wikitext", wikitext, split=split)
    except Exception:
        ds = load_dataset("wikitext", wikitext, split=split)
    texts = [t for t in ds["text"] if t]
    # Stream token ids into int32 numpy buffers instead of one giant Python
    # list: on the full WikiText-103 train split the old code built ~120M
    # boxed ints (>3 GB) and then torch.tensor() re-walked a nested list of
    # slices — on a 13 GB Kaggle VM that thrashed for hours producing no
    # output (the 2026-07-15 E1 timeout). Buffers keep it a few hundred MB.
    # max_tokens (optional) caps the split — 2000-step runs consume only
    # ~4M tokens, so a cap slashes tokenization time without changing the
    # comparison (every arch sees the identical capped corpus).
    bufs, total = [], 0
    eos = tok.eos_token_id
    t0 = time.time()
    for i in range(0, len(texts), 1000):
        flat = []
        for row in tok(texts[i:i + 1000])["input_ids"]:
            flat.extend(row)
            flat.append(eos)
        bufs.append(np.asarray(flat, dtype=np.int32))
        total += len(flat)
        if max_tokens is not None and total >= max_tokens:
            break
        if i and i % 100_000 == 0:
            print(f"    [tokenize {split}] {i}/{len(texts)} rows, "
                  f"{total / 1e6:.1f}M tokens, {time.time() - t0:.0f}s",
                  flush=True)
    ids = np.concatenate(bufs) if bufs else np.zeros(0, dtype=np.int32)
    if max_tokens is not None:
        ids = ids[:max_tokens]
    n = (len(ids) // seq_len) * seq_len
    print(f"    [tokenize {split}] done: {len(ids) / 1e6:.1f}M tokens -> "
          f"{n // seq_len} chunks of {seq_len} ({time.time() - t0:.0f}s)",
          flush=True)
    return torch.from_numpy(ids[:n].astype(np.int64)).reshape(-1, seq_len)


def train_arch(arch, args, device, dtype, seed=0):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    train_c = build_chunks(tok, "train", args.seq_len, args.wikitext,
                           max_tokens=args.train_token_cap)
    test_c = build_chunks(tok, "test", args.seq_len, args.wikitext)
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train_c), generator=g)

    torch.manual_seed(seed)   # reproducible weight init per seed
    # AMP recipe: with fp16 COMPUTE the parameters must stay fp32 master
    # weights — GradScaler.unscale_ refuses fp16 gradients outright
    # (ValueError at step 1), so a full fp16 model cast can never train.
    # autocast below still runs the matmuls in fp16. bf16 keeps the
    # historical full-cast, scaler-free path.
    param_dtype = torch.float32 if dtype == torch.float16 else dtype
    m = build(arch, args.d_model, args.n_layers, args.vocab, args.seq_len, device, param_dtype,
              selective_decay=args.selective_decay, signed_decay=args.signed_decay)
    if arch == "mt_lnn_mtp":
        # Controlled A/B: the extra MTP-head Linear params draw init RNG DURING
        # MTLNNModel.__init__ *before* its self.apply(init_weights) pass, which
        # shifts the RNG and perturbs the whole trunk's init (verified maxdiff
        # ~0.13 vs plain mt_lnn). Overwrite the shared trunk (params + buffers)
        # with a same-seed plain mt_lnn so the two variants start from a byte-
        # identical trunk and differ ONLY by the freshly-init MTP heads + the aux
        # gradient — otherwise a PPL delta could be init luck, not MTP.
        torch.manual_seed(seed)
        _base = build("mt_lnn", args.d_model, args.n_layers, args.vocab,
                      args.seq_len, device, param_dtype,
                      selective_decay=args.selective_decay,
                      signed_decay=args.signed_decay)
        _bsd = _base.state_dict()
        _msd = m.state_dict()
        m.load_state_dict({k: (_bsd[k] if k in _bsd else v)
                           for k, v in _msd.items()}, strict=True)
        del _base
    n_params = count_params(m)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, betas=(0.9, 0.95))
    scaler = (torch.amp.GradScaler("cuda")
              if device == "cuda" and dtype == torch.float16 else None)
    m.train()
    stable, last, t0, step, cursor = True, float("nan"), time.time(), 0, 0
    ckpt_path = _checkpoint_path(args, arch, seed)
    if args.resume and os.path.exists(ckpt_path):
        ckpt = _load_checkpoint(ckpt_path, m, opt, scaler, device)
        step = int(ckpt.get("step", 0))
        cursor = int(ckpt.get("cursor", 0))
        last = float(ckpt.get("last", float("nan")))
        stable = bool(ckpt.get("stable", True))
        print(f"  [{arch}] resumed checkpoint: step {step}/{args.steps}, "
              f"cursor {cursor}", flush=True)
    while step < args.steps:
        # Fancy-index permuted rows into REAL shuffled minibatches. Slicing
        # train_c[idx:idx+batch] would draw `batch` temporally-ADJACENT corpus
        # windows (and overlapping start indices share batch-1 rows) — the
        # comparison stays fair (same for every arch) but the sampling is
        # autocorrelated, unlike standard shuffled-minibatch SGD.
        # Gradient accumulation (--grad_accum N): N micro-batches of size
        # --batch per optimizer step, loss scaled by 1/N. With batch=1,
        # grad_accum=4 an optimizer step consumes the same 4 permuted rows as
        # batch=4 — same data, same effective batch, ~1/4 the peak activation
        # memory. Needed for HF Mamba's sequential fallback, whose TRAINING
        # memory at B=4 T=512 exceeds a 15 GB T4 (it is an inference path).
        accum = max(1, args.grad_accum)
        micro = 0
        opt.zero_grad(set_to_none=True)
        if cursor >= len(order):
            cursor = 0
        for b in range(cursor, len(order), args.batch):
            if step >= args.steps:
                break
            sel = order[b:b + args.batch]
            ids = train_c[sel].to(device)
            if ids.shape[0] < 1:
                continue
            with torch.amp.autocast("cuda", dtype=dtype,
                                    enabled=device == "cuda" and dtype != torch.float32):
                out = m(ids, labels=ids)
                loss = out["loss"] / accum
            if not torch.isfinite(loss):
                stable = False
                print(f"  [{arch}] NON-FINITE loss at step {step} — UNSTABLE", flush=True)
                break
            (scaler.scale(loss).backward() if scaler else loss.backward())
            micro += 1
            if micro < accum:
                continue
            micro = 0
            if scaler:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            (scaler.step(opt), scaler.update()) if scaler else opt.step()
            opt.zero_grad(set_to_none=True)
            last = loss.item() * accum
            step += 1
            cursor = b + args.batch
            if step % args.log_every == 0:
                dt = max(time.time() - t0, 1e-3)
                print(f"  [{arch:11s}] {step}/{args.steps} loss {last:.4f} "
                      f"| {args.log_every*args.batch*accum*args.seq_len/dt:.0f} tok/s", flush=True)
                t0 = time.time()
            if args.ckpt_every and step % args.ckpt_every == 0:
                _save_checkpoint(ckpt_path, arch, seed, m, opt, scaler, step,
                                 cursor, last, stable)
                print(f"  [{arch}] checkpoint saved at step {step}: "
                      f"{ckpt_path}", flush=True)
        if not stable:
            break
        cursor = 0

    if args.ckpt_every and stable:
        _save_checkpoint(ckpt_path, arch, seed, m, opt, scaler, step, cursor,
                         last, stable)
        print(f"  [{arch}] checkpoint saved at step {step}: {ckpt_path}",
              flush=True)

    # held-out PPL
    m.eval()
    nll, ntok = 0.0, 0
    with torch.no_grad():
        for i in range(0, min(len(test_c), args.eval_chunks or len(test_c)), args.batch):
            ids = test_c[i:i + args.batch].to(device)
            with torch.amp.autocast("cuda", dtype=dtype,
                                    enabled=device == "cuda" and dtype != torch.float32):
                out = m(ids, labels=ids)
            n = ids.shape[0] * (ids.shape[1] - 1)
            # PPL from the PURE next-token CE (lm_loss), never the training
            # objective (out["loss"] folds in the MTP aux term for mt_lnn_mtp).
            # Baselines/Mamba have no "lm_loss" key → fall back to out["loss"],
            # which for them IS the pure CE.
            ce = out.get("lm_loss", out["loss"])
            nll += ce.float().item() * n
            ntok += n
    # Guard exp() for exactly the divergence this benchmark exists to measure:
    # a diverged-but-finite model (mean CE > ~709) would raise OverflowError
    # and abort the whole multi-seed sweep; a NaN would poison statistics.mean.
    # Map both to a reported inf (a large-but-reported value) instead.
    mean_nll = nll / ntok if ntok else float("nan")
    ppl = (math.exp(mean_nll) if math.isfinite(mean_nll) and mean_nll < 709.0
           else float("inf"))
    del m, opt
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"arch": arch, "seed": seed, "params": n_params, "stable": stable,
            "final_loss": last, "val_ppl": ppl, "steps": step}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["profile", "train", "decode"],
                    default="profile")
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
    ap.add_argument("--seeds", default="0,1,2",
                    help="train mode: comma seeds for error bars")
    ap.add_argument("--mamba_hidden", type=int, default=768)
    ap.add_argument("--mamba_layers", type=int, default=24)
    ap.add_argument("--mtp_k", type=int, default=3,
                    help="mt_lnn_mtp: MTP lookahead K (default 3)")
    ap.add_argument("--mtp_weight", type=float, default=0.1,
                    help="mt_lnn_mtp: MTP aux-loss weight λ (default 0.1)")
    ap.add_argument("--selective_decay", action="store_true",
                    help="mt_lnn: input-dependent transition λ_t = "
                         "decay·tanh(W_sel·x_t+b) — the parity-capable "
                         "parameterisation (2026-08-05 probe result). OFF by "
                         "default = bit-identical to historical runs")
    ap.add_argument("--signed_decay", action="store_true",
                    help="mt_lnn: negative-eigenvalue extension λ = "
                         "decay·tanh(s) (Grazzi ICLR 2025). Superseded by "
                         "--selective_decay when both are set")
    ap.add_argument("--wikitext", default="wikitext-103-raw-v1",
                    help="wikitext-2-raw-v1 for a cheap smoke")
    ap.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"],
                    default="auto",
                    help="override autocast dtype (fp32 disables autocast/"
                         "scaler entirely)")
    ap.add_argument("--grad_accum", type=int, default=1,
                    help="micro-batches per optimizer step (batch*accum = "
                         "effective batch; use batch=1 accum=4 to fit HF "
                         "Mamba's memory-hungry sequential TRAINING path)")
    ap.add_argument("--train_token_cap", type=int, default=None,
                    help="cap the tokenized TRAIN split at N tokens (identical "
                         "for every arch; 2000-step runs consume ~4M, so 50M "
                         "keeps sampling diversity while cutting tokenization "
                         "cost ~15x on constrained VMs). None = full split.")
    ap.add_argument("--ckpt_every", type=int, default=500,
                    help="train mode: save model/optimizer/step checkpoint "
                         "every N optimizer steps. 0 disables checkpoints.")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="train mode: resume from an existing checkpoint in "
                         "out_dir/checkpoints when present.")
    ap.add_argument("--profile_lens", default="512,1024,2048,4096")
    ap.add_argument("--profile_batch", type=int, default=2)
    ap.add_argument("--profile_iters", type=int, default=3)
    ap.add_argument("--out_dir", default="benchmarks/scaling_out")
    args = ap.parse_args()

    _MAMBA["hidden"], _MAMBA["layers"] = args.mamba_hidden, args.mamba_layers
    _MTP["k"], _MTP["weight"] = args.mtp_k, args.mtp_weight
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bf16 only where it has HARDWARE support (sm_80+). torch>=2.3's
    # is_bf16_supported() returns True on older GPUs via EMULATION, which on a
    # P100 (sm_60) is orders of magnitude slower than fp16 — the 2026-07-15 E1
    # run burned its whole 8h budget without reaching the first logged step.
    # --dtype overrides: fp32 exists because MT-LNN diverged (non-finite loss,
    # step 629) under fp16 AMP on a T4 while the Transformer baseline trained
    # clean — a REAL fp16-robustness gap, recorded as a finding. Until it is
    # root-caused, same-precision fp32 runs are the fair-comparison fallback.
    if args.dtype != "auto":
        dtype = {"fp32": torch.float32, "fp16": torch.float16,
                 "bf16": torch.bfloat16}[args.dtype]
    else:
        _bf16_hw = device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8
        dtype = (torch.bfloat16 if _bf16_hw
                 else (torch.float16 if device == "cuda" else torch.float32))
    archs = ARCHS if args.archs == "all" else [a for a in args.archs.split(",") if a in ARCHS]
    print(f"device={device} dtype={dtype} mode={args.mode} "
          f"d_model={args.d_model} n_layers={args.n_layers} archs={archs}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == "decode":
        print("\n=== decode: carried-state bytes vs context (the O(1) test) ===",
              flush=True)
        res = decode_state_profile(args, device, dtype)
        json.dump(res, open(os.path.join(args.out_dir, "decode.json"), "w"), indent=2)
        print("\n" + "=" * 66, flush=True)
        print("CARRIED STATE vs CONTEXT | attention KV-cache O(T) vs ARR state O(1)",
              flush=True)
        print(f"{'context T':>10} {'llama KV (MB)':>15} {'ARR state (MB)':>16} "
              f"{'ratio':>8}", flush=True)
        for r in res["rows"]:
            ratio = r["llama_kv_mb"] / max(r["arr_state_mb"], 1e-9)
            print(f"{r['seq_len']:>10} {r['llama_kv_mb']:>15.3f} "
                  f"{r['arr_state_mb']:>16.3f} {ratio:>7.1f}x", flush=True)
        print("\nARR state should be FLAT across T (O(1)); llama KV grows linearly.",
              flush=True)
        return

    if args.mode == "profile":
        results = {}
        for arch in archs:
            path = os.path.join(args.out_dir, f"profile_{arch}.json")
            if os.path.exists(path):
                results[arch] = json.load(open(path)); print(f"[skip] {arch}", flush=True); continue
            print(f"\n=== {arch} (profile) ===", flush=True)
            results[arch] = profile_arch(arch, args, device, dtype)
            json.dump(results[arch], open(path, "w"), indent=2)
        print("\n" + "=" * 66, flush=True)
        print(f"SCALING PROFILE | peak MB (lower+flatter = better) | "
              f"d_model={args.d_model} x {args.n_layers}L", flush=True)
        lens = [int(x) for x in args.profile_lens.split(",")]
        print("arch         " + "".join(f"T={T:<10}" for T in lens), flush=True)
        for arch in archs:
            cells = {row["seq_len"]: row for row in results[arch]}
            line = f"{arch:<12} "
            for T in lens:
                r = cells.get(T, {})
                line += (f"{r['peak_mb']:>5.0f}MB " if "peak_mb" in r
                         else f"{r.get('error','-'):>7} ")[:11]
            print(line, flush=True)
        return

    # --- train mode: multi-seed, mean±std per arch (resume-safe per arch/seed) ---
    seeds = [int(s) for s in str(args.seeds).split(",") if s != ""]
    runs = {arch: [] for arch in archs}
    for arch in archs:
        for seed in seeds:
            path = os.path.join(args.out_dir, f"train_{arch}_s{seed}.json")
            if os.path.exists(path):
                runs[arch].append(json.load(open(path)))
                print(f"[skip] {arch} seed {seed}", flush=True); continue
            print(f"\n=== {arch} (train, seed {seed}) ===", flush=True)
            r = train_arch(arch, args, device, dtype, seed=seed)
            json.dump(r, open(path, "w"), indent=2)
            runs[arch].append(r)

    import statistics
    print("\n" + "=" * 72, flush=True)
    print(f"SCALING TRAIN | WikiText-103 | {args.steps} steps | seeds {seeds}", flush=True)
    print(f"{'arch':<12} {'params':>12} {'stable':>7} {'val_ppl (mean±std)':>22} {'n':>3}", flush=True)
    for arch in archs:
        rs = runs[arch]
        ppls = [x["val_ppl"] for x in rs]
        mean = statistics.mean(ppls)
        std = statistics.stdev(ppls) if len(ppls) > 1 else 0.0
        stable = all(x["stable"] for x in rs)
        params = rs[0]["params"]
        print(f"{arch:<12} {params:>12,} {str(stable):>7} "
              f"{mean:>11.2f} ± {std:<7.2f} {len(ppls):>3}", flush=True)


if __name__ == "__main__":
    main()
