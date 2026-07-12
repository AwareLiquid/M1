"""Cross-window associative recall — the O(1)-memory kill-shot experiment.

Protocol
--------
Segment A shows n key->value pairs (random token ids, disjoint key/value
vocab ranges). Then the KV CACHE IS DROPPED — segment B (each key queried
once, shuffled order) runs as a FRESH attention context. The frozen
attention physically cannot see segment A; the ONLY channel from A to B is
the MT adapters' streaming recurrent state (and the fast-weight matrix).

  * baseline / lora_only have no state channel  -> chance accuracy on
    cross-window BY CONSTRUCTION. This is the control.
  * mt_only (v1) carries a decaying multi-timescale EMA state — can it
    hold discrete associations? (Honest expectation: poorly.)
  * mt_v2 carries the fast-weight associative matrix F = sum_i k_i v_i^T —
    the mechanism BUILT for exactly this. mt_v2_nofw ablates it.

Also reports in-window accuracy (single [A|B] forward, attention intact)
so "can recall at all" and "can recall ACROSS the window" separate cleanly.

Training runs the same two-segment protocol with gradients THROUGH the
carried state (stream_in_training + stream_detach=False). Gradient
checkpointing must stay OFF: its backward-time forward replay would re-read
mutated stream state and silently corrupt gradients.

Usage:
    python benchmarks/cross_window_recall.py --configs mt_v2,lora_only \
        --steps 2000 --n_pairs 8 --out_dir benchmarks/recall_out
Writes one JSON per config (resume-safe, shardable), like the attribution
script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

# Same Windows DLL-order guard as attribution_ablation (datasets unused here,
# but transformers+torch order still matters for tokenizers).
from benchmarks.attribution_ablation import setup as attr_setup  # noqa: E402

CONFIG_NAMES = ["baseline", "lora_only", "mt_only", "mt_lora",
                "mt_v2_nofw", "mt_v2", "mt_v2s", "arr", "mt_v2_delta"]


def setup(cfg: str, m, lora_r: int, lora_alpha: int, arr_ckpt: str = ""):
    """attribution_ablation's setup + recall-specific configs.

    'arr' = the O-series attention-free student: convert every layer to a
    recurrent mixer and load distilled mixer weights from --arr_ckpt.
    """
    if cfg == "arr":
        from mt_lnn.arr import convert_to_arr, probe_return_convention
        ck = (torch.load(arr_ckpt, map_location="cpu", weights_only=False)
              if arr_ckpt else None)
        dargs = (ck or {}).get("args", {})
        convert_to_arr(
            m,
            d_proto=int(dargs.get("d_proto", 96)),
            proj_rank=int(dargs.get("proj_rank", 384)),
            fast_weight_dim=int(dargs.get("fw_dim", 96)),
        )
        probe_return_convention(m)
        if ck is not None:
            sd = ck["state_dict"]
            missing, unexpected = m.load_state_dict(sd, strict=False)
            assert not unexpected, f"mixer ckpt mismatch: {unexpected[:4]}"
            print(f"[arr] loaded {len(sd)} distilled mixer tensors", flush=True)
        return m, 0
    if cfg == "mt_v2_nofw":
        from mt_lnn.mt_lnn_v2 import attach_mt_v2_adapters
        attach_mt_v2_adapters(m, every=4, use_fast_weight=False)
        return m, 0
    if cfg == "mt_v2":
        return attr_setup("mt_v2_only", m, lora_r, lora_alpha)
    if cfg == "mt_v2s":
        return attr_setup("mt_v2s_only", m, lora_r, lora_alpha)
    return attr_setup(cfg, m, lora_r, lora_alpha)


def make_batch(bsz: int, n_pairs: int, key_lo: int, key_hi: int,
               val_lo: int, val_hi: int, g: torch.Generator):
    """Returns (seg_a (B, 2n), seg_b (B, 2n), value_pos (2n//2 indices)).

    seg_a: K1 V1 ... Kn Vn.  seg_b: keys re-queried in shuffled order, each
    followed by its value (teacher forcing); loss/accuracy read the logits AT
    the key position (predicting the following value token).
    """
    keys = torch.stack([
        torch.randperm(key_hi - key_lo, generator=g)[:n_pairs] + key_lo
        for _ in range(bsz)
    ])                                                       # (B, n) unique keys
    vals = torch.randint(val_lo, val_hi, (bsz, n_pairs), generator=g)
    seg_a = torch.stack([keys, vals], dim=-1).reshape(bsz, 2 * n_pairs)

    order = torch.stack([torch.randperm(n_pairs, generator=g)
                         for _ in range(bsz)])               # (B, n)
    q_keys = torch.gather(keys, 1, order)
    q_vals = torch.gather(vals, 1, order)
    seg_b = torch.stack([q_keys, q_vals], dim=-1).reshape(bsz, 2 * n_pairs)
    return seg_a, seg_b


def recall_loss_and_acc(logits_b: torch.Tensor, seg_b: torch.Tensor):
    """logits_b: (B, 2n, V) over segment B. Key positions are 0,2,4,...; the
    logit AT a key position predicts the NEXT token (its value)."""
    key_pos = torch.arange(0, seg_b.shape[1], 2, device=seg_b.device)
    pred_logits = logits_b[:, key_pos, :]                    # (B, n, V)
    targets = seg_b[:, key_pos + 1]                          # (B, n)
    loss = F.cross_entropy(
        pred_logits.reshape(-1, pred_logits.shape[-1]), targets.reshape(-1)
    )
    acc = (pred_logits.argmax(-1) == targets).float().mean().item()
    return loss, acc


def set_stream_mode(model, enabled: bool, train_through: bool = False):
    from mt_lnn.arr import set_mixer_streaming
    from mt_lnn.llama_adapter import _iter_all_adapters
    set_mixer_streaming(model, enabled, train_through)   # O-series mixers
    for a in _iter_all_adapters(model):                  # M-series adapters
        a.stream_enabled = enabled
        a.stream_in_training = train_through
        a.stream_detach = not train_through
        a.reset_stream()


def two_segment_forward(model, seg_a, seg_b, reset_fn):
    """Segment A primes the adapters' state; KV cache is NOT carried into
    segment B — attention starts fresh, only adapter state crosses."""
    reset_fn(model)
    model(input_ids=seg_a, use_cache=False)                  # state write only
    out_b = model(input_ids=seg_b, use_cache=False)
    return out_b.logits


@torch.no_grad()
def evaluate(model, args, device, g_eval, reset_fn, n_batches: int = None):
    if n_batches is None:
        n_batches = getattr(args, "eval_batches", 8)
    model.eval()
    accs = {"in_window": [], "cross_window": []}
    for _ in range(n_batches):
        seg_a, seg_b = make_batch(args.batch, args.n_pairs, args.key_lo,
                                  args.key_hi, args.val_lo, args.val_hi, g_eval)
        seg_a, seg_b = seg_a.to(device), seg_b.to(device)
        # In-window: one contiguous sequence (for ARR there is no attention,
        # but the scan covers the whole sequence in one forward).
        reset_fn(model)
        full = model(input_ids=torch.cat([seg_a, seg_b], dim=1),
                     use_cache=False).logits
        _, acc_in = recall_loss_and_acc(full[:, seg_a.shape[1]:, :], seg_b)
        accs["in_window"].append(acc_in)
        # Cross-window: KV dropped between segments.
        logits_b = two_segment_forward(model, seg_a, seg_b, reset_fn)
        _, acc_x = recall_loss_and_acc(logits_b, seg_b)
        accs["cross_window"].append(acc_x)
    return {k: sum(v) / len(v) for k, v in accs.items()}


def run_config(cfg: str, args, device, dtype) -> dict:
    from mt_lnn.llama_adapter import (_iter_all_adapters,
                                      count_trainable_parameters,
                                      reset_adapter_streams)
    from transformers import AutoModelForCausalLM

    torch.manual_seed(args.seed)
    m = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    m.config.use_cache = False
    # NO gradient checkpointing here: its backward-time replay re-reads the
    # MUTATED stream state and corrupts gradients. Sequences are short.
    m, n_rearmed = setup(cfg, m, args.lora_r, args.lora_alpha,
                         arr_ckpt=getattr(args, 'arr_ckpt', ''))
    if args.state_scale_init > 0:
        for a in _iter_all_adapters(m):
            a.scale.data.fill_(args.state_scale_init)
            if getattr(a, "fw_scale", None) is not None:
                a.fw_scale.data.fill_(args.state_scale_init)
    if args.freeze_tau:
        n_frozen = 0
        for name, p in m.named_parameters():
            if name.endswith("log_tau"):
                p.requires_grad = False
                n_frozen += p.numel()
        print(f"[freeze_tau] froze {n_frozen} tau params at bio init", flush=True)
    m.to(device)
    trainable = count_trainable_parameters(m)
    print(f"\n=== {cfg} ===  trainable {trainable:,}  re-armed {n_rearmed:,}",
          flush=True)

    from mt_lnn.arr import MTRecurrentMixer
    has_state = (any(True for _ in _iter_all_adapters(m))
                 or any(isinstance(x, MTRecurrentMixer) for x in m.modules()))

    def reset_fn(model):
        reset_adapter_streams(model)
        from mt_lnn.arr import reset_mixer_streams
        reset_mixer_streams(model)

    g_eval = torch.Generator().manual_seed(10_000 + args.seed)

    if trainable == 0:
        if has_state:
            set_stream_mode(m, True)
        res = evaluate(m, args, device, g_eval, reset_fn)
        print(f"[{cfg}] (no training) in-window {res['in_window']:.3f} | "
              f"cross-window {res['cross_window']:.3f}", flush=True)
        del m
        torch.cuda.empty_cache()
        return {"config": cfg, "trainable": trainable, **res}

    # --- train, alternating protocols 50/50 (grads THROUGH carried state) ---
    # Pure cross-window training is an UNLEARNABLE objective for stateless
    # configs (the answer is not in their input): they degrade toward the
    # value-marginal and their in-window skill is destroyed with nothing
    # gained. Mixing teaches the recall task itself on even steps and the
    # window-crossing variant on odd steps, so the in-window column shows
    # "can it recall at all" and cross-window isolates "beyond attention".
    if has_state:
        set_stream_mode(m, True, train_through=True)
    m.train()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                            lr=args.lr)
    # fp16 (T4/P100 have no bf16) needs loss scaling or the adapters' small
    # gradients underflow to zero. GradScaler rejects bf16, hence conditional.
    scaler = (torch.amp.GradScaler("cuda")
              if device == "cuda" and dtype == torch.float16 else None)
    g_train = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    ema = {"x-win": [None, None], "in-win": [None, None]}   # loss, acc
    for step in range(1, args.steps + 1):
        seg_a, seg_b = make_batch(args.batch, args.n_pairs, args.key_lo,
                                  args.key_hi, args.val_lo, args.val_hi, g_train)
        seg_a, seg_b = seg_a.to(device), seg_b.to(device)
        cross = torch.rand((), generator=g_train).item() < args.cross_frac
        with torch.amp.autocast("cuda", dtype=dtype, enabled=device == "cuda"):
            if cross:
                logits_b = two_segment_forward(m, seg_a, seg_b, reset_fn)
            else:
                reset_fn(m)
                full = m(input_ids=torch.cat([seg_a, seg_b], dim=1),
                         use_cache=False).logits
                logits_b = full[:, seg_a.shape[1]:, :]
            loss, acc = recall_loss_and_acc(logits_b, seg_b)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in m.parameters() if p.requires_grad], 1.0)
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        k = "x-win" if cross else "in-win"
        for i, v in enumerate((loss.item(), acc)):
            ema[k][i] = v if ema[k][i] is None else 0.9 * ema[k][i] + 0.1 * v
        if step % args.log_every == 0:
            dt = max(time.time() - t0, 1e-3)
            parts = []
            for k2 in ("x-win", "in-win"):
                if ema[k2][0] is not None:
                    parts.append(f"{k2} loss {ema[k2][0]:.3f} acc {ema[k2][1]:.3f}")
            print(f"[{cfg}] step {step:5d}/{args.steps} | " + " | ".join(parts)
                  + f" | {args.log_every / dt:.2f} it/s", flush=True)
            t0 = time.time()

    if has_state:
        set_stream_mode(m, True, train_through=False)   # eval: detached stream
    res = evaluate(m, args, device, g_eval, reset_fn)
    print(f"[{cfg}] FINAL in-window {res['in_window']:.3f} | "
          f"cross-window {res['cross_window']:.3f}", flush=True)
    del m, opt
    torch.cuda.empty_cache()
    return {"config": cfg, "trainable": trainable, **res}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--configs", default="all")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n_pairs", type=int, default=8)
    # Higher than the LM-finetune 2e-4: this task builds NEW behaviour from
    # scratch and 2e-4 provably stalls (loss flat ~9.3 for 2000 steps while
    # lr 2e-3 overfits a single batch in 20 steps).
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cross_frac", type=float, default=0.5,
                    help="fraction of training batches using the cross-window "
                         "protocol (in-window saturates early; give the hard "
                         "task the budget)")
    ap.add_argument("--state_scale_init", type=float, default=0.0,
                    help=">0: re-init every adapter's residual gates (scale, "
                         "fw_scale) to this value. The 1e-3 default makes the "
                         "state path's output influence tiny at step 0, so "
                         "the CE gradient bootstraps it very slowly.")
    ap.add_argument("--eval_batches", type=int, default=8)
    ap.add_argument("--freeze_tau", action="store_true",
                    help="bio-prior ablation: keep every mixer/adapter tau "
                         "ladder (log_tau) FROZEN at its biologically-derived "
                         "init instead of training it")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--key_lo", type=int, default=5000)
    ap.add_argument("--key_hi", type=int, default=6000)
    ap.add_argument("--val_lo", type=int, default=7000)
    ap.add_argument("--val_hi", type=int, default=8000)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arr_ckpt", default="",
                    help="distilled mixer checkpoint for the 'arr' config")
    ap.add_argument("--out_dir", default="benchmarks/recall_out")
    args = ap.parse_args()

    wanted = (CONFIG_NAMES if args.configs.strip() == "all"
              else [c.strip() for c in args.configs.split(",") if c.strip()])
    for c in wanted:
        if c not in CONFIG_NAMES:
            raise SystemExit(f"unknown config '{c}'; choose from {CONFIG_NAMES}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported()
             else (torch.float16 if device == "cuda" else torch.float32))
    print(f"device={device} dtype={dtype} n_pairs={args.n_pairs} "
          f"configs={wanted}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    results = []
    for cfg in wanted:
        path = os.path.join(
            args.out_dir,
            f"recall_{args.n_pairs}pairs_{args.steps}steps_{cfg}"
            f"{'_ftau' if args.freeze_tau else ''}_s{args.seed}.json")
        if os.path.exists(path):
            with open(path) as f:
                results.append(json.load(f))
            print(f"[skip] {cfg}: {path} exists", flush=True)
            continue
        r = run_config(cfg, args, device, dtype)
        r["meta"] = vars(args)
        with open(path, "w") as f:
            json.dump(r, f, indent=2)
        results.append(r)

    print("\n" + "=" * 66, flush=True)
    print(f"CROSS-WINDOW RECALL | {args.n_pairs} pairs | {args.steps} steps | "
          f"chance={1.0 / (args.val_hi - args.val_lo):.4f}", flush=True)
    print("=" * 66, flush=True)
    print(f"{'config':<12} {'trainable':>12} {'in-window':>10} {'cross-window':>13}",
          flush=True)
    for r in results:
        print(f"{r['config']:<12} {r['trainable']:>12,} "
              f"{r['in_window']:>10.3f} {r['cross_window']:>13.3f}", flush=True)


if __name__ == "__main__":
    main()
