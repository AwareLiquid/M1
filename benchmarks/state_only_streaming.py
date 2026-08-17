"""Benchmark normal KV streaming against state-only recurrent streaming.

The benchmark uses synthetic token streams so it does not need a tokenizer or
dataset. It reports:

- latency for a full-sequence oracle forward
- latency and peak cache bytes for incremental KV cache
- latency and peak cache bytes for state-only recurrent cache
- logit divergence of each streaming path against the full-sequence oracle

Example:
    python benchmarks/state_only_streaming.py --steps 100 1000
"""

import argparse
import dataclasses
import json
import os
import sys
import time
import warnings
from typing import Dict, List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn import MTLNNConfig, MTLNNModel
from mt_lnn.observability import JsonlMetricWriter
from mt_lnn.streaming import streaming_inference


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _elapsed(device: str, fn):
    _sync(device)
    start = time.perf_counter()
    result = fn()
    _sync(device)
    return result, time.perf_counter() - start


def build_config(args, max_steps: int) -> MTLNNConfig:
    # By default the RoPE/mask window is grown to fit the whole stream so every
    # path stays inside its tables. With --fixed_window the window is pinned at
    # --max_seq_len instead, so state-only streaming runs in the *real* O(1)
    # regime (T >> window, wrapping the RoPE table) -- the regime that actually
    # exercises the constant-memory claim.
    if args.fixed_window:
        max_seq_len = args.max_seq_len
    else:
        max_seq_len = max(args.max_seq_len, max_steps + 1)
    return MTLNNConfig(
        vocab_size=args.vocab_size,
        max_seq_len=max_seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        d_head=args.d_model // args.n_heads,
        dropout=0.0,
        attention_dropout=0.0,
        gwtb_compression_ratio=args.gwtb_compression_ratio,
        gwtb_n_heads=args.gwtb_n_heads,
        coherence_heads=args.coherence_heads,
        dynamic_scale_gates=not args.disable_dynamic_scale_gates,
        scale_gate_init_bias=args.scale_gate_init_bias,
        scale_gate_active_threshold=args.scale_gate_active_threshold,
        scale_gate_skip_threshold=args.scale_gate_skip_threshold,
        compute_skip_threshold=args.compute_skip_threshold,
        sparse_resonance_kernel=args.sparse_resonance_kernel,
        sparse_resonance_top_k=args.sparse_resonance_top_k,
    )


@torch.no_grad()
def full_sequence_oracle(model: MTLNNModel, tokens: torch.Tensor) -> torch.Tensor:
    out = model(tokens, use_lnn_recurrence=True)
    return out["logits"].detach().cpu()


@torch.no_grad()
def kv_stream(model: MTLNNModel, tokens: torch.Tensor) -> Dict:
    cache = None
    logits = []
    cache_bytes = []
    for t in range(tokens.shape[1]):
        out = model(
            tokens[:, t:t + 1],
            cache=cache,
            use_cache=True,
            use_lnn_recurrence=True,
        )
        cache = out["cache"]
        logits.append(out["logits"].detach().cpu())
        cache_bytes.append(cache.tensor_bytes())

    return {
        "logits": torch.cat(logits, dim=1),
        "final_cache_bytes": cache_bytes[-1],
        "peak_cache_bytes": max(cache_bytes),
        "cache_bytes_curve": cache_bytes,
    }


@torch.no_grad()
def state_only_stream(model: MTLNNModel, tokens: torch.Tensor) -> Dict:
    cache = None
    logits = []
    cache_bytes = []
    for t in range(tokens.shape[1]):
        step_logits, cache = streaming_inference(
            model,
            tokens[:, t:t + 1],
            cache,
            state_only=True,
        )
        logits.append(step_logits.detach().cpu())
        cache_bytes.append(cache.tensor_bytes())

    return {
        "logits": torch.cat(logits, dim=1),
        "final_cache_bytes": cache_bytes[-1],
        "peak_cache_bytes": max(cache_bytes),
        "cache_bytes_curve": cache_bytes,
    }


@torch.no_grad()
def prefix_replay_stream(
    model: MTLNNModel,
    tokens: torch.Tensor,
    max_steps: int,
) -> Optional[Dict]:
    if tokens.shape[1] > max_steps:
        return None

    logits = []
    processed_tokens = 0
    for t in range(tokens.shape[1]):
        prefix = tokens[:, :t + 1]
        out = model(prefix, use_lnn_recurrence=True)
        logits.append(out["logits"][:, -1:, :].detach().cpu())
        processed_tokens += prefix.numel()

    return {
        "logits": torch.cat(logits, dim=1),
        "processed_tokens": processed_tokens,
        "final_cache_bytes": 0,
        "peak_cache_bytes": 0,
    }


def divergence(reference: torch.Tensor, candidate: torch.Tensor) -> Dict[str, float]:
    diff = (reference - candidate).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def summarize_mode(
    name: str,
    elapsed_s: float,
    steps: int,
    cache_bytes: int,
    peak_cache_bytes: int,
    div: Optional[Dict[str, float]] = None,
    extra: Optional[Dict] = None,
) -> Dict:
    row = {
        "mode": name,
        "elapsed_s": elapsed_s,
        "tokens_per_s": steps / max(elapsed_s, 1e-9),
        "final_cache_bytes": cache_bytes,
        "peak_cache_bytes": peak_cache_bytes,
    }
    if div is not None:
        row["divergence_vs_full_sequence"] = div
    if extra:
        row.update(extra)
    return row


def run_one(model: MTLNNModel, tokens: torch.Tensor, args, device: str) -> Dict:
    steps = tokens.shape[1]

    oracle_logits, oracle_time = _elapsed(
        device, lambda: full_sequence_oracle(model, tokens)
    )

    kv_result, kv_time = _elapsed(device, lambda: kv_stream(model, tokens))
    state_result, state_time = _elapsed(device, lambda: state_only_stream(model, tokens))

    replay_result, replay_time = _elapsed(
        device,
        lambda: prefix_replay_stream(model, tokens, args.max_prefix_replay_steps),
    )

    modes: List[Dict] = [
        summarize_mode(
            "full_sequence_oracle",
            oracle_time,
            steps,
            cache_bytes=0,
            peak_cache_bytes=0,
            extra={"note": "single full causal forward; used as logit reference"},
        ),
        summarize_mode(
            "kv_cache_stream",
            kv_time,
            steps,
            cache_bytes=kv_result["final_cache_bytes"],
            peak_cache_bytes=kv_result["peak_cache_bytes"],
            div=divergence(oracle_logits, kv_result["logits"]),
        ),
        summarize_mode(
            "state_only_stream",
            state_time,
            steps,
            cache_bytes=state_result["final_cache_bytes"],
            peak_cache_bytes=state_result["peak_cache_bytes"],
            div=divergence(oracle_logits, state_result["logits"]),
            extra={
                "note": "drops historical KV; preserves recurrent h_prev only",
                "cache_constant": len(set(state_result["cache_bytes_curve"][1:])) <= 1
                if steps > 1 else True,
            },
        ),
    ]

    if replay_result is None:
        modes.append({
            "mode": "prefix_replay_stream",
            "skipped": True,
            "reason": (
                f"steps={steps} exceeds --max_prefix_replay_steps="
                f"{args.max_prefix_replay_steps}"
            ),
        })
    else:
        modes.append(
            summarize_mode(
                "prefix_replay_stream",
                replay_time,
                steps,
                cache_bytes=0,
                peak_cache_bytes=0,
                div=divergence(oracle_logits, replay_result["logits"]),
                extra={"processed_tokens": replay_result["processed_tokens"]},
            )
        )

    diagnostics = model.get_mt_diagnostics()
    gate_keys = {
        key: value for key, value in diagnostics.items()
        if key.startswith("scale_gate") or key.startswith("sparse_resonance")
    }

    return {
        "steps": steps,
        "batch": tokens.shape[0],
        "scale_gate_diagnostics": gate_keys,
        "modes": modes,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, nargs="+", default=[100],
                        help="One or more stream lengths, e.g. --steps 100 1000")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", default="benchmarks/state_only_streaming_results.json")

    parser.add_argument("--vocab_size", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--fixed_window", action="store_true",
                        help="Pin the RoPE/mask window at --max_seq_len instead of "
                             "growing it to fit the stream; lets state-only streaming "
                             "run the true O(1) regime at T >> window.")
    parser.add_argument("--d_model", type=int, default=104)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_kv_heads", type=int, default=2)
    parser.add_argument("--gwtb_compression_ratio", type=int, default=4)
    parser.add_argument("--gwtb_n_heads", type=int, default=2)
    parser.add_argument("--coherence_heads", type=int, default=2)
    parser.add_argument("--max_prefix_replay_steps", type=int, default=128)
    parser.add_argument("--disable_dynamic_scale_gates", action="store_true")
    parser.add_argument("--scale_gate_init_bias", type=float, default=2.0)
    parser.add_argument("--scale_gate_active_threshold", type=float, default=0.5)
    parser.add_argument("--scale_gate_skip_threshold", type=float, default=0.0,
                        help="Mask scale gates below this value in the blend only")
    parser.add_argument("--compute_skip_threshold", type=float, default=0.0,
                        help="Legacy alias for blend-level scale masking")
    parser.add_argument("--sparse_resonance_kernel", action="store_true",
                        help="Compute only top-k tau scales selected by gate means")
    parser.add_argument("--sparse_resonance_top_k", type=int, default=1)
    parser.add_argument("--metrics_jsonl", default=None,
                        help="Optional JSONL path for per-mode structured metric events")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "auto":
        device = "cpu"

    max_steps = max(args.steps)
    cfg = build_config(args, max_steps=max_steps)
    model = MTLNNModel(cfg).to(device).eval()

    results = {
        "benchmark": "state_only_streaming",
        "device": device,
        "torch": torch.__version__,
        "config": dataclasses.asdict(cfg),
        "runs": [],
    }

    print("MT-LNN state-only streaming benchmark")
    print(f"device={device} steps={args.steps} batch={args.batch}")
    print()

    metrics = None
    if args.metrics_jsonl:
        metrics = JsonlMetricWriter(
            args.metrics_jsonl,
            static_fields={
                "benchmark": "state_only_streaming",
                "device": device,
                "torch": torch.__version__,
            },
        )

    try:
        for steps in args.steps:
            tokens = torch.randint(0, cfg.vocab_size, (args.batch, steps), device=device)
            run = run_one(model, tokens, args, device)
            results["runs"].append(run)

            print(f"steps={steps}")
            gate_diag = run.get("scale_gate_diagnostics", {})
            if gate_diag:
                active = gate_diag.get("scale_gate_active_ratio")
                nonzero = gate_diag.get("scale_gate_nonzero_ratio")
                mean = gate_diag.get("scale_gate_mean")
                print(
                    f"  scale gates: mean={mean:.3f} "
                    f"active_ratio={active:.3f} nonzero_ratio={nonzero:.3f}"
                )
                sparse_ratio = gate_diag.get("sparse_resonance_scale_ratio")
                if sparse_ratio is not None:
                    print(f"  sparse resonance: selected_scale_ratio={sparse_ratio:.3f}")
                if metrics is not None:
                    metrics.write("scale_gate_diagnostics", {"steps": steps, **gate_diag})
            for row in run["modes"]:
                if metrics is not None:
                    metrics.write("stream_mode", {"steps": steps, **row})
                if row.get("skipped"):
                    print(f"  {row['mode']}: skipped ({row['reason']})")
                    continue
                div = row.get("divergence_vs_full_sequence")
                div_text = ""
                if div is not None:
                    div_text = f" div.max={div['max_abs']:.3e} div.mean={div['mean_abs']:.3e}"
                print(
                    f"  {row['mode']}: {row['elapsed_s']:.4f}s "
                    f"{row['tokens_per_s']:.1f} tok/s "
                    f"cache={row['final_cache_bytes']}B peak={row['peak_cache_bytes']}B"
                    f"{div_text}"
                )
            print()
    finally:
        if metrics is not None:
            metrics.close()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
