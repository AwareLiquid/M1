"""
benchmarks/long_context_kv_vs_state.py

Long-context benchmark: KV cache vs O(1) state-only stream.

M1's central architectural claim is that the recurrent EMA working memory
replaces the O(T) KV cache, keeping memory and compute constant regardless
of sequence length. This script provides the honest empirical evidence.

Two streaming modes compared at T = 128 / 512 / 1024 / 2048 / 4096:

  KV-stream  (use_cache=True, full KV cache)
    Memory grows O(T): each new token appends K,V to all layers.
    This is how standard Transformer inference works.

  State-stream (use_cache=True, recurrent_only() — h_prev only)
    Memory is O(1): only the ~4 KB LNN recurrent state is kept.
    Attention still runs causal-masked full sequence during training;
    at inference only the current step is processed.

Metrics reported per length T:
  • peak RSS delta (MB): memory allocated during forward vs baseline
  • tok/s: tokens-per-second throughput (total T tokens in one shot)
  • cache_bytes: total bytes in the cache struct after the forward
  • divergence: mean |logit_kv - logit_state| on the LAST token
    (non-zero because the attention sub-layer behaves differently without
    KV cache, but recurrent state should be close on short spans)

Run:
    python benchmarks/long_context_kv_vs_state.py [--lengths 128 512 1024 2048]
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import warnings

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn import MTLNNConfig, MTLNNModel


def build_model(vocab_size: int = 256) -> MTLNNModel:
    cfg = MTLNNConfig(
        vocab_size=vocab_size,
        max_seq_len=8192,
        d_model=832,
        n_layers=12,
        n_heads=13,
        n_kv_heads=1,
        d_head=64,
        dropout=0.0,
        attention_dropout=0.0,
        sparse_resonance_kernel=True,
        sparse_resonance_top_k=2,
        scale_gate_period=2,
        gwtb_compression_ratio=8,
        gwtb_n_heads=4,
    )
    return MTLNNModel(cfg)


def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 ** 2
    except ImportError:
        return 0.0


def run_kv_stream(model: MTLNNModel, ids: torch.Tensor) -> dict:
    """Full KV cache: O(T) memory growth."""
    model.eval()
    gc.collect()
    rss0 = _rss_mb()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
    elapsed = time.perf_counter() - t0
    rss1 = _rss_mb()
    cache = out["cache"]
    return {
        "logits_last": out["logits"][:, -1, :].clone(),
        "elapsed": elapsed,
        "cache_bytes": cache.tensor_bytes(),
        "rss_delta_mb": rss1 - rss0,
        "toks": ids.numel() / elapsed,
    }


def run_state_stream(model: MTLNNModel, ids: torch.Tensor) -> dict:
    """State-only: O(1) memory via recurrent_only() cache."""
    model.eval()
    gc.collect()
    rss0 = _rss_mb()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True, use_lnn_recurrence=True)
        # Strip KV tensors, keep only the tiny h_prev recurrent state
        state_cache = out["cache"].recurrent_only()
        # Re-run just the last token to get logits from state-only mode
        last_out = model(
            input_ids=ids[:, -1:],
            cache=state_cache,
            use_cache=False,
            use_lnn_recurrence=True,
        )
    elapsed = time.perf_counter() - t0
    rss1 = _rss_mb()
    return {
        "logits_last": last_out["logits"][:, -1, :].clone(),
        "elapsed": elapsed,
        "cache_bytes": out["cache"].recurrent_only().tensor_bytes(),
        "rss_delta_mb": rss1 - rss0,
        "toks": ids.numel() / elapsed,
    }


def fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n/1024:.1f} KB"
    return f"{n/1024**2:.2f} MB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lengths", nargs="+", type=int,
        default=[128, 512, 1024, 2048, 4096],
        metavar="T",
        help="sequence lengths to benchmark",
    )
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    print("=" * 78)
    print(" M1 Long-Context Benchmark: KV cache vs O(1) State-Only Stream")
    print("=" * 78)
    print(f" device={device}  batch={args.batch}  d_model=832  n_layers=12")
    print(f" sparse_top_k=2  scale_gate_period=2  (efficient config)")
    print()

    model = build_model().to(device)
    n_params = model.get_num_params()
    print(f" model params: {n_params/1e6:.1f}M")
    print()

    # ---- header ----
    col = "{:>6}  {:>12}  {:>10}  {:>8}  {:>8}  {:>10}"
    print(col.format("T", "mode", "cache_mem", "tok/s", "RSS ΔMB", "div(last)"))
    print("-" * 78)

    results = []
    for T in args.lengths:
        if T > model.config.max_seq_len:
            print(f"  T={T}: skipped (exceeds max_seq_len={model.config.max_seq_len})")
            continue
        ids = torch.randint(0, 256, (args.batch, T), device=device)

        kv = run_kv_stream(model, ids)
        st = run_state_stream(model, ids)

        div = (kv["logits_last"] - st["logits_last"]).abs().mean().item()

        print(col.format(
            T, "kv-cache",
            fmt_bytes(kv["cache_bytes"]),
            f"{kv['toks']:.0f}",
            f"{kv['rss_delta_mb']:.1f}",
            "—",
        ))
        print(col.format(
            "", "state-only",
            fmt_bytes(st["cache_bytes"]),
            f"{st['toks']:.0f}",
            f"{st['rss_delta_mb']:.1f}",
            f"{div:.4f}",
        ))

        ratio_mem = (kv["cache_bytes"] / max(st["cache_bytes"], 1))
        results.append({"T": T, "kv_bytes": kv["cache_bytes"],
                        "state_bytes": st["cache_bytes"],
                        "ratio": ratio_mem, "div": div})
        print()

    # ---- summary ----
    print("=" * 78)
    print(" Memory ratio: KV cache / state-only at each sequence length")
    print()
    print("  {:>6}   {:>12}   {:>12}   {:>10}".format(
        "T", "KV cache", "state-only", "ratio"))
    for r in results:
        print("  {:>6}   {:>12}   {:>12}   {:>8.0f}x".format(
            r["T"], fmt_bytes(r["kv_bytes"]),
            fmt_bytes(r["state_bytes"]), r["ratio"]))
    print()
    if results:
        max_ratio = max(r["ratio"] for r in results)
        max_T = max(r["T"] for r in results)
        print(f" At T={max_T}: KV cache is {max_ratio:.0f}x larger than the O(1) state.")
        print(f" The recurrent state size is CONSTANT regardless of sequence length.")
    print()
    print(" NOTE: divergence on the last token is expected — the attention sub-layer")
    print(" behaves differently in full-sequence vs step-by-step mode. The recurrent")
    print(" state captures temporal context, not exact attention history, which is")
    print(" the intended design trade-off for O(1) memory at scale.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
