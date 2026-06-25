"""
benchmarks/cross_layer_gate_sharing_ablation.py

Cross-layer scale-gate sharing ablation (IndexShare analog).

Tests three configurations on a standard 12-layer MT-LNN:
  period=1  (baseline): every layer computes its own kappa_gate (current default)
  period=2  : leader every 2 layers
  period=4  : leader every 4 layers (mirrors GLM-5.2 IndexShare grouping)

Reports: tok/s, mean step time, divergence from baseline output.
Only meaningful when sparse_resonance_kernel=True (gate sharing + compute skip).

Run:
    python benchmarks/cross_layer_gate_sharing_ablation.py
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn import MTLNNConfig, MTLNNModel


def make_config(period: int, top_k: int = 2) -> MTLNNConfig:
    return MTLNNConfig(
        vocab_size=256,
        max_seq_len=512,
        d_model=832,
        n_layers=12,
        n_heads=13,
        n_kv_heads=1,
        d_head=64,
        dropout=0.0,
        attention_dropout=0.0,
        sparse_resonance_kernel=True,
        sparse_resonance_top_k=top_k,
        dynamic_scale_gates=True,
        scale_gate_period=period,
        gwtb_compression_ratio=8,
        gwtb_n_heads=4,
    )


def benchmark(model: MTLNNModel, ids: torch.Tensor, n_warmup: int = 3, n_bench: int = 10):
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            out = model(input_ids=ids)

        times = []
        for _ in range(n_bench):
            t0 = time.perf_counter()
            out = model(input_ids=ids)
            times.append(time.perf_counter() - t0)

    B, T = ids.shape
    tok_per_step = B * T
    mean_t = sum(times) / len(times)
    min_t = min(times)
    toks = tok_per_step / mean_t
    return toks, mean_t, min_t, out["logits"]


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, T = 2, 128
    ids = torch.randint(0, 256, (B, T), device=device)
    top_k = 2

    print("=" * 68)
    print(" Cross-Layer Scale-Gate Sharing Ablation (IndexShare analog)")
    print("=" * 68)
    print(f" device={device}  B={B}  T={T}  sparse_top_k={top_k}  n_layers=12")
    print(" (same model weights, only scale_gate_period varies)")
    print()

    # Use a SINGLE model — switch config.scale_gate_period between runs so
    # logit divergence reflects only the gate-sharing effect, not weight noise.
    cfg = make_config(period=1, top_k=top_k)
    model = MTLNNModel(cfg).to(device)

    results = {}
    for period in [1, 2, 4]:
        model.config.scale_gate_period = period
        toks, mean_t, min_t, logits = benchmark(model, ids)
        results[period] = {"toks": toks, "mean_t": mean_t, "min_t": min_t,
                           "logits": logits.detach()}
        label = "baseline (no sharing)" if period == 1 else f"share every {period} layers"
        print(f"  period={period}  {label:30s}  {toks:7.1f} tok/s  "
              f"mean={mean_t*1000:.2f}ms  min={min_t*1000:.2f}ms")

    print()
    base_logits = results[1]["logits"]
    base_toks = results[1]["toks"]
    for period in [2, 4]:
        diff = (results[period]["logits"] - base_logits).abs()
        mean_div = diff.mean().item()
        max_div = diff.max().item()
        speedup = (results[period]["toks"] / base_toks - 1) * 100
        print(f"  period={period}  speedup={speedup:+.1f}%  "
              f"mean_logit_div={mean_div:.5f}  max_logit_div={max_div:.4f}")

    print()
    print(" Interpretation:")
    print("  speedup  = throughput gain from skipping top-k re-selection")
    print("  logit_div = output change (reflects τ-scale routing stability)")
    print("  Small div (<0.05) means follower layers are insensitive to τ-routing.")
    print("=" * 68)


if __name__ == "__main__":
    main()
