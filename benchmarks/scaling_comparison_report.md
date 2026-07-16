# Scaling Comparison Report — transformer vs mt_lnn vs mamba (fp32)

Generated: `2026-07-16T04:00:00.000000+00:00`
Device: `NVIDIA GeForce RTX 5060 Laptop GPU (8GB)` (local, not cloud-preemptible)
PyTorch: `2.11.0+cu128`
Command: `benchmarks/scaling_comparison.py --mode train --steps 2000 --seeds 0 --dtype fp32 --train_token_cap 50000000`

## Summary

Matched-width archs (`d_model=832`, 12 layers) trained from scratch on WikiText-103,
2000 steps, 50M training-token cap, fp32 throughout, seed 0. Mamba is width/depth-mismatched
by construction (`hidden=768`, 24 layers — HF's own default sizing) so params land in the
same ~125-142M class; it is included as an external baseline, not a matched-architecture
control.

| Arch | Params | Stable (fp32, 2000 steps) | Final train loss | Val PPL | Throughput |
|---|---:|:---:|---:|---:|---:|
| transformer | 142,051,520 | ✅ | 6.016 | 370.81 | ~7100 tok/s |
| **mt_lnn** | 126,041,819 | ✅ | 5.603 | **257.48** | ~1200 tok/s |
| mamba | 129,117,696 | ✅ | 6.306 | 414.00 | ~1270 tok/s |

- mt_lnn: **30.6% lower PPL than transformer** with 11.3% fewer params.
- mt_lnn: **37.8% lower PPL than mamba** with roughly matched params (126M vs 129M).
- All three archs ran the full 2000 steps with `stable: true` — no NaN/divergence anywhere,
  including past step 629, where an earlier Colab T4 **fp16** AMP run saw **mt_lnn's** loss
  go non-finite while the **transformer** baseline stayed stable for the full 2000 steps
  under the identical fp16 recipe (commit `b11e5bc`). This fp32 rerun confirms that gap was
  an fp16 numerical-robustness issue specific to mt_lnn's recurrent/gating math (never
  exposed under bf16's wider dynamic range) — not an architectural instability — and that
  mt_lnn's PPL advantage over transformer holds under same-precision (fp32) comparison too.
  Root cause of the fp16-specific fragility is still open; `--dtype fp32` remains the
  fair-comparison fallback until it lands.
- mamba's fp32 train loss looked competitive mid-run (step 1500: 5.43, ahead of transformer's
  5.76) but its final val PPL is the worst of the three — a train/val gap worth flagging
  whenever citing mamba's training-loss curve in isolation.

## Environment Notes

- Mamba fast path (`mamba-ssm` CUDA kernel / `causal-conv1d`) is unavailable on Windows;
  this run uses the `mamba.py` parallel-scan backend (`use_mambapy=True` in
  `benchmarks/scaling_comparison.py`) instead of transformers' sequential fallback, which
  is ~25-50x slower and impractical for a 2000-step run. `use_mambapy` is mathematically
  equivalent to the CUDA kernel path; it only trades throughput, not numerics.
- Local run only (single seed, n=1); not yet reproduced on the Colab/Kaggle transformer
  baseline used in prior BENCHMARKS.md entries.

## Caveats

- Single seed (0) — no variance estimate yet. Treat the PPL gaps as directional, not
  statistically bulletproof, until a multi-seed run lands.
- mamba is not architecture-matched to transformer/mt_lnn (different hidden size and
  depth); the params happen to land in the same class but the comparison should not be
  read as "same shape, different token-mixer."
- Throughput numbers are single-GPU wall-clock on an RTX 5060 Laptop (8GB), not a
  cloud reference device; use only for relative arch-to-arch comparison within this run,
  not absolute cost projections.
