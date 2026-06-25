# MT-LNN Operator Compression Report

Generated: `2026-06-08T09:18:13.486097+00:00`
Device: `cpu`
PyTorch: `2.5.1+cu121`

## Summary

State-only streaming preserves recurrent `h_prev` while dropping historical KV tensors. This is compressed memory, not lossless transcript recall.

| Steps | Mode | Tok/s | Final cache | Peak cache | Mean div vs full | Max div vs full |
|---:|---|---:|---:|---:|---:|---:|
| 100 | `full_sequence_oracle` | 4838.2 | 0 B | 0 B | 0 | 0 |
| 100 | `kv_cache_stream` | 228.6 | 106.4 KB | 106.4 KB | 6.6e-07 | 4.11e-06 |
| 100 | `state_only_stream` | 229.2 | 4.1 KB | 4.1 KB | 0.0887 | 0.453 |
| 100 | `prefix_replay_stream` | 88.0 | 0 B | 0 B | 3.67e-07 | 4.4e-06 |
| 1000 | `full_sequence_oracle` | 6473.1 | 0 B | 0 B | 0 | 0 |
| 1000 | `kv_cache_stream` | 212.0 | 1020.5 KB | 1020.5 KB | 5.45e-07 | 3.61e-06 |
| 1000 | `state_only_stream` | 230.0 | 4.1 KB | 4.1 KB | 0.0892 | 0.523 |
| 1000 | `prefix_replay_stream` | skipped | - | - | - | - |

## Scale-Gate Diagnostics

| Steps | Gate mean | Active ratio | Nonzero ratio | Sparse selected ratio | Per-scale means |
|---:|---:|---:|---:|---:|---|
| 100 | 0.881 | 1.000 | 1.000 | 1.000 | s0_mean=0.881, s1_mean=0.881, s2_mean=0.881, s3_mean=0.881, s4_mean=0.881 |
| 1000 | 0.881 | 1.000 | 1.000 | 1.000 | s0_mean=0.881, s1_mean=0.881, s2_mean=0.881, s3_mean=0.881, s4_mean=0.881 |

## Caveats

- KV streaming remains the exact incremental path; its divergence from the full-sequence oracle should stay near numerical noise.
- State-only streaming is intentionally lossy relative to full attention because it keeps only recurrent state.
- Current scale-gate masking affects the blend weights and diagnostics. It is not yet a custom sparse kernel that avoids the upstream resonance matrix multiply.
- Optional `--sparse_resonance_kernel` skips inactive tau-scale projection/scan work, but top-k scale selection is chunk-dependent and can change KV parity relative to dense full-sequence execution.
- `benchmarks/state_only_streaming.py --metrics_jsonl <path>` emits structured JSONL metric events for observability plumbing.
- Use `requirements.lock` to reproduce the local smoke benchmark environment; `requirements.txt` intentionally keeps flexible research-time lower bounds.
- SQLite session persistence is local-first and should not be treated as a distributed high-QPS state store.
- Claims in external material should report measured cache size and latency separately from future compute-skipping hypotheses.
