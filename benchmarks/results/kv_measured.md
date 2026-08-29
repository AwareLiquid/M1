# KV ledger — measured validation on real models

Device: `mps` — fp16 scale/group + zero-point at data width

fp16 rows must match the formula exactly; floor vs g32 bounds the metadata the ledger's floor omits (real KIVI sits between them; the excess enlarges the opponent, so published ARR advantages are lower bounds).

## TinyLlama/TinyLlama-1.1B-Chat-v1.0

L=22, n_kv=4, head_dim=64, dtype=fp16

| T | fp16 real | formula | dev | 2-bit floor | 2-bit g32 | g32 extra |
|---|---|---|---|---|---|---|
| 512 | 11,534,336 | 11,534,336 | 0.0% | 1,593,856 | 1,982,464 | +24.38% |
| 2048 | 46,137,344 | 46,137,344 | 0.0% | 6,324,736 | 7,929,856 | +25.38% |
| 8192 | 184,549,376 | 184,549,376 | 0.0% | 25,248,256 | 31,719,424 | +25.63% |

Eviction @T=8192, keep=4100: fp16 92,364,800 B · 2-bit 12,644,896 B

## unsloth/Llama-3.2-1B

L=16, n_kv=8, head_dim=64, dtype=fp16

| T | fp16 real | formula | dev | 2-bit floor | 2-bit g32 | g32 extra |
|---|---|---|---|---|---|---|
| 512 | 16,777,216 | 16,777,216 | 0.0% | 2,318,336 | 2,883,584 | +24.38% |
| 2048 | 67,108,864 | 67,108,864 | 0.0% | 9,199,616 | 11,534,336 | +25.38% |
| 8192 | 268,435,456 | 268,435,456 | 0.0% | 36,724,736 | 46,137,344 | +25.63% |

Eviction @T=8192, keep=4100: fp16 134,348,800 B · 2-bit 18,392,576 B
