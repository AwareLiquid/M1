# KV-compression frontier — exact carried-state byte ledger

ARR measured state: **0.381 MB, flat** (loaded from `benchmarks/results/decode.json`, never hand-copied). Geometry: L=12, d_head=64 (kv-heads measured: 1).

## Full grid — carried MB, all configs x contexts

| config | bits | gqa | window | T=512 | T=2048 | T=8192 | T=32768 | T=131072 | T=524288 | T=1048576 |
|---|---|---|---|---|---|---|---|---|---|---|
| arr_state_measured (O-series) | — | — | — | **0.381** | **0.381** | **0.381** | **0.381** | **0.381** | **0.381** | **0.381** |
| kv_16bit_gqa1 | 16 | 1 | — | 1.5 | 6.0 | 24.0 | 96.0 | 384.0 | 1536.0 | 3072.0 |
| evict_sink4_w512_16bit_gqa1 | 16 | 1 | 512 | 1.5 | 1.512 | 1.512 | 1.512 | 1.512 | 1.512 | 1.512 |
| evict_sink4_w4096_16bit_gqa1 | 16 | 1 | 4096 | 1.5 | 6.0 | 12.012 | 12.012 | 12.012 | 12.012 | 12.012 |
| kv_16bit_gqa8 | 16 | 8 | — | 12.0 | 48.0 | 192.0 | 768.0 | 3072.0 | 12288.0 | 24576.0 |
| evict_sink4_w512_16bit_gqa8 | 16 | 8 | 512 | 12.0 | 12.094 | 12.094 | 12.094 | 12.094 | 12.094 | 12.094 |
| evict_sink4_w4096_16bit_gqa8 | 16 | 8 | 4096 | 12.0 | 48.0 | 96.094 | 96.094 | 96.094 | 96.094 | 96.094 |
| kv_8bit_gqa1 | 8 | 1 | — | 0.763 | 3.048 | 12.189 | 48.751 | 195.001 | 780.001 | 1560.001 |
| evict_sink4_w512_8bit_gqa1 | 8 | 1 | 512 | 0.763 | 0.769 | 0.769 | 0.769 | 0.769 | 0.769 | 0.769 |
| evict_sink4_w4096_8bit_gqa1 | 8 | 1 | 4096 | 0.763 | 3.048 | 6.101 | 6.101 | 6.101 | 6.101 | 6.101 |
| kv_8bit_gqa8 | 8 | 8 | — | 6.105 | 24.387 | 97.512 | 390.012 | 1560.012 | 6240.012 | 12480.012 |
| evict_sink4_w512_8bit_gqa8 | 8 | 8 | 512 | 6.105 | 6.153 | 6.153 | 6.153 | 6.153 | 6.153 | 6.153 |
| evict_sink4_w4096_8bit_gqa8 | 8 | 8 | 4096 | 6.105 | 24.387 | 48.809 | 48.809 | 48.809 | 48.809 | 48.809 |
| kv_4bit_gqa1 | 4 | 1 | — | 0.388 | 1.548 | 6.189 | 24.751 | 99.001 | 396.001 | 792.001 |
| evict_sink4_w512_4bit_gqa1 | 4 | 1 | 512 | 0.388 | 0.391 | 0.391 | 0.391 | 0.391 | 0.391 | 0.391 |
| evict_sink4_w4096_4bit_gqa1 | 4 | 1 | 4096 | 0.388 | 1.548 | 3.098 | 3.098 | 3.098 | 3.098 | 3.098 |
| kv_4bit_gqa8 | 4 | 8 | — | 3.105 | 12.387 | 49.512 | 198.012 | 792.012 | 3168.012 | 6336.012 |
| evict_sink4_w512_4bit_gqa8 | 4 | 8 | 512 | 3.105 | 3.13 | 3.13 | 3.13 | 3.13 | 3.13 | 3.13 |
| evict_sink4_w4096_4bit_gqa8 | 4 | 8 | 4096 | 3.105 | 12.387 | 24.786 | 24.786 | 24.786 | 24.786 | 24.786 |
| kv_2bit_gqa1 | 2 | 1 | — | 0.201 | 0.798 | 3.189 | 12.751 | 51.001 | 204.001 | 408.001 |
| evict_sink4_w512_2bit_gqa1 | 2 | 1 | 512 | 0.201 | 0.202 | 0.202 | 0.202 | 0.202 | 0.202 | 0.202 |
| evict_sink4_w4096_2bit_gqa1 | 2 | 1 | 4096 | 0.201 | 0.798 | 1.597 | 1.597 | 1.597 | 1.597 | 1.597 |
| kv_2bit_gqa8 | 2 | 8 | — | 1.605 | 6.387 | 25.512 | 102.012 | 408.012 | 1632.012 | 3264.012 |
| evict_sink4_w512_2bit_gqa8 | 2 | 8 | 512 | 1.605 | 1.618 | 1.618 | 1.618 | 1.618 | 1.618 | 1.618 |
| evict_sink4_w4096_2bit_gqa8 | 2 | 8 | 4096 | 1.605 | 6.387 | 12.774 | 12.774 | 12.774 | 12.774 | 12.774 |
| hybrid_kv2bit_gqa1 | 2 | 1 | — | 0.582 | 1.179 | 3.57 | 13.132 | 51.382 | 204.382 | 408.382 |
| hybrid_kv2bit_gqa8 | 2 | 8 | — | 1.986 | 6.768 | 25.893 | 102.393 | 408.393 | 1632.393 | 3264.393 |

## Crossover boundaries — where ARR stops being smaller

| config | ARR smaller for | config/ARR @128k | @1M | boundary threat |
|---|---|---|---|---|
| kv_16bit_gqa1 | T >= 131 | 1007.88x | 8063.0x | T* <= 128k |
| evict_sink4_w512_16bit_gqa1 | T >= 131 | 3.97x | 3.97x | T* <= 128k |
| evict_sink4_w4096_16bit_gqa1 | T >= 131 | 31.53x | 31.53x | T* <= 128k |
| kv_16bit_gqa8 | T >= 17 | 8063.0x | 64504.01x | T* <= 128k |
| evict_sink4_w512_16bit_gqa8 | T >= 17 | 31.74x | 31.74x | T* <= 128k |
| evict_sink4_w4096_16bit_gqa8 | T >= 17 | 252.21x | 252.21x | T* <= 128k |
| kv_8bit_gqa1 | T >= 256 | 511.82x | 4094.5x | T* <= 128k |
| evict_sink4_w512_8bit_gqa1 | T >= 256 | 2.02x | 2.02x | T* <= 128k |
| evict_sink4_w4096_8bit_gqa1 | T >= 256 | 16.01x | 16.01x | T* <= 128k |
| kv_8bit_gqa8 | T >= 32 | 4094.52x | 32755.97x | T* <= 128k |
| evict_sink4_w512_8bit_gqa8 | T >= 32 | 16.15x | 16.15x | T* <= 128k |
| evict_sink4_w4096_8bit_gqa8 | T >= 32 | 128.11x | 128.11x | T* <= 128k |
| kv_4bit_gqa1 | T >= 503 | 259.85x | 2078.75x | T* <= 128k |
| evict_sink4_w512_4bit_gqa1 | T >= 503 | 1.03x | 1.03x | T* <= 128k |
| evict_sink4_w4096_4bit_gqa1 | T >= 503 | 8.13x | 8.13x | T* <= 128k |
| kv_4bit_gqa8 | T >= 62 | 2078.77x | 16629.97x | T* <= 128k |
| evict_sink4_w512_4bit_gqa8 | T >= 62 | 8.21x | 8.21x | T* <= 128k |
| evict_sink4_w4096_4bit_gqa8 | T >= 62 | 65.05x | 65.05x | T* <= 128k |
| kv_2bit_gqa1 | T >= 976 | 133.86x | 1070.87x | T* <= 128k |
| evict_sink4_w512_2bit_gqa1 | never | 0.53x | 0.53x | never exceeds ARR |
| evict_sink4_w4096_2bit_gqa1 | T >= 976 | 4.19x | 4.19x | T* <= 128k |
| kv_2bit_gqa8 | T >= 119 | 1070.9x | 8566.97x | T* <= 128k |
| evict_sink4_w512_2bit_gqa8 | T >= 119 | 4.25x | 4.25x | T* <= 128k |
| evict_sink4_w4096_2bit_gqa8 | T >= 119 | 33.53x | 33.53x | T* <= 128k |

Footnotes — the byte account is not the capability account:

1. Eviction rows that sit BELOW the ARR line win on carried bytes at every T,
   but pay in retrieval: attention is structurally 0.000 on cross-window
   recall while the fast-weight state scores 0.56 (RESULTS.md, "Cross-window
   associative recall"). A token evicted past the window is gone; a binding
   written into the state is not.
2. Hybrid rows apply the same audit to ourselves: the M-series hybrid's KV
   quantizes like anyone else's, so it is charged kv@2bit + the measured
   liquid stream, never the fp16 KV the old tables implied.
3. Every opponent number here is a byte-exact analytic carried-state floor,
   NOT a quality claim: 2-bit KV needs KIVI's asymmetric scheme to stay
   usable, eviction needs a sink, and ARR's own quality beyond 512 tokens is
   unproven (RESULTS.md out-of-window nulls).
