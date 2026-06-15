# PC-Liquid-Core A->B->C generative-replay continual learning

Seeds: `[0, 1, 2, 3, 4]` (n=5)  |  chained dream buffer=16 (warmup 16)  |  runtime 1123.1s

Deep Generative Replay (scholar): train A; dream D1(~A); train B rehearsing D1; dream D2(~A+B); train C rehearsing D2. Forgetting is measured vs each task's error right after it was learned. `worst_final` = max held-out 1-step MSE across A,B,C = the 'is the whole system still usable' number (lower=better).

| kind | forget A | forget B | mean forget | A final | B final | C final | WORST task |
|---|---:|---:|---:|---:|---:|---:|---:|
| PC | +0.03146 | +0.00411 | +0.01778 +/- 0.00367 | 0.03557 | 0.04040 | 0.01286 | 0.04234 +/- 0.00400 (max 0.04849) |
| GRU | +0.01636 | +0.00071 | +0.00853 +/- 0.00258 | 0.02216 | 0.05039 | 0.01528 | 0.05039 +/- 0.00515 (max 0.05384) |

**Paired PC-GRU mean forgetting:** diff +0.00925, PC lower in 0/5.

**Paired PC-GRU WORST-task final MSE:** diff -0.00805, PC lower in 5/5.  (worst-task is the usability number; large positive GRU worst = collapse.)
