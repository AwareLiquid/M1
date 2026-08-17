# PC-Liquid-Core 5-task generative-replay continual learning (deep chain)

Seeds: `[0, 1, 2, 3, 4]` (n=5)  |  5 tasks  |  chained dream buffer=16 (warmup 16)  |  runtime 3248.2s

Deep Generative Replay (scholar) over 5 well-separated timescale regimes (slow->fast). After each task the current model dreams a buffer of all tasks seen so far; the next task rehearses it. Forgetting is measured vs each task's error right after it was learned. `worst_final` = max held-out 1-step MSE across all tasks = the 'is the whole system still usable' number (lower=better).

| kind | mean forget | WORST task final MSE |
|---|---:|---:|
| PC | +0.55152 +/- 0.12021 | 0.84554 +/- 0.20465 (max 1.13768) |
| GRU | +0.95647 +/- 0.57620 | 1.38276 +/- 0.73942 (max 2.77216) |

**Paired PC-GRU mean forgetting:** diff -0.40495, PC lower in 5/5.

**Paired PC-GRU WORST-task final MSE:** diff -0.53722, PC lower in 5/5.  (worst-task is the usability number; large positive GRU worst = collapse.)

Per-seed worst-task index (which timescale regime is the system's weakest link):
- PC : [0, 0, 1, 0, 0]
- GRU: [0, 0, 0, 0, 0]
