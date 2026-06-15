# PC-Liquid-Core generative-replay SCARCITY sweep

Seeds: `[0, 1, 2, 3, 4]` (n=5)  |  buffers [4, 8, 16, 32] (warmup 16)  |  runtime 3296.1s

Does PCLiquidCore's generative-replay forgetting edge over a GRU WIDEN as the dream buffer shrinks? Lower `forgetting` is better.

| buffer | PC forgetting | GRU forgetting | PC-GRU diff | PC-lower-in |
|---:|---:|---:|---:|---:|
| 4 | +0.04949 +/- 0.02135 | +0.07612 +/- 0.05779 | -0.02662 | 2/5 |
| 8 | +0.03150 +/- 0.00682 | +0.05643 +/- 0.05472 | -0.02492 | 3/5 |
| 16 | +0.02394 +/- 0.00206 | +0.03900 +/- 0.02372 | -0.01506 | 4/5 |
| 32 | +0.02171 +/- 0.00289 | +0.03588 +/- 0.01979 | -0.01417 | 4/5 |
