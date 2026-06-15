# PC-Liquid-Core continual-learning: EWC vs replay report

Seeds: `[0, 1, 2, 3, 4]` (n=5)  |  EWC lambda=100000.0  |  replay buffer=32 (batch 16)  |  runtime 2267.0s

Forgetting probe: train A -> consolidate A -> train B (EWC and/or replay on) -> re-measure A. Lower `forgetting` = less catastrophic forgetting; `B_after` must stay low or the lever merely blocked learning B.

| variant | #params | A_before | A_after | forgetting | B_after |
|---|---:|---:|---:|---:|---:|
| PC-astro | 14509 | 0.00411 +/- 0.00043 | 1.20640 +/- 0.11301 | +1.20229 +/- 0.11316 | 0.02809 +/- 0.00174 |
| PC-ewc | 14509 | 0.00411 +/- 0.00043 | 0.97024 +/- 0.07239 | +0.96613 +/- 0.07254 | 0.03044 +/- 0.00240 |
| PC-replay | 14509 | 0.00411 +/- 0.00043 | 0.01010 +/- 0.00082 | +0.00598 +/- 0.00102 | 0.03580 +/- 0.00127 |
| PC-ewc-replay | 14509 | 0.00411 +/- 0.00043 | 0.01260 +/- 0.00113 | +0.00849 +/- 0.00118 | 0.04025 +/- 0.00317 |
| GRU-replay | 15401 | 0.00580 +/- 0.00062 | 0.01175 +/- 0.00267 | +0.00595 +/- 0.00256 | 0.05250 +/- 0.00987 |

Paired (per-seed) forgetting deltas (negative = first variant forgets LESS):

| comparison | mean diff | first-lower-in |
|---|---:|---:|
| EWC vs astro-only | -0.23616 | 5/5 |
| replay vs astro-only | -1.19630 | 5/5 |
| replay vs EWC | -0.96014 | 5/5 |
| EWC+replay vs replay | +0.00250 | 0/5 |
| PC-replay vs GRU-replay | +0.00004 | 3/5 |
