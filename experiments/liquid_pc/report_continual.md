# PC-Liquid-Core continual-learning (calcium-weighted EWC) report

Seeds: `[0, 1, 2, 3, 4]` (n=5)  |  EWC lambda=100000.0  |  runtime 1439.1s

Forgetting probe: train A -> consolidate A -> train B (EWC on) -> re-measure A. Lower `forgetting` = less catastrophic forgetting; `B_after` must stay low or EWC merely blocked learning B.

| variant | #params | A_before | A_after | forgetting | B_after |
|---|---:|---:|---:|---:|---:|
| PC-astro | 14509 | 0.00411 +/- 0.00043 | 1.20640 +/- 0.11301 | +1.20229 +/- 0.11316 | 0.02809 +/- 0.00174 |
| PC-ewc-plain | 14509 | 0.00411 +/- 0.00043 | 0.97024 +/- 0.07239 | +0.96613 +/- 0.07254 | 0.03044 +/- 0.00240 |
| PC-ewc-cal | 14509 | 0.00411 +/- 0.00043 | 0.96931 +/- 0.07262 | +0.96519 +/- 0.07277 | 0.03042 +/- 0.00246 |
| GRU | 15401 | 0.00580 +/- 0.00062 | 2.71333 +/- 0.72897 | +2.70753 +/- 0.72905 | 0.07521 +/- 0.03323 |

Paired (per-seed) forgetting deltas (negative = first variant forgets LESS):

| comparison | mean diff | first-lower-in |
|---|---:|---:|
| EWC-plain vs astro-only | -0.23616 | 5/5 |
| EWC-cal vs astro-only | -0.23709 | 5/5 |
| EWC-cal vs EWC-plain | -0.00094 | 4/5 |
