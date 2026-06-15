# PC-Liquid-Core generative-replay (pseudo-rehearsal) report

Seeds: `[0, 1, 2, 3, 4]` (n=5)  |  replay buffer=32 (batch 16, warmup 16)  |  runtime 1707.1s

Forgetting probe: train A -> build replay buffer (REAL subset, or GENERATIVE dream from the post-A model) -> train B rehearsing that buffer -> re-measure A. `dream` = the post-A model's own 1-step MSE on its dream (low = self-consistent, NOT a fidelity-to-real-A measure).

| variant | #params | forgetting | B_after | dream self-MSE |
|---|---:|---:|---:|---:|
| PC-astro | 14509 | +1.20229 +/- 0.11316 | 0.02809 +/- 0.00174 | - |
| PC-replay | 14509 | +0.00598 +/- 0.00102 | 0.03580 +/- 0.00127 | - |
| PC-genreplay | 14509 | +0.02171 +/- 0.00289 | 0.03658 +/- 0.00125 | 0.00062 +/- 0.00021 |
| GRU-replay | 15401 | +0.00595 +/- 0.00256 | 0.05250 +/- 0.00987 | - |
| GRU-genreplay | 15401 | +0.03588 +/- 0.01979 | 0.05152 +/- 0.00446 | 0.02240 +/- 0.00286 |

Paired (per-seed) forgetting deltas (negative = first variant forgets LESS):

| comparison | mean diff | first-lower-in |
|---|---:|---:|
| PC-genreplay vs astro | -1.18057 | 5/5 |
| PC-genreplay vs PC-replay(real) | +0.01573 | 0/5 |
| GRU-genreplay vs astro | -1.16641 | 5/5 |
| PC-genreplay vs GRU-genreplay | -0.01417 | 4/5 |
