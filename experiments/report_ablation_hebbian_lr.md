# Hebbian-lr ablation: is the co-activation term a real knob?

Seeds `[0, 1]` | 150 steps | model 128d x 2L x 8H | seq 64 batch 8 | real WikiText-103 (gpt2) | cpu | runtime 593.0s

Isolated Hebbian (every other v2.0 module OFF). `g_hebb/g_full` is the fraction of the per-batch gradient norm contributed by the Hebbian term alone -- the key diagnostic (value can be ~1e-8 while gradient still matters, or not).

| hebbian_lr | val PPL (mean+/-std) | dPPL vs off | NaN | |L_hebb| | g_hebb/g_full |
|---|---:|---:|---:|---:|---:|
| off | 336.866 +/- 6.589 | -- | 0 | 0.00e+00 | 0.00e+00 |
| 1e-4 | 337.185 +/- 7.247 | +0.320 | 0 | 4.72e-09 | 8.31e-08 |
| 1e-2 | 337.188 +/- 7.248 | +0.322 | 0 | 4.75e-07 | 8.32e-06 |
| 1e-1 | 337.212 +/- 7.258 | +0.346 | 0 | 5.06e-06 | 8.47e-05 |

**Read:** largest Hebbian gradient fraction across the sweep = 8.47e-05; largest |dPPL vs off| = 0.346 (seed noise ~7.258). PPL change is within seed noise and the gradient share is negligible even at hebbian_lr=1e-1 -> the term is effectively inert at these scales; safe to leave OFF (or raise its strength by orders of magnitude before it can matter).
