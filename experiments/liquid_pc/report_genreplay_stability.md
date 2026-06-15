# PC-Liquid-Core generative-replay STABILITY probe

Seeds: `[0, 1, 2, 3, 4]` (n=5)  |  6 dreams/seed at scarce buffer=8 (warmup 16)  |  runtime 3182.9s

Isolation: per (seed,kind) train A ONCE, snapshot post-A, then repeat N dreams -- each restores the IDENTICAL snapshot and trains B with an IDENTICAL training RNG, so the ONLY varying factor is the dream content. Within-seed std of forgetting = pure dream-induced instability. Lower std = more reliable dreams.

| kind | within-seed forgetting std (mean over seeds) | within-seed forgetting mean | pooled forgetting |
|---|---:|---:|---:|
| PC | 0.00464 +/- 0.00308 | +0.02813 | +0.02813 +/- 0.00802 |
| GRU | 0.01857 +/- 0.01976 | +0.06025 | +0.06025 +/- 0.05865 |

**Stability edge (paired, within-seed std):** PC more stable (tighter dream-to-dream forgetting) in **4/5** seeds; mean std diff PC-GRU = **-0.01394** (negative = PC more stable).

**Mean forgetting (paired):** PC lower in 3/5 seeds; mean diff PC-GRU = -0.03212.
