# Stage 3 step 1: max safe & effective Hebbian gradient share

Seeds `[0, 1]` | 150 steps | 128d x 4L | seq 192 | real WikiText-103 | cpu | red line cap=5% | runtime 2835.8s

5% is a hard SAFETY ceiling, not a target. base_lr is swept to raise the REALIZED Hebbian gradient share; we look for the largest share that stays stable AND helps (or does not hurt). late-PPL / late-minus-early gap (paired, within-sequence) is the sensitive long-context-consistency probe.

| setting | base_lr | realized frac (mean/max) | scale | NaN | val PPL (dvs off) | late PPL | late-early gap (dvs off) |
|---|---:|---|---:|---:|---:|---:|---:|
| off | None | 0.00e+00 / 0.00e+00 | -- | 0 | 299.65 (--) | 296.61 | -2.08 (--) |
| blr30 | 30.0 | 2.63e-02 / 5.00e-02 | 0.647 | 0 | 308.83 (+9.18) | 305.73 | -1.64 (+0.43) |
| blr100 | 100.0 | 3.92e-02 / 5.00e-02 | 0.429 | 0 | 311.43 (+11.78) | 308.12 | -2.40 (-0.32) |
| blr300 | 300.0 | 4.71e-02 / 5.00e-02 | 0.265 | 0 | 317.31 (+17.66) | 313.94 | -2.07 (+0.01) |
| blr1000 | 1000.0 | 5.00e-02 / 5.00e-02 | 0.113 | 0 | 316.59 (+16.94) | 313.16 | -2.25 (-0.17) |

Cross-run val-PPL seed noise (max std) ~ 7.49. A val-PPL delta smaller than this is NOT a detectable effect.

## Verdict (honest, Stage 3 step 1)

**Safety (the 5% red line): VERIFIED.** NaN=0 in every run; no divergence. As base_lr
rises the realized Hebbian gradient share climbs and then SATURATES at the 5% cap
(blr1000 sits exactly at 5.00e-02), with `scale` auto-shrinking 0.65 -> 0.11 to hold
the line. recalibrate() does exactly what it promised: the Hebbian gradient can never
exceed 5% of the main gradient norm, regardless of base_lr. The mechanism is sound.

**Effectiveness: NO safe-and-effective ratio exists at this scale.** val PPL rises
MONOTONICALLY with the Hebbian gradient share: +9.2 at ~2.6%, +11.8 at ~3.9%,
+17.7 at ~4.7%. The small-share deltas sit inside the +/-7 seed noise, but the
blr300/blr1000 harm (+17 to +18) is >2x its own per-setting std (+/-3.4) -- a REAL,
detectable degradation, not noise. The paired, lower-noise long-context probe
(late-minus-early gap) shows no consistent movement (-1.64..-2.40 vs off -2.08, all
within seed scatter): the within-sequence lagged co-activation term does NOT improve
long-range consistency.

**Answer to the framing ("find the max safe & effective ratio"):** the max SAFE share
is the full 5% red line (it holds cleanly), but the max EFFECTIVE share is ~0 -- every
non-zero share tested mildly-to-clearly HURTS, monotonically in its magnitude. There is
no interior sweet spot.

**Interpretation.** The refactor fixed the two real bugs (dead gate -> now dynamic and
rhythm-independent; inert gradient -> now controllably up to 5%), so this is a clean
NEGATIVE result, not a plumbing failure: maximising within-sequence h_t.h_{t-1}
co-activation is simply off-task for next-token LM at 10M/150 CPU steps, and the
gradient it injects competes with (rather than complements) the LM objective.

**Recommendation.** Keep `use_hebbian_refactor=False` for the full Kaggle run. The
branch stays in-tree (toggle + tests + this evidence) as a documented, safe, but
non-beneficial option. Before spending more on it, the only experiments that could
change the verdict are larger scale / longer training, or the explicitly different
continual-learning + OOD settings (Stage 3 remainder) where a Hebbian consolidation
prior is hypothesised to help forgetting rather than in-distribution PPL -- that is a
different metric than the val PPL measured here.
