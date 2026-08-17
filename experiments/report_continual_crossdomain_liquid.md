# Cross-domain continual learning: does the `consolidation` mechanism reduce forgetting?

True cross-domain forgetting probe. Train on **A=wikitext103**, then on **B=tinystories**; `forgetting` = A_after_PPL - A_before_PPL (lower = better retention), measured on each domain's held-out split with PURE next-token cross-entropy. `b_learned` = B_before - B_after must be > 0 or the arm failed to learn B.

Arms: `dense` (no anti-forgetting control), `liquid` (full liquid core), `consolidation` (dense backbone + EWC: Fisher-weighted anchor to A, lambda=5000.0, fisher_batches=50). Headline treatment = **consolidation** (paired vs the dense control).

Seeds `[0, 1, 2]` | train A 1200 + B 1200 steps | 512d x 6L x 8H | seq 256 | batch 16 | lr 0.0003 | cuda

| arm | params | A_before | A_after | forgetting | B_before | B_after | B_learned | NaN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 60.7M | 244.93 | 10806.86 | +10561.93 +/- 2187.88 | 2273.77 | 15.46 | +2258.31 | 0 |
| liquid | 62.0M | 248.48 | 9791.99 | +9543.51 +/- 2977.14 | 1971.43 | 15.57 | +1955.86 | 0 |
| consolidation | 60.7M | 244.93 | 1878.02 | +1633.09 +/- 105.23 | 2273.77 | 16.03 | +2257.74 | 0 |

**Paired (per-seed) forgetting delta, consolidation - dense:** -8928.8417 +/- 2155.1229  (SNR=4.143; consolidation forgets less in 3/3 seeds; per-seed diffs [-10171.0494, -10718.0329, -5897.4429])
**(also) liquid - dense:** -1018.4204 +/- 2080.1950 (liquid lower in 2/3; diffs [1572.5391, -3520.5541, -1107.2463])

**consolidation learns B:** True  |  **Both control & treatment catastrophically forget A:** True

## Verdict: WEAK-TREND

Grading: SUPPORTED requires a relative forgetting reduction that clears seed variance (SNR>=1, treatment lower in EVERY seed) AND leaves A usable. WEAK-TREND = the mean favours the treatment but the effect is noise-dominated and/or both arms still forget A catastrophically -- the anti-forgetting mechanism is BUILT but its effectiveness is a TARGET, not yet validated. The verdict is reported as-is regardless of sign.