# Cross-domain continual learning: does M1's liquid core reduce forgetting?

True cross-domain forgetting probe. Train on **A=wikitext103**, then on **B=tinystories**; `forgetting` = A_after_PPL - A_before_PPL (lower = better retention), measured on each domain's held-out split with PURE next-token cross-entropy. `b_learned` = B_before - B_after must be > 0 or the arm failed to learn B.

Seeds `[0, 1, 2]` | train A 1200 + B 1200 steps | 512d x 6L x 8H | seq 256 | batch 16 | lr 0.0003 | cuda

| arm | params | A_before | A_after | forgetting | B_before | B_after | B_learned | NaN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 60.7M | 244.93 | 10806.86 | +10561.93 +/- 2187.88 | 2273.77 | 15.46 | +2258.31 | 0 |
| liquid | 62.0M | 248.48 | 9791.99 | +9543.51 +/- 2977.14 | 1971.43 | 15.57 | +1955.86 | 0 |

**Paired (per-seed) forgetting delta, liquid - dense:** -1018.4204 +/- 2080.1950  (SNR=0.49; liquid forgets less in 2/3 seeds; per-seed diffs [1572.5391, -3520.5541, -1107.2463])

**Liquid learns B:** True  |  **Both arms catastrophically forget A:** True

## Verdict: WEAK-TREND

Grading: SUPPORTED requires a relative forgetting reduction that clears seed variance (SNR>=1, liquid lower in EVERY seed) AND leaves A usable. WEAK-TREND = the mean favours the liquid core but the effect is noise-dominated and/or both arms still forget A catastrophically -- the continual-learning mechanism is BUILT but its effectiveness is a TARGET, not yet validated. The verdict is reported as-is regardless of sign.