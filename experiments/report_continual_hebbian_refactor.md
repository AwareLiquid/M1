# Stage 3 step 2: Hebbian refactor as a continual-learning consolidation prior

Seeds `[0, 1]` | train A 120 + B 120 steps | 128d x 4L | seq 128 | disjoint WikiText-103 regions | cpu | red line cap=5% | runtime 1388.8s

Forgetting probe: train A -> measure A PPL -> train B -> re-measure A. `forgetting` = A_after - A_before (lower = less catastrophic forgetting). `B_after` must stay low or the variant merely blocked learning B. `C` is a held-out region never trained on. Tasks are disjoint but SAME-domain WikiText slices (a mild shift); true cross-domain OOD needs a second corpus and is deferred.

| setting | base_lr | realized frac | A_before | A_after | forgetting | B_after | C (held-out) | NaN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| off | None | 0.00e+00 | 466.68 | 227.93 | -238.75 +/- 6.16 | 207.31 | 176.08 | 0 |
| blr100 | 100.0 | 4.48e-02 | 473.78 | 256.00 | -217.78 +/- 24.56 | 234.62 | 198.58 | 0 |
| blr300 | 300.0 | 4.85e-02 | 476.95 | 259.17 | -217.78 +/- 24.04 | 237.59 | 201.49 | 0 |

Paired (per-seed) forgetting delta, Hebbian ON minus OFF (negative => Hebbian forgets LESS):

| setting | mean diff | on-lower-in |
|---|---:|---:|
| blr100 | +20.9690 | 0/2 |
| blr300 | +20.9687 | 0/2 |

## Verdict (honest, Stage 3 step 2)

**The same-domain setup produced POSITIVE transfer, not forgetting.** `forgetting` is
NEGATIVE for every variant (off -238.75): task A's PPL *improved* after training on B,
because B is same-domain WikiText and continued LM training transfers positively back to
A. So there is no catastrophic forgetting here to consolidate against -- the probe is
inconclusive *about forgetting specifically*, and that is a property of the mild
same-domain shift, not of the Hebbian term.

**Within that, Hebbian does NOT help retention and uniformly HURTS PPL.** The paired
per-seed forgetting delta is +21 (Hebbian forgets MORE / improves A LESS), on-lower-in
0/2. B_after (234-238 vs 207) and the never-trained held-out region C (198-201 vs 176)
are consistently WORSE with Hebbian on, at both base_lr settings. This is the exact same
signature as step 1: the Hebbian gradient competes with the LM objective and raises PPL
everywhere -- including on the task we hoped it would protect. Stability is intact
(NaN=0, red line held at realized share ~4.5-4.9%).

**Per-seed structure.** seed 0 shows Hebbian clearly worse (A_after 276 vs 231); seed 1
shows it nearly matching off (236 vs 225). So the harm is seed-dependent in magnitude but
never turns into a benefit.

**Combined Stage 3 conclusion (steps 1 + 2).** Across in-distribution PPL, long-context
consistency, and continual retention, the refactored Hebbian branch -- once made to
actually contribute up to the 5% safety red line -- never helps and mildly-to-clearly
hurts. The refactor is mechanistically sound and safe (dynamic gate, capped gradient,
NaN-free), so this is a clean NEGATIVE result, not a plumbing failure. Recommendation
stands: keep `use_hebbian_refactor=False`; the branch remains in-tree as documented,
safe, but non-beneficial.

**What is NOT yet tested (honest scope).** A genuine catastrophic-forgetting regime needs
genuinely DISJOINT DOMAINS (e.g. Wikipedia vs code vs dialogue), which requires a second
tokenised corpus this local run does not have. Only in such a regime -- where training B
truly overwrites A -- could a consolidation prior plausibly pay off. That cross-domain
corpus is the necessary next ingredient and is deferred to a dedicated follow-up rather
than faked with same-domain slices.
