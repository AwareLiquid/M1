# MT-LNN Ablation Studies

Infrastructure for systematically testing different adapter configurations to understand what drives performance.

## Overview

> ⚠️ **Correction (2026-07-04)**: the Phase 5b "−28.5% to −34.4%" gains
> referenced below were later shown to be **pure LoRA** — the MT adapters
> were frozen at random init by a PEFT integration bug (BENCHMARKS.md
> correction note 2026-07-04; controlled attribution: MT adds −0.064 PPL
> over lora_only, noise-level). The ablation INFRASTRUCTURE in this file
> remains valid; the motivating numbers do not.

Phase 5b achieved consistent PPL improvements (-28.5% to -34.4%) across three bases — **since retracted, see the correction above**. But **which components matter most**? This ablation suite tests:

1. **Layer Interval**: Does adapter density matter? (every 2 vs 4 vs 8)
2. **LoRA Rank**: Is it about capacity? (r=4 vs 8 vs 16)
3. **Adapter Type**: MT vs LoRA vs MT+LoRA (which drives the improvement?)
4. **Protofilaments**: Does biological count (13) matter? (8 vs 13 vs 21)

## Quick Start

### Local (requires GPU)

```bash
# Run adapter_type ablation (MT vs LoRA vs MT+LoRA)
python scripts/run_ablations.py --group adapter_type --device cuda

# Run all ablations (~8h on T4)
python scripts/run_ablations.py --group all --device cuda

# Analyze results
python scripts/analyze_ablations.py artifacts/ablations/ablation_adapter_type_results.json
```

### Kaggle

1. Create new notebook
2. Copy `kaggle/run_ablations.py` content
3. Set `GROUP_TO_RUN` to desired group ('adapter_type', 'layer_interval', 'lora_rank', 'protofilaments', or 'all')
4. Run all cells (~2h per group on T4)
5. Download `ablation_results.zip` from output
6. Extract to `artifacts/ablations/`

## Ablation Groups

### 1. Layer Interval

**Question**: Does adapter placement density matter?

| Config | Coverage | Expected Params | Hypothesis |
|---|---|---:|---|
| every 2 | Dense | ~2× Phase 5b | More coverage → better, but diminishing returns |
| every 4 | Phase 5b default | 1× | Sweet spot (validated) |
| every 8 | Sparse | ~0.5× | Still effective if MT inductive bias is strong |

**Interpretation guide**:
- If `every 2 ≈ every 4 ≈ every 8`: MT architecture (not density) is key
- If `every 2 >> every 8`: More coverage helps (or just more params)

### 2. LoRA Rank

**Question**: Does LoRA capacity matter when combined with MT?

| Config | Rank | Expected Params | Hypothesis |
|---|---:|---:|---|
| r=4 | 4 | ~0.5× Phase 5b LoRA | Sufficient if MT does heavy lifting |
| r=8 | 8 | 1× Phase 5b LoRA | Phase 5b default |
| r=16 | 16 | ~2× Phase 5b LoRA | Overkill if MT is the driver |

**Interpretation guide**:
- If `r=4 ≈ r=8 ≈ r=16`: MT (not LoRA) drives improvement
- If `r=16 >> r=4`: It's about parameter count, not architecture

### 3. Adapter Type (Most Important)

**Question**: What is the contribution of MT vs LoRA?

| Config | Components | Expected PPL | Hypothesis |
|---|---|---|---|
| MT only | MT adapters | Good | MT provides long-context bias |
| LoRA only | LoRA (vanilla) | Baseline | Standard PEFT, no architectural prior |
| MT + LoRA | Both | Best | Complementary (MT=architecture, LoRA=capacity) |

**Critical test**: If `MT-only > LoRA-only`, then MT architecture (not just parameter efficiency) is the key innovation.

**Interpretation guide**:
- If `MT >> LoRA`: Architecture matters (validates microtubule-inspired design)
- If `LoRA >> MT`: It's just efficient fine-tuning (architecture is irrelevant)
- If `MT+LoRA >> both`: Complementary benefits

### 4. Protofilaments

**Question**: Does biological microtubule count (13) matter?

| Config | Count | Expected PPL | Hypothesis |
|---|---:|---|---|
| proto=8 | 8 | Slightly worse | Biological prior matters |
| proto=13 | 13 | Best | Default (from biology) |
| proto=21 | 21 | Similar or worse | 13 is sweet spot, not just "more is better" |

**Interpretation guide**:
- If `13 is best`: Supports biological prior hypothesis
- If `21 > 13 > 8`: It's just model capacity (biology is coincidence)

## Design-coupling audit (unrun ablations that are really design bugs)

Two places where biological naming quietly constrained the math. The first was
caught and fixed in M2-P0; the second is recorded here so it is not rediscovered
the hard way.

1. **GTP-cap distance-decay init** (caught, M2-P0 rounds 3–5): the per-head
   decay spectrum left at most one quasi-global head, starving induction-head
   composition. Pointer-chase went 0.249 → 1.0000 once heads could see far.
   Fix shipped as the `n_global_heads` quota — biology keeps the local heads,
   the global slots are explicit.

2. **`n_heads = 13` couples attention to the protofilament count — and 13 is
   prime.** "One head per protofilament" is naming aesthetics, not mechanism:
   the multi-timescale resonance semantics live in the LNN's
   `n_protofilaments=13`, and nothing in the attention path uses that number.
   The cost is real: with 13 heads the only legal GQA settings are 13:1 (MQA,
   the current default — the config P0 round 4 measured at **acc 0.248**) or
   1:1 (full MHA — **acc 1.0000 but 13× the KV cache**). Every intermediate
   ratio is arithmetically impossible, so the accuracy/memory trade-off cannot
   be tuned at all; production hybrids pick middle ratios (LFM2: 32 heads,
   4:1) precisely because their head counts have divisors. Decoupling proposals
   (16 heads × 64 at d_model 1024, or 16 × 52 keeping 832) are logged in
   HANDOFF §3.8; the sweep has NOT been run — this entry exists so the prime
   lock is treated as a design constraint to remove, not a finding to rediscover.

## Parity verdict, good recipe (2026-08-06) — my own headline from yesterday is downgraded

`m1-parity-verdict-goodrecipe`: pure-LNN stack, beta2 0.999, clip off, 8k
steps, 6 seeds/arm, k=2 and k=8. Rows `pv-*` in
`benchmarks/results/reasoning_depth.jsonl`; arms verified real by the
1,170-parameter delta.

| arm | grok rate | accs |
|---|---|---|
| k=2, selective | **2/6** | 1.000, 1.000, 0.513, 0.513, 0.511, 0.487 |
| k=2, stock | **1/6** | 1.000, 0.513, 0.513, 0.503, 0.497, 0.487 |
| k=8, selective | 0/6 | all ~0.50 |
| k=8, stock | 0/6 | all ~0.50 (per-seed bit-identical with sel — the constant-collapse artifact again) |

### Length-generalization v2, full model, good recipe (2026-08-06)

Complementary axis to the isolated-core verdict: FULL model (attention on),
curriculum L ~ U{1..32}, 30k steps, one arm per process, beta2=0.999 clip=0
lr=3e-4. Rows in `benchmarks/results/parity_lengthgen.jsonl` (v1 rows from
2026-08-05 used the vetoing recipe — not interpretable, superseded).

| run | L16 | L32 | L48 | L64+ |
|---|---:|---:|---:|---:|
| sel s0 | 0.753 | 0.498 | 0.492 | chance |
| stock s0 | 0.996 | 0.566 | 0.492 | chance |
| sel s1 | 0.475 | 0.519 | 0.504 | chance |
| stock s1 | 0.491 | 0.515 | 0.509 | chance |
| **sel s2** | **0.9995** | **0.9951** | **0.749** | 0.536 → chance |
| stock s2 | 0.565 | 0.502 | 0.484 | chance |

Verdict: (a) the ONLY out-of-length transfer in six runs is sel s2
(L48 = 0.749, ~1.5× train length, decayed by L64) — no run learned a clean
length-invariant flip-flop; (b) sel 1/3 vs stock 0/3 on "any L48 transfer"
is directionally consistent with the selective hypothesis and nowhere near
significance; (c) good-recipe in-dist learning is WEAKER here than the
vetoing-recipe v1 (v1 sel s0 hit L32 0.96 in-dist) — the recipe effect is
context-dependent (bare core vs full model), so recipe must be treated as a
swept axis, not a fixed constant, in any promotable claim.

### Where the parity program stands (2026-08-06 synthesis)

Anecdotal positives for input-dependent signed transitions exist in three
independent places (branch bare-probe 3/3; curriculum full-solve seed;
sel s2 length transfer) but **no multi-seed matched-control test has
separated the arms yet**: isolated core 2/6 vs 1/6 (p≈1.0), length-gen 1/3
vs 0/3. k=2 is now understood to be an INVALID separator (bounded parity is
reachable by counting + nonlinear readout in any accumulator; the
impossibility theorems are asymptotic). The live discriminators are k≥8
grok rate at larger budget, and length extrapolation. Both are
budget-limited, not design-limited.

### What this corrects

**The stock core grokked k=2 once.** Not a violation of Sarrof Thm 2 — a
correction to my reading of where the theorem bites. At fixed k=2 the answer
state is a linear sum of two inputs with DISTINCT positional coefficients;
four input combinations give four distinct state values, and the nonlinear
readout (MAP gate + head) classifies XOR from them. The excluded regime is
large/unbounded k, where that route collapses. **k=2 was never a decisive
task**, and yesterday's "first theory-confirmed positive" (bare probe:
selective groks, stock at chance) was a 2/6-vs-1/6 coin-flip difference read
as a separation — Fisher p = 1.0. Downgraded to: consistent with the
mechanism, discriminates nothing.

### What survives

- The **recipe veto** (beta2=0.95 / clip=1.0 each independently prevent the
  k=2 breakthrough): bisected with reproduced controls, stands.
- The **grok-rate protocol** stands — and just did its job: it is the only
  reason this over-claim was caught by the verdict experiment instead of
  shipped as a result.
- The theorem's actual battleground is **k=8+**, where NEITHER arm learns at
  this budget. Stock failing is what the theory demands; selective failing
  means the capability the mechanism exists to add is **not yet demonstrated**
  where it matters.

### Next, in order of information-per-cost

1. **Compare implementations with `experiment/consciousness-m1-v2`** (parity
   0.000 → 1.000, 3/3 seeds claimed) — establish their k, budget and init; if
   that result is at large k, their implementation choices are the lead.
2. **Curriculum k=2 → 8** under the good recipe: both arms CAN learn k=2, so
   it can seed the flip-transition solution cold-start k=8 never finds.
   (`benchmarks/parity_lengthgen.py` is already moving this direction.)
3. **sel_b init**: tanh(1.0)≈0.76 starts transitions far from the flip regime
   (λ_t ≈ −decay needs W·x+b ≲ −2); a mixed/negative-init arm is one config.
4. 8k steps may be short of the k=8 grok time even when reachable — 30k-step
   budgets before any "cannot" is concluded.

## Parity, core isolation: the recipe was vetoing the experiment (2026-08-05)

The follow-up isolation run (`m1-parity-core-isolation`, pure-LNN stack via
`attention_layers=()`, 3 seeds/arm, 10k steps) came back with BOTH arms
bit-identical at chance for every k **including k=1** — and k=1 parity is
copy-a-bit. A setup that fails the trivial task cannot indict the hard one, so
the run was discarded and the failure debugged locally instead of being
recorded as a null.

### The debugging chain, kept because each step exonerates something

1. Bare probe, pure-LNN, copy-a-bit: **1.000 in 150 steps** — the architecture
   moves information fine; the failure is in the harness path.
2. lr (3e-4 vs 3e-3): both learn copy in the bare loop — exonerated.
3. `core_iterations` built-2-run-1 vs built-1: identical learning — the
   bit-equivalence claim holds; exonerated.
4. `evaluate()` shift convention: correct (reads `ans_pos-1`, matching the
   model's internal shifted loss). My own head-to-head diagnostic had read the
   wrong position — the harness was innocent here.
5. My earlier "k=2" bare test had the second bit FIXED at 0 — that is copy,
   not XOR. **True 2-bit XOR is the minimal decisive task.**

### The positive result (bare probe, pure-LNN stack, true XOR)

| arm | outcome |
|---|---|
| `selective_decay=True` | **1.000** (breakthrough ~step 800; reproduced again as the bisect control and its +cosine arm) |
| stock core | **0.499–0.508, pinned at ln 2** — exactly Sarrof Thm 2's prediction |

First theory-predicted, experiment-confirmed result on main: input-dependent
signed transitions enable parity-class computation in the isolated liquid
core; the stock core cannot break the symmetry at all. Converges with the
independent implementation on `experiment/consciousness-m1-v2` (parity 0.000 →
1.000, 3/3 seeds, via input-dependence + negative eigenvalues).

**Caveat that must travel with the result**: the breakthrough is data-stream
sensitive. The same model seed groks on one RNG stream (2/2) and sits at
chance on another (0/3 across three seeds, 2k steps). Verdict experiments must
report **grok rates over many seeds**, never accuracies over few.

### Why the harness (and the Kaggle run) could not see it — bisected

One delta at a time added to the working bare loop (1500 steps, lr 3e-3):

| variant | acc |
|---|---|
| control | **1.000** |
| + `betas=(0.9, 0.95)` | **0.496** |
| + cosine schedule | 1.000 |
| + `clip_grad_norm 1.0` | **0.496** |
| + internal-loss path (`model(ids, labels=labels)`) | 1.000 |

`beta2=0.95` and `clip=1.0` — the harness's historical recipe — **each
independently veto the breakthrough**. Same mechanism, opposite sides: the
breakthrough is a rare large-gradient event; clip truncates it, a fast second
moment adapts to it and neutralises the effective step. `train_model` used
both, which fully explains the isolation run's all-arms-at-chance and casts
the same doubt on every previous parity attempt under this recipe (including
the signed_decay all-arms null: it was run under the same veto and should be
re-examined with the good recipe before "signed is inert" is treated as
final).

Harness now exposes `--beta2` / `--clip` (defaults unchanged for archival
comparability), records both in every row's provenance.

**Verdict experiment running**: `m1-parity-verdict-goodrecipe` — pure-LNN,
beta2 0.999, clip off, 6 seeds/arm, difficulty 2 and 8, tags `pv-d2-*` /
`pv-d8-*`. Deliverable is the grok-rate table.

## Parity x selective_decay, hybrid A/B (2026-08-04, Kaggle T4, 3 seeds — NULL, and a design flaw worth more than the null)

Arms: `--selective_decay` on/off, parity mix curriculum (difficulty 32), 10k
steps, 3 seeds each. Rows `parity-sel-on` / `parity-sel-off` in
`benchmarks/results/reasoning_depth.jsonl` (merged from Kaggle 2026-08-05).

| k | sel-on (3 seeds) | sel-off (3 seeds) | transformer control |
|---|---|---|---|
| 8 | 0.999 / 1.000 / 0.992 | 1.000 / 1.000 / 0.912 | 1.0 |
| 16 | 0.970 / 0.922 / 0.681 | 0.923 / 1.000 / 0.524 | 1.0 |
| 32 | **0.515 ± 0.028** | **0.531 ± 0.039** | 0.68–0.81 |

**No difference between arms at any k.** But the more important finding is
that the experiment could not have seen one: the "mt_lnn" probe is the full
HYBRID block, and its **attention sub-layer computes the parity-adjacent
counting on its own** (the attention-only transformer control reaches 0.68–0.81
at k=32). A stock core scoring 0.92–1.00 at k=16 is not a refutation of Sarrof
Thm 2 — the theorem is about the recurrence, and the recurrence was never
isolated. The null between arms means "attention already does what it can",
nothing about the liquid core.

Two side findings:
- The hybrid is WORSE than the attention-only control at k=32 (≈0.52 vs
  0.68–0.81) — the liquid sub-layer is not merely idle on this task, it appears
  to be in the way. Unexplained; single protocol; noted, not concluded.
- Provenance gap: the jsonl rows did not record `selective_decay` (arms were
  distinguishable only by tag and a 1,170-parameter delta — exactly sel_w+sel_b
  for P=13, S=5, D=8 over 2 layers, which is how the arms were verified to be
  real). The recorder now writes `selective_decay` and `attention_layers`.

**Follow-up running**: `m1-parity-core-isolation` — identical protocol with
`attention_layers=()` (pure LNN stack, the new hybrid-thinning knob). Theory
predicts the stock arm hard-stuck at 0.5 for every k; selective learning any
nontrivial k is the positive. This is the experiment the first one should have
been.

## GQA x global-head quota, first sweep (2026-07-31, Kaggle T4, n=1 — UNREPLICATED)

Decoupled heads made middle GQA ratios expressible for the first time, so the
question was whether a middle ratio plus the `n_global_heads` quota recovers
probe accuracy without paying full-MHA KV. Three configs finished before the
session was cancelled at 12 h (~4 h each); the fourth (full MHA + quota) did
not run. Raw: `benchmarks/results/gqa_quota_sweep_kaggle.jsonl`.

Probe: pointer_chase, difficulty 2, n_values 8, 30k steps, seed 0, 4 heads,
`n_layers=2`, depth 1. Chance is 1/8 = 0.125; the transformer control sat at
0.148 in all three.

| tag | n_kv_heads | n_global_heads | params | MT-LNN acc |
|---|---:|---:|---:|---:|
| gqa-kv1_g2 | 1 (4:1) | 2 | 192,141 | 0.176 |
| gqa-kv2_g2 | 2 (2:1) | 2 | 202,957 | 0.179 |
| **gqa-kv2_g0** | 2 (2:1) | **0** | 202,957 | **1.0000** |

**The middle two rows are a clean A/B: identical parameter count, one knob
different — and the quota is the difference between chance and a perfect
score, in the direction opposite to what M2-P0 concluded.** P0 rounds 3-5 found
the all-decaying GTP init was blocking relational lookup and that freeing heads
fixed it (0.249 -> 1.0000); here, freeing two heads *prevents* the solve that
the untouched decay schedule reaches.

**This is n=1 and must not be treated as a finding.** The task is visibly
bimodal — every run either cracks it (1.0) or sits at chance (~0.15), with
nothing in between — which is exactly the grokking-like behaviour rounds 1-2
described. A single seed cannot distinguish "the quota hurts" from "seed 0
happened not to grok under that init". Before anything is concluded or any
default is changed:

1. Re-run the kv2_g2 / kv2_g0 pair at **3+ seeds**. That is the whole
   experiment; it is 8 GPU-hours and it settles the direction.
2. If the effect survives, the two results are not actually in conflict —
   P0 varied `gamma_init` (making *all* heads global) while this varies
   `n_global_heads` (making *some* heads global while the rest keep a decay
   schedule computed over fewer local heads). Reconciling those two mechanisms
   is the real question, and `_build_alibi_gamma` recomputing the local
   spectrum over `n_heads - n_global_heads` is the first place to look.
3. Note also that difficulty 2 here vs difficulty 4 in P0 means these are not
   the same task instance.

Defaults are unchanged (`n_global_heads=0`), which — if this replicates — is
already the better setting. Nothing shipped on the strength of one seed.

### Replication (2026-08-03, local RTX 5060, seeds 1–2 × 30k steps) — probe DEPRECATED

The 3+-seed replication asked for above is now in. Same pair, same fixed-
difficulty probe (pointer_chase, difficulty 2, n_values 8, kv=2):

| config | seed 0 (Kaggle) | seed 1 (local) | seed 2 (local) | grok rate |
|---|---:|---:|---:|---:|
| kv2_g0 | 1.0000 | 0.1832 | 0.1652 | **1/3** |
| kv2_g2 | 0.179 | 0.1863 | 0.2980 | **0/3** |

1/3 vs 0/3 at n=3 is not evidence of anything (Fisher exact p = 1.0). The
real finding is about the PROBE, not the configs: at 30k steps the fixed-
difficulty task is a Bernoulli grok/no-grok coin flip per seed — every
headline contrast so far (P0 rounds 3–5 "γ freeing fixes it", the 2026-07-31
"quota prevents it") was sampling noise from this bimodality, mutually
consistent and mutually uninformative. Raw: `benchmarks/results/gqa_rep_g{0,2}.log`,
rows tagged `gqa-rep-g0/g2` in `reasoning_depth.jsonl`.

**Protocol decision**: the fixed-difficulty probe is deprecated for config
comparisons. The replacement is the single-cycle + curriculum-mix task
(`--mix`, per-k eval), which grokked reliably in the one local run to date
(loss → 0 by 28k). The g0-vs-g2 question re-runs under that protocol on
Kaggle (`m1-gqa-quota-replication-g0` kernel et seq.); until it lands, the
quota direction is UNDECIDED and `n_global_heads=0` stays default.

### Mix-protocol A/B (2026-08-04, Kaggle T4, 3 seeds each × 30k) — VERDICT

Single-cycle + curriculum-mix (k ~ U{1..4} announced in-input), per-k eval,
kv=2, depth 1, 200K params. Chance ≈ 1/7 = 0.143 (answer ≠ start on a cycle).
Rows tagged `kg-mix-g0` / `kg-mix-g2` in `reasoning_depth.jsonl`.

| config | seed | k=1 | k=2 | k=3 | k=4 |
|---|---:|---:|---:|---:|---:|
| g0 | 0 | **1.000** | 0.184 | 0.160 | 0.167 |
| g0 | 1 | 0.676 | 0.216 | 0.205 | 0.194 |
| g0 | 2 | **0.994** | 0.188 | 0.164 | 0.173 |
| g2 | 0 | 0.240 | 0.131 | 0.142 | 0.144 |
| g2 | 1 | **1.000** | 0.461 | 0.250 | 0.220 |
| g2 | 2 | **1.000** | 0.329 | 0.326 | 0.319 |

Two verdicts:

1. **Quota: no promotable difference.** k=1 solve rate 2–3/3 both ways (g2
   seed 0 failed even 1-hop; g2's surviving seeds show mildly better k=2
   partial credit, ~0.33–0.46 vs ~0.19 — suggestive, not conclusive at n=3).
   `n_global_heads=0` STAYS DEFAULT; the 2026-07-31 "quota prevents the
   solve" headline does not replicate under the mix protocol.
2. **The real result: 1-hop lookup is learnable under the DEFAULT decaying
   attention (5/6 seeds ≥0.68), but composition (k≥2) emerges in NO config at
   depth 1 / 30k / 200K params.** Head configuration was never the binding
   constraint for multi-hop — depth is, exactly as the circuit-depth analysis
   predicts (liquid core in TC⁰; composition needs depth). The earlier local
   loss→0 mix run that DID compose used γ=0.001 + full MHA — freed attention
   may lower the depth barrier, but under production-realistic attention the
   binding constraint is depth. The decisive test is the stack_iterations
   sweep on s5_word (`m1-circuit-separation-parity-s5` kernel, running).

### Circuit-separation kernel, first data (2026-08-04, Kaggle T4, seed 0 — landed)

Rows `sep-parity` / `sep-s5-d1` / `sep-s5-d4` in `reasoning_depth.jsonl`
(30k steps, 200K params):

| task | depth knob | acc | chance | verdict |
|---|---|---:|---:|---|
| parity L=32 | core, d=1 | 0.4921 | 0.5 | **theory hit**: Sarrof Thm 2 (strictly positive gating cannot express parity) confirmed empirically |
| s5_word L=8 | stack, d=1 | 0.0086 | 1/120 ≈ 0.0083 | chance |
| s5_word L=8 | stack, d=4 | 0.0094 | 1/120 | chance — depth alone did NOT unlock S5 |

Reading: parity failing at chance is the PREDICTED parameterisation defect,
not news — it validates the debug gate. S5 at chance for BOTH stack depths is
the finding that needs care: Θ(log n) depth suffices *expressively* (Merrill &
Sabharwal 2025), but at 30k steps / 200K params SGD found nothing at either
depth, so expressivity ≠ learnability here — S5 from scratch likely needs a
curriculum (the mix lesson) and/or far longer training before the depth axis
can show a separation. Single seed; no ranking claims.

Next (theory-driven, falsifiable): `signed_decay` (λ = decay·tanh(s), Grazzi
ICLR 2025) is now implemented behind a default-off flag — prediction: parity
becomes learnable at depth 1 with the flag ON and stays at chance OFF. That
A/B is the cleanest single-variable test in the whole program: theory names
the defect, one parameter family fixes it, both arms falsifiable.

### signed_decay A/B: NEGATIVE — and the diagnosis was wrong by one axis (2026-08-04/05)

Kaggle 3-seed A/B (`parity-signed` / `parity-stock`, 10k steps) + local 3-arm
probe (off / stock-init / mixed-init s~U(−2,2), 2500 steps): **every arm sits
at exactly ln 2 loss and chance accuracy.** The per-seed bit-identical accs
across Kaggle arms are the constant-prediction + shared-eval-set artifact,
NOT a dead flag — a local differential probe confirmed the two arms' losses
diverge numerically (flag live), and the mixed init rules out the tanh(3)
saturation explanation.

**Corrected diagnosis: sign was the wrong (or at least insufficient) axis.**
A constant-λ diagonal core — any sign — computes fixed-weight linear sums
Σ λ^(T−t)·b(x_t). Parity needs the TRANSITION to read the token (flip on 1,
hold on 0): λ_t = λ(x_t). Grazzi's negative-eigenvalue result applies to
SELECTIVE SSMs (Mamba-class, where Δ(x) is already input-dependent and only
the sign range was missing). M1's liquid core is input-INDEPENDENT (the
2026-08-03 three-question audit said exactly this) — so it is missing BOTH
selectivity and sign. Notably the ORIGINAL LTC theory has input-dependent
τ(x); this implementation dropped it for `pscan_constant_A` speed.

`selective_decay` (λ_t = decay · tanh(W_sel·x_t + b_sel), general per-step
pscan, b_sel init 1.0 = live-gradient region) is implemented behind a
default-off flag. Prediction: parity learnable with selectivity ON, chance
OFF. If ON also fails, the next suspect is the readout path, not the
transition.

### selective_decay A/B: HIT — theory-confirmed separation (2026-08-05, local)

Pure-LNN stack (`--attention_layers` = none, 138K params — attention REMOVED
so the liquid core cannot borrow TC⁰ from the attention side), parity
difficulty 8, mix curriculum, depth=1 fixed, 2500 steps, 3 seeds:

| arm | k=1 | k=2 | k=4 | k=8 |
|---|---:|---:|---:|---:|
| stock (input-independent λ) | 0.826 | 0.583 | 0.481 | 0.509 |
| **selective_decay** | **1.000** | **1.000** | **1.000** | **1.000** |
| transformer control (246K) | 1.000 | 1.000 | 1.000 | 0.999 |

- **Sarrof Thm 2 — where it actually bites (updated 2026-08-06 with the
  fixed-k=2 verdict)**: the theorem's excluded regime is LARGE/unbounded k.
  At FIXED small k (k=2: 4 input combos → 4 distinct linear-sum states),
  the nonlinear readout can classify XOR from distinct positional
  coefficients — so stock CAN grok k=2 (6-seed fixed-k=2 A/B: selective
  2/6 vs stock 1/6, Fisher p=1.0 — "separation at k=2" was my misreading,
  correctly downgraded in 87532b5). The clean d8 separation below is
  dominated by k≥4, and the decisive large-k evidence is the d16 k=16
  arm: selective 3/3 at 1.000 vs stock 3/3 at chance (0.505/0.510/0.492)
  — the pure-LNN stock core CANNOT do large-k parity, exactly the
  theorem's regime. The earlier hybrid runs showed stock at 1.0 only
  because the attention layers (TC⁰-complete) silently solved parity at
  short T — hybrid was not a valid liquid-core probe.
- **selective_decay closes the gap completely**: 3/3 seeds at 1.000 across
  all k (incl. k=16), matching the transformer at **44% fewer parameters**
  (138K vs 246K).
- The difficulty-16 arm at 2500 steps failed for BOTH arms (k≥2 ≈ chance):
  training time, not mechanism — the d8 arm above separates at the same
  step budget. Parity needs more steps as L grows; curriculum helps but does
  not remove the budget.
- Rows: `purelnn-parity-ab-{stock,selective}` / `purelnn-d8-ab-{stock,selective}`
  in `reasoning_depth.jsonl`.
- **Open**: does selective keep 1.0 at L=16/32 where the transformer starts
  to sag (k=16 ≈ 0.92–0.99)? That is the "beyond" evidence chain: the
  transition-parameterised liquid core holding where attention sags.

### Difficulty-16 long-budget confirmation: selective 1.0 where transformer sags (2026-08-05, local)

Same pure-LNN stack, difficulty 16, 6000 steps, 3 seeds:

| arm | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| stock (input-independent λ) | 0.501 | 0.750 | 0.481 | 0.520 | 0.502 |
| **selective_decay** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| transformer control (246K) | 1.000 | 1.000 | 1.000 | 1.000 | 0.986 |

- 6k steps does NOT rescue stock: 3/3 seeds still at chance for k≥2 — the
  defect is expressivity (Sarrof Thm 2), not training budget. (Seed 2's
  k=2:1.0 is grokking noise; k≥4 still chance.)
- **selective_decay holds 1.000 across ALL k including k=16 where the
  transformer begins to sag (0.986) — at 138K vs 246K params (−44%)**.
  First measured point where the transition-parameterised liquid core
  exceeds a matched transformer on the same probe.
- Rows: `purelnn-d16-sel-long` / `purelnn-d16-stock-long`.
- Honest scope: toy probe (200K-class), depth-1 only; k=32 (Kaggle
  protocol) and stack-depth interplay remain open. No ranking claims
  beyond this task.

### k=32 protocol: everyone hits the grokking budget wall (2026-08-05, local)

Same pure-LNN stack, difficulty 32 (T=35), 6000 steps, 3 seeds:

| arm | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 |
|---|---:|---:|---:|---:|---:|---:|
| stock | 0.513 | 0.500 | 0.495 | 0.505 | 0.494 | 0.507 |
| selective | 0.513 | 0.500 | 0.495 | 0.505 | 0.494 | 0.507 |
| transformer control (246K) | 1.000 | 1.000 | 1.000 | 1.000 | 0.997 | **0.557** |

- **k=32 is the budget wall for EVERY architecture at 6000 steps** — even
  the transformer (which solves k≤16 at 1.0) collapses to 0.557 ≈ chance at
  k=32. This reproduces the historical "fixed L=32 all-chance for every arm"
  (ABLATIONS P0-C′); mix curriculum lowers the wall but does not remove it.
- selective at k=32 does NOT grok within 6000 steps — this is a training
  budget finding, NOT a mechanism failure: grokking delay grows with
  sequence length, and d8/d16 show the mechanism works. A decisive k=32
  test needs 12k+ steps (or the Kaggle T4 line where the 10k-step
  `m1-parity-selective-ab` kernel is still the canonical protocol).
- **Interpretation for the "beyond" claim**: the sweet spot is d16 — the
  highest difficulty where selective still groks to 1.0 within budget while
  the transformer has already begun to sag. k=32's wall is shared, so it
  does not separate architectures; it only bounds all of them.
- Rows: `purelnn-d32-sel-long` / `purelnn-d32-stock-long`.

### Stack-depth interaction: depth does NOT substitute for selectivity (2026-08-05, local)

The M2 main-line question — can weight-tied whole-block depth
(`stack_iterations`, the J-Space "add depth at the bottleneck, cost 1/64"
idea) rescue the input-independent core from TC⁰? Pure-LNN stack, stack
depth 4 (`--stack --eval_depths 4`), mix curriculum, 3 seeds:

difficulty 16, 2500 steps (budget-matched to the core-depth-1 runs):

| arm (stack4) | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| stock | 0.501 | 0.501 | 0.501 | 0.505 | 0.497 |
| selective | 0.501 | 0.501 | 0.501 | 0.505 | 0.497 |
| transformer control | 1.000 | 1.000 | 1.000 | 1.000 | 0.904 |

- **stack depth 4 does not rescue either arm at 2500 steps / d16** — both
  sit at exactly ln 2. Same budget, core-depth-1 selective had already
  grokked d8 (3/3 at 1.0). Weight-tied depth alone does not substitute for
  input-dependent transition at this scale/budget.
- The stack-depth axis multiplies compute per step (~4×), so it makes the
  grokking-budget problem WORSE per wall-clock second — consistent with the
  theory that selectivity is the binding constraint (the transition must
  READ the token), not raw depth.
- **Open**: stack × selective at d8 with a budget long enough to isolate
  depth effects (running as `stack4-d8-*`); stack_iterations at d16 with
  6000+ steps would also be needed before depth can be ruled out entirely.
- Rows: `stack4-d16-stock` / `stack4-d16-selective`.

### Stack-depth interaction verdict: depth does NOT rescue the core (2026-08-05, local, complete)

Difficulty 8, stack depth 4, 3000 steps, 3 seeds (budget-matched to the
core-depth-1 d8 runs where selective grokked 3/3):

| arm (stack4) | k=1 | k=2 | k=4 | k=8 |
|---|---:|---:|---:|---:|
| stock | 0.667 | 0.581 | 0.503 | 0.499 |
| **selective** | **1.000** | **0.915** | **0.854** | **0.835** |
| transformer control | 1.000 | 1.000 | 1.000 | 1.000 |

- **Weight-tied whole-block depth does NOT rescue the input-independent
  core**: stock+stack4 sits at chance for k≥2 (mean 0.58/0.50/0.50; only
  seed 0 manages k=1:1.0 k=2:0.75 — grokking noise, k≥4 all chance).
  Selective+stack4 does grok (k=1 1.0, k=2 0.92, k=8 0.84) but strictly
  WORSE than selective+core-depth-1 (which was 3/3 at 1.000 across all k
  at d8 with FEWER steps per block).
- **Verdict for the M2 main-line question** ("can an O(1) recurrent model
  climb out of TC⁰ via input-dependent transfer + weight-tied depth?"):
  selectivity is the binding constraint; depth alone buys nothing, and
  depth ON TOP of selectivity hurts (4× compute per step for a strictly
  worse result than core-depth-1). The liquid core's escape route from
  TC⁰ is the transition reading the token — J-Space workspace residency
  is a depth argument and this experiment says depth is not the lever.
- Rows: `stack4-d8-stock` / `stack4-d8-selective`.

### Curriculum unlocks parity for BOTH arms — attention counting confound (2026-08-05)

Local 8k probe: fixed L=32 all-chance for every arm, but curriculum-mix
(L ~ U{1..32}) separates immediately at n=1: selective L=2 = 1.000 vs stock
0.753. Kaggle 30k × interleaved arms (`m1-parity-selective-curriculum-ab`,
CANCELLED at ~2h — weekly GPU quota exhausted — 4/6 runs salvaged from
per-run snapshots):

| arm | seed | L=8 | L=16 | L=32 |
|---|---:|---:|---:|---:|
| selective | 0 | 1.0 | **1.000** | **1.000** — first full parity solve in the program |
| stock | 0 | 1.0 | 0.879 | 0.528 |
| selective | 1 | 1.0 | 1.000 | 0.619 |
| stock | 1 | 1.0 | 1.000 | 0.658 |

Honest reading: (a) curriculum + 30k makes parity largely learnable for BOTH
arms to L≥16 — parity ∈ TC⁰ and the ATTENTION path can count bits and read
out mod 2, bypassing the liquid core (the 2026-08-03 memory warned exactly
this); (b) at L=32 the arms overlap across seeds (sel {1.0, 0.62} vs stock
{0.53, 0.66}) — suggestive, not conclusive at n=2. **The in-distribution
task cannot isolate the core's contribution.**

Discriminator now running: `benchmarks/parity_lengthgen.py` — train
L ~ U{1..32}, evaluate at L ∈ {48, 64, 96, 128}. A counting shortcut
degrades out-of-length; a genuine flip-flop recurrence (expressible only
with input-dependent λ) generalizes. Both directions falsifiable: if
selective also collapses out-of-length, it learned the shortcut too.

### selective_decay on real TEXT — protocol lesson + v2 rerun (2026-08-05/06, local CPU)

Does the parity-proven selective transition help language modeling?
`benchmarks/text_selective_ab.py`: WikiText-2, tiny matched model
(d_model=104, 2 layers, 10.65M params — embedding-dominated), 2000 steps,
3 seeds, fp32 CPU, stock vs selective_decay.

**v1 protocol (broken — do not cite)**: trained on first 2000 lines of the
TRAIN split (~80k tokens), evaluated on the OFFICIAL VALIDATION split (a
different distribution). Result: val_ppl ~ 10^4 for BOTH arms
(stock 39287±7398, selective 41593±13746) — the model cannot generalize
80k tokens → full val set, and a PPL of 10^4 drowns any A/B difference.
Rows in `text_selective_ab.jsonl` from this protocol are marked by their
magnitude; they are NOT evidence either way.

**v2 protocol (running)**: held-out eval FROM THE TRAIN SPLIT (lines
2000-2300, same distribution as training), checkpoints saved for resume.
This is the protocol lesson the parity line already learned: a fixed
out-of-distribution probe reads as "both fail" and cannot separate arms.

**v3 protocol — THE valid one (complete)**: full WikiText-2 train split
(~3.6M tokens, 17.8k chunks), held-out = last 3000 lines of the SAME
split (~974 chunks). 2000 steps, batch 16, 3 seeds, fp32 CPU:

| arm | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---:|---:|---:|---:|
| stock | 666.16 | 674.80 | 678.23 | 673.07 ± 6.22 |
| **selective** | **662.96** | **671.94** | **677.25** | **670.72 ± 7.22** |

- **Directional signal, not yet significant**: selective is lower in
  ALL 3 paired seeds (−3.20 / −2.86 / −0.99 PPL, ~0.35% mean), but the
  gap is inside the ±6-7 seed std. At 10.6M params the selective weights
  are 0.01% of the model — 2000 steps cannot amplify them. The consistent
  paired direction (3/3) is the reason to run the 125M version (GPU).
- v3 rows in `text_selective_ab.jsonl` are the last 6; v1/v2 rows (10^4
  PPL) are protocol artifacts, not evidence.
- Checkpoints in `data/text_ab_ckpt/` (sel{0,1}_s{0,1,2}.pt) enable
  longer-budget reruns without retraining.

Windows tooling note: torch 2.5.1 + transformers 5.x + pyarrow have an
OpenMP/DLL conflict — pyarrow read_table crashes (0xC0000005) once torch is
loaded; also `-X faulthandler` alone crashes with transformers 5.x. The
parquet data was converted to plain .txt by a standalone pyarrow pass
(`data/wikitext2_{train,val}.txt`); the script reads text only.

## Expected Results

Based on Phase 5b validation, we expect:

1. **Adapter Type**: `MT-only` should significantly outperform `LoRA-only` (validates architecture)
2. **Layer Interval**: `every 4` should be near-optimal (diminishing returns beyond)
3. **LoRA Rank**: `r=8` should be sufficient (flat performance across ranks)
4. **Protofilaments**: `13` should be best (biological prior)

If results deviate, we learn:
- MT architecture may not be the driver (if LoRA-only wins)
- It's about coverage, not architecture (if every 2 >> every 8)
- It's about capacity, not priors (if r=16 >> r=8 or proto=21 >> proto=13)

## Running Ablations

### Full Suite (Kaggle T4, ~8h)

```bash
python scripts/run_ablations.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --group all \
    --steps 200 \
    --batch 1 \
    --seq_len 384 \
    --grad_accum 8 \
    --device cuda
```

### Single Group (Kaggle T4, ~2h)

```bash
# Most important: MT vs LoRA vs MT+LoRA
python scripts/run_ablations.py --group adapter_type --device cuda

# Layer density
python scripts/run_ablations.py --group layer_interval --device cuda

# LoRA capacity
python scripts/run_ablations.py --group lora_rank --device cuda

# Biological prior
python scripts/run_ablations.py --group protofilaments --device cuda
```

### Dry Run (Check Configs)

```bash
python scripts/run_ablations.py --group all --dry_run
```

## Analyzing Results

### Single Group

```bash
python scripts/analyze_ablations.py artifacts/ablations/ablation_adapter_type_results.json
```

Output:
- Comparison table sorted by PPL
- Group-specific analysis (e.g., "MT vs LoRA contribution")
- Interpretation guide

### Compare Across Groups

```bash
python scripts/analyze_ablations.py artifacts/ablations/*.json --compare
```

Output:
- Individual group analysis
- Cross-group comparison (best from each)
- Overall winner

## File Structure

```
scripts/
  run_ablations.py        # Main runner (local or Kaggle)
  analyze_ablations.py    # Analysis + visualization

kaggle/
  run_ablations.py        # Kaggle notebook version

artifacts/ablations/      # Results (after running)
  ablation_adapter_type_results.json
  ablation_layer_interval_results.json
  ablation_lora_rank_results.json
  ablation_protofilaments_results.json
```

## Next Steps

1. **Run adapter_type first** (most important) → validates MT architecture claim
2. If MT wins: run layer_interval to find optimal density
3. If LoRA wins: re-evaluate architecture contribution
4. Run protofilaments to test biological prior
5. Document findings in `ABLATIONS_RESULTS.md`

## Adding New Ablations

Edit `scripts/run_ablations.py` and add to `ABLATION_GROUPS`:

```python
ABLATION_GROUPS["my_new_group"] = [
    AblationConfig(
        name="my_config",
        description="What this tests",
        recipe_fn="phase5b",  # or "mt_only" or "lora_only"
        recipe_kwargs={"lora_rank": 12, ...},
    ),
    # ... more configs
]
```

Then run:
```bash
python scripts/run_ablations.py --group my_new_group
```

## Notes

- Each ablation trains for 200 steps (same as Phase 5 validation)
- Uses WikiText-2 (same as Phase 5b)
- Results are comparable across groups (same base, data, steps)
- Wall time: ~10-15 min per config on T4 (200 steps × batch 1 × grad_accum 8)

## Honest Negative: Learnable `log_tau` (`--learn_tau`) — tested, NOT adopted

**Question**: The MT resonance time-constants `tau = softplus(log_tau) + tau_min`
(shape (P,S) = (13,5) per layer) are initialised as a fixed multi-scale filter
bank. Should `log_tau` be *learned* per layer instead of staying a fixed prior?

**Why it looked broken at first (and wasn't)**: Diagnostics showed every layer's
`log_tau` moving by an *identical, data-independent* amount across a run
(cross-layer std == 0). Root cause measured, not guessed:

- The model trains in **bf16**, so `log_tau` is a bf16 parameter. Near |log_tau|~1
  the bf16 ULP is ~0.0039.
- Under the default `lr=2e-4`, the Adam step on `log_tau` is ~1e-4 **< ULP** →
  rounded away. The only thing that moved it was `weight_decay=0.01` dragging all
  elements uniformly (hence cross-layer std == 0). Gradient was *not* disconnected
  (fp32 grad norm 2.2e-7, bf16 1.87e-7 — alive but below the rounding floor) and
  `requires_grad` was correctly True after the re-arm fix.

**The A/B fix tested** (`exp/learn-tau` branch): put `log_tau` in its own AdamW
group with `lr × 50` and `weight_decay = 0`, so the step clears the bf16 ULP.

| Metric | baseline (main, no `--learn_tau`) | `--learn_tau ×50` |
|---|---|---|
| `log_tau` cross-layer std (mean / max) | 0 / 0 (pure WD drift) | **0.109 / 0.408** |
| Mechanism verdict | — | **LEARN_TAU_PASS** (data-dependent) |
| SFT loss, steps 2500–3000 (mean) | **1.1333** | 1.2178 (~7.5% worse) |

Local reproduction (`_diag_learn_tau.py`) confirmed the mechanism independently:
baseline single-group bf16 moved exactly 0/65 elements; `learn_tau ×50` moved
25/65 and **diverged across data seeds** (cross-seed divergence 0.21 > 0 ⇒ genuine
data-dependent learning).

**Verdict — adopted? NO.** The mechanism works (`log_tau` *can* be learned
data-dependently once it clears the bf16 ULP), but learning it **does not improve
quality — it slightly hurts** (~7.5% worse final-window loss, gap widening over
training). This is consistent with the design intent: the multi-scale resonance is
meant to be a *fixed* filter bank (a structural prior), not a free parameter. Per
project discipline ("only merge if it goes smoothly **AND** is better"), this is
recorded as an **honest negative**: the `exp/learn-tau` branch is **kept for
archive, not merged**. `main` keeps the core re-arm fix (scale gate trains) and
fixed `log_tau`.

> Discipline note: same as the Kuramoto R=0.58 artefact lesson — measure first,
> and report negatives faithfully rather than shipping a change that "works" but
> doesn't help.
