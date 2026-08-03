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
