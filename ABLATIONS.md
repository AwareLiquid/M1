# MT-LNN Ablation Studies

Infrastructure for systematically testing different adapter configurations to understand what drives performance.

## Overview

Phase 5b achieved consistent PPL improvements (-28.5% to -34.4%) across three bases. But **which components matter most**? This ablation suite tests:

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
