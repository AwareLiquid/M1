># Testing Phase 5b on New Model Families

## Current Status

**Validated families**:
1. ✅ **Llama family** (TinyLlama-1.1B-Chat): -28.5% PPL
2. ✅ **Qwen family** (Qwen-2.5-1.5B, 3B): -27.7%, -34.4% PPL

**Next targets**:
3. ⏳ **Llama-3 family** (Llama-3-8B-Instruct): Expected -25% to -35%
4. 🔲 **Mistral family** (Mistral-7B-Instruct-v0.3): To be tested
5. 🔲 **Phi family** (Phi-3-mini-4k-instruct): To be tested

## Llama-3-8B Test (Item 4.2)

**Model**: `meta-llama/Meta-Llama-3-8B-Instruct`  
**Family**: Llama-3 (GPT-NeoX derivative, 32 layers)  
**Expected**: Similar PPL improvement (-25% to -35%) with same Phase 5b recipe

### Why Llama-3?

1. **Third family validation**: After TinyLlama (Llama-1) and Qwen-2.5, Llama-3 is a distinct architecture
2. **Scale test**: 8B is largest model tested so far (positive scaling trend predicts stronger improvement)
3. **Production relevance**: Llama-3-8B is widely used, so validation matters for practitioners
4. **Open weights**: Fully reproducible without API keys

### Running the Test

#### Local Sanity Check (CPU)

```bash
# Verify recipe attaches correctly (no training)
PYTHONPATH=/e/M1 python scripts/test_llama3_recipe.py
```

Expected output:
- Wrapped layers: [3, 7, 11, 15, 19, 23, 27, 31]
- Trainable %: ~0.1-0.15% (~8-10M params for 8B base)
- Generation works

#### Kaggle Training (~4h on T4)

1. Create new Kaggle notebook
2. Copy content from `kaggle/awareliquid_train_llama3_phase5b.py`
3. Run all cells
4. Download `llama3_phase5b.zip`
5. Extract to `benchmarks/kaggle_llama3_run/`

**Expected results**:
- Base PPL: ~8-12 (WikiText-2 validation)
- Adapter PPL: ~6-9 (25-35% improvement)
- Trainable: 0.1-0.15% (~8-10M params)
- Wall: ~4h (1000 steps × batch 1 × grad_accum 8)

### Success Criteria

| Criterion | Target | Reasoning |
|---|---|---|
| PPL improvement | ≥25% | Consistent with previous results |
| Trainable % | <0.2% | Ultra-parameter-efficient |
| Training stability | No NaN/divergence | Recipe should work without tuning |
| Generation quality | Functional | Sanity check (not production eval) |

**If successful**: Phase 5b recipe validated on **3 model families** across **5 sizes** (1.1B, 1.5B, 3B, 8B).

**If unsuccessful** (<20% improvement or training fails):
1. Check if 8B scale needs different `init_scale` (may need smaller, e.g., 5e-4)
2. Try 2000 steps instead of 1000 (larger models may need more steps)
3. Verify LoRA targets match Llama-3 architecture
4. Check logs for numerical issues

## Other Families to Test

### Mistral-7B-Instruct-v0.3

**Why**: Mixture of Experts architecture (different from dense transformers)  
**Expected**: If Phase 5b works, validates recipe on MoE architectures  
**Script**: Copy `test_llama3_recipe.py` and change model ID

### Phi-3-mini-4k-instruct

**Why**: Microsoft's small model (3.8B, different architecture from Llama/Qwen)  
**Expected**: Validates on non-Meta, non-Alibaba architecture  
**Script**: Copy `test_llama3_recipe.py` and change model ID

### Gemma-2-9B

**Why**: Google's model family (different architectural choices)  
**Expected**: Another data point for cross-family reproducibility  
**Script**: Copy `test_llama3_recipe.py` and change model ID

## Files

- `kaggle/awareliquid_train_llama3_phase5b.py` — Full training notebook for Llama-3-8B
- `scripts/test_llama3_recipe.py` — CPU sanity check (no training, just verification)

## Timeline

1. **Immediate**: Run CPU sanity check on Llama-3-8B (5 min)
2. **Short term**: Kaggle training on Llama-3-8B (~4h)
3. **Medium term**: Mistral-7B and Phi-3-mini tests (~4h each)
4. **Long term**: Systematic cross-family benchmark suite

## Cross-Family Summary (When Complete)

| Family | Models Tested | PPL Range | Status |
|---|---|---|---|
| Llama-1/2 | TinyLlama-1.1B | -28.5% | ✅ |
| Qwen-2.5 | 1.5B, 3B | -27.7%, -34.4% | ✅ |
| Llama-3 | 8B | TBD | ⏳ |
| Mistral | 7B-Instruct | TBD | 🔲 |
| Phi-3 | mini-4k-instruct | TBD | 🔲 |
| Gemma-2 | 9B | TBD | 🔲 |

**Goal**: Validate Phase 5b recipe on ≥4 model families to claim **universal cross-architecture reproducibility**.
