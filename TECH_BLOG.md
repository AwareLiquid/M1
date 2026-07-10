# Cross-Architecture Reproducibility with MT-LNN Adapters: From 1.1B to 3B

> **⚠️ CORRECTION (2026-07-05) — supersedes the adapter results below.**
> The Phase 5/5b adapter numbers quoted in this document (−28.5 %/−27.7 %/−34.4 % PPL
> at "0.196 %/0.139 %/0.117 % trainable") are **retracted**: those runs predate the
> re-arm fix (`8d9d741`) — PEFT had silently frozen the MT adapters at random init,
> so **only LoRA trained**, and the quoted "trainable" counts are exactly the
> LoRA-only parameter counts. A controlled 6-config attribution confirms plain LoRA
> reproduces those PPL gains; the MT adapter adds ≈nothing on in-window perplexity.
> The architecture's real, reproducible differentiator is **cross-window recall
> through streaming state** (fast-weight memory: 0.62 accuracy where attention/LoRA
> are 0 by construction), delivered by the 7.5× smaller v2s adapter now serving.
> Authoritative results and protocols: **BENCHMARKS.md** (attribution, cross-window
> recall, out-of-window LM nulls, ARR distillation).


**TL;DR** *(revised 2026-07-11 — the −28…−34% PPL claim is retracted; see [RESULTS.md](RESULTS.md))*: the headline adapter-PPL numbers below were **LoRA-only** (the MT adapter was frozen by PEFT and adds ≈0 PPL beyond LoRA). What actually replicates: a **from-scratch native MT-LNN at 125M beats a matched Transformer by −31% val PPL** (299 vs 436, stable at scale), and the streaming fast-weight state gives **cross-window associative recall (0.56) that attention/LoRA score 0.000 on by construction**, plus genuine **O(1) inference memory** in the attention-free O-series (1008× smaller than a KV cache at 128k).

---

## The Reproducibility Problem in Adapter Research

When a paper reports impressive results on a specific model, the first question is always: *Does it generalize?* Adapter methods often suffer from:

- **Hyperparameter brittleness**: Different learning rates, warmup schedules, or initialization schemes for each base
- **Architecture coupling**: Designs that exploit quirks of specific model families
- **Scale sensitivity**: Methods that work at 1B but fail at 7B (or vice versa)

We wanted to test whether a biologically-inspired adapter architecture could achieve **true cross-architecture reproducibility**: the *exact same recipe* working across model families and scales.

## The Recipe: Phase 5b

**Configuration**:
- **MT residual adapters** every 4th decoder layer
  - 13 parallel protofilaments (inspired by microtubule structure)
  - 5 temporal integration time scales
  - Pre-norm + residual with init_scale=1e-3
- **LoRA** on attention projections (q, k, v, o)
  - Rank 8, alpha 16, dropout 0.05
- **Training**: 1000 steps, batch 1, grad_accum 8, WikiText-2
- **Optimizer**: AdamW (lr=2e-4, weight_decay=0.01)

**That's it.** No per-model tuning. No architecture-specific adjustments.

## Results: Three Bases, Three Wins

| Base Model | Size | PPL (base) | PPL (+adapter) | Δ PPL | Trainable % |
|---|---:|---:|---:|---:|---:|
| TinyLlama-1.1B-Chat | 1.1B | 22.49 | **16.08** | **-28.5%** | 0.196% |
| Qwen-2.5-1.5B-Instruct | 1.5B | 15.18 | **10.98** | **-27.7%** | 0.139% |
| Qwen-2.5-3B-Instruct | 3B | 10.72 | **7.03** | **-34.4%** | 0.117% |

**Key observations**:

1. **Consistent improvement**: -28% to -34% band across all three bases
2. **Cross-family**: Works on both Llama (GPT-NeoX-style) and Qwen (Qwen2-style) architectures
3. **Positive scaling**: Improvement *increases* with model size (1.1B: -28.5% → 3B: -34.4%)
4. **Ultra-low overhead**: <0.2% trainable parameters (3-4M params total)

## Why Microtubules?

The MT-LNN (Microtubule Liquid Neural Network) architecture is inspired by neuronal microtubules—the cytoskeletal structures hypothesized to play a role in neural computation and consciousness (Penrose-Hameroff Orch OR theory).

**Key architectural features**:

1. **13 Parallel Protofilaments**: Biological microtubules have 13 protofilaments arranged in a hollow cylinder. Our adapters use 13 parallel processing streams with lateral coupling.

2. **Temporal Integration**: Multi-timescale dynamics for integrating information across different temporal windows (inspired by GTP-cap propagation along microtubules).

3. **Residual Bypass**: Small initialization scale (1e-3) ensures adapters start as near-identity, preserving base model capabilities while gradually learning long-context patterns.

**Hypothesis**: Microtubule-inspired parallel processing with recurrent dynamics provides a strong inductive bias for long-range temporal dependencies—exactly what language models need for long-context understanding.

## Ablation Studies (In Progress)

To understand *what* drives the improvement, we designed four ablation groups:

### 1. Adapter Type (Critical Test)

| Config | Components | Hypothesis |
|---|---|---|
| MT-only | MT adapters | Architecture provides long-context bias |
| LoRA-only | Vanilla LoRA | Standard parameter-efficient fine-tuning |
| MT+LoRA | Both | Complementary benefits |

**If MT-only >> LoRA-only**, the microtubule architecture (not just parameter efficiency) is the key innovation.

### 2. Layer Interval

| Config | Coverage | Hypothesis |
|---|---|---|
| every 2 | Dense | More coverage helps (or just more params) |
| every 4 | Phase 5b | Sweet spot |
| every 8 | Sparse | Sufficient if inductive bias is strong |

Tests whether adapter *density* matters or if architectural prior is sufficient.

### 3. LoRA Rank

| Config | Capacity | Hypothesis |
|---|---|---|
| r=4 | Low | Sufficient if MT does heavy lifting |
| r=8 | Medium | Phase 5b default |
| r=16 | High | Overkill if architecture is the driver |

Tests whether it's about LoRA *capacity* or MT *architecture*.

### 4. Protofilaments

| Config | Count | Hypothesis |
|---|---|---|
| 8 | Sub-biological | Biological prior matters |
| 13 | Biological | From microtubule structure |
| 21 | Super-biological | 13 is optimal, not "more is better" |

Tests whether the biological count (13) is meaningful or coincidental.

**Status**: Infrastructure complete (`scripts/run_ablations.py`), experiments pending GPU time.

## Practical Implications

### For Researchers

**Reproducible baseline**: The Phase 5b recipe provides a strong, reproducible baseline for long-context adapter research. No per-model hyperparameter search needed.

**Ablation framework**: Our ablation suite (`ABLATIONS.md`) makes it trivial to test variations and understand architectural contributions.

**Cross-family validation**: If your method only works on one model family, it may be exploiting architecture-specific quirks rather than capturing fundamental principles.

### For Practitioners

**Plug-and-play**: One function call to apply the recipe to any HuggingFace causal LM:

```python
from transformers import AutoModelForCausalLM
from mt_lnn.recipes import apply_phase5b_recipe

model = AutoModelForCausalLM.from_pretrained("your-model-here")
result = apply_phase5b_recipe(model)
# Train as usual - only adapter params are trainable
```

**Low-resource friendly**: <0.2% trainable params means you can fine-tune 3B models on consumer GPUs.

**No architecture changes**: Attaches to frozen base models as residual adapters. Compatible with existing training pipelines, PEFT, and HuggingFace Trainer.

## Open Questions

1. **Does it scale to 7B+ models?** Results suggest positive scaling (3B > 1.5B > 1.1B). Testing on Llama-3-8B or Qwen-7B is the natural next step.

2. **Long-context benchmarks**: PPL improvements are clear, but do they translate to needle-in-a-haystack, RULER, or LongBench scores? (Needle harness recently fixed; experiments pending.)

3. **What drives the improvement?** Ablations will tell us whether it's the MT architecture, LoRA capacity, or their combination.

4. **Other modalities**: Microtubule-inspired architecture may benefit vision transformers (ViT) or multimodal models (CLIP, Flamingo) for temporal reasoning.

## Reproducing Results

**Quickstart**:
```bash
# Clone repo
git clone https://github.com/everest-an/M1.git
cd M1

# Train Qwen-1.5B + Phase 5b (Kaggle T4, ~90 min)
python train_llama_mt_adapter.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --steps 1000 --batch 1 --grad_accum 8 \
    --mt_every 4 --lora --lora_targets q_proj,k_proj,v_proj,o_proj

# Evaluate PPL
python scripts/bench_llama_mt_ppl.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter checkpoints/llama_mt_adapter/llama_mt_adapter_001000.pt
```

**Kaggle notebooks**:
- `kaggle/awareliquid_train_qwen_phase5b.ipynb` (Qwen-1.5B)
- `kaggle/awareliquid_train_qwen3b.ipynb` (Qwen-3B)

**Expected wall time**: ~90 min on Kaggle T4 for 1000 steps.

## Acknowledgments

This work builds on:
- **Penrose-Hameroff Orch OR**: Microtubule quantum computation hypothesis
- **Liquid Neural Networks** (Hasani et al.): Continuous-time recurrent dynamics
- **LoRA** (Hu et al.): Parameter-efficient fine-tuning
- **TinyLlama**, **Qwen teams**: Open base models for validation

## Code & Resources

- **GitHub**: [everest-an/M1](https://github.com/everest-an/M1)
- **Recipes module**: `mt_lnn/recipes.py` (one-line apply)
- **Ablation suite**: `ABLATIONS.md` + `scripts/run_ablations.py`
- **Documentation**: `RECIPES.md`, `BENCHMARKS.md`, `PRD.md`

## What's Next?

1. **Run ablations** to understand architectural contributions (MT vs LoRA vs both)
2. **Test on Llama-3 or Mistral** for third model family validation
3. **Long-context benchmarks** (needle, RULER) with fixed harness
4. **Scale to 7B+** to validate positive scaling trend
5. **Explore other domains**: Vision transformers, multimodal, code models

---

**Bottom line**: If you're working on long-context LM adapters, the Phase 5b recipe gives you a reproducible starting point that works across architectures and scales. The microtubule-inspired design suggests biology may offer architectural priors for temporal reasoning that generalize beyond specific neural architectures.

*Questions? Issues? PRs welcome at [github.com/everest-an/M1](https://github.com/everest-an/M1)*

---

**Date**: 2026-05-31  
**Version**: v1.0 (Track 1 results)  
**License**: MIT
