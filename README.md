<div align="center">

# MT-LNN

### Microtubule-Inspired Liquid Neural Network

**Brain-inspired LLM architecture with $O(1)$ working memory, multi-scale predictive coding, and dynamic compute skipping.**

[![Stars](https://img.shields.io/github/stars/everest-an/M1?style=flat&color=1f75fe)](https://github.com/everest-an/M1)
[![Paper EN](https://img.shields.io/badge/PDF-EN-red)](https://huggingface.co/EverestAn/MT-LNN/resolve/main/mt_lnn_arxiv.pdf)
[![Paper ZH](https://img.shields.io/badge/PDF-ZH-red)](https://huggingface.co/EverestAn/MT-LNN/resolve/main/mt_lnn_arxiv_zh.pdf)
[![HF Model](https://img.shields.io/badge/HF-MT--LNN-yellow)](https://huggingface.co/EverestAn/MT-LNN)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## What this repo is

MT-LNN replaces a Transformer block's FFN with a recurrent **Microtubule Liquid Neural Network** layer (13 protofilaments × 5 timescales, continuous-time LTC ODE), and adds two cross-layer modules: a **GWTB** workspace bottleneck (Global Workspace Theory) and a **Global Coherence** sparse top-$k$ collapse gate (Orch-OR inspired).

The architecture targets three pain points of modern LLMs:

| Pain point | Standard LLM | MT-LNN |
|---|---|---|
| KV-cache memory | $O(T)$, blows up at long context | $O(1)$ recurrent state (`h_prev`) |
| Wasted compute | All neurons fire on every token | Dynamic $\kappa$-gating + LAVI rhythm gating skips idle channels |
| No world-model loss | Pure next-token prediction | Optional BYOL/V-JEPA predictive-state head with stop-grad EMA target |

Everything ships behind config flags. Defaults reproduce the legacy MT-LNN forward pass; opt-in flags enable the v2.0/v2.1 brain-inspired modules without changing the main forward signature.

---

## Track 1 results (v1.0.0, 2026-05-30)

Cross-base universal PPL uplift on WikiText-2-raw-v1 with 0.1–0.2 % trainable params (frozen base + MT residual adapter every 4th layer + LoRA on q/k/v/o):

| Base LM         | Trainable | PPL drop | Status |
|---              |---:       |---:      |:---:   |
| TinyLlama-1.1B  | 0.196 %   | −28.5 %  | ✅     |
| Qwen-2.5-1.5B   | 0.139 %   | −27.7 %  | ✅     |
| **Qwen-2.5-3B** | **0.117 %** | **−34.4 %** | ✅ |

Same recipe transfers across Llama and Qwen families; PPL improvement grows with base size. Real $O(N)$ generation with `past_key_values` is implemented in `scripts/awareliquid_real_trace_v3.py`. Raw artifacts in `benchmarks/kaggle_{run,qwen_run,qwen3b_run}/`. Tag: `v1.0.0-track1-ppl34`.

Selective Copy at matched ~200K params (training-from-scratch ablation):

| Model                   | #Params | Held-out tok-acc | **Held-out seq-exact** |
|---                      |---:     |---:              |---:                    |
| Random                  | —       | 0.250            | 0.004                  |
| Vanilla Transformer     | 199 K   | 0.432            | 0.023                  |
| LNN (CfLTC FFN)         | 136 K   | 0.433            | 0.023                  |
| **MT-LNN (full arch)**  | 204 K   | **0.983**        | **0.965** (×42 over Transformer) |

Long-context advantage grows with $T$: at $T{=}101$, MT-LNN seq-exact is 34× the Transformer baseline. Full table in [BENCHMARKS.md](BENCHMARKS.md).

---

## Quick start

### Install

```bash
git clone https://github.com/everest-an/M1.git && cd M1
pip install -r requirements.txt
```

### Apply MT-LNN to any HuggingFace causal LM (recommended)

```python
from transformers import AutoModelForCausalLM
from mt_lnn.recipes import apply_phase5b_recipe

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
result = apply_phase5b_recipe(model)   # MT residual adapter every 4th layer + LoRA on q/k/v/o
# Now train as usual; only ~0.1-0.2% of params are trainable.
```

Available recipes in `mt_lnn.recipes`:

| Function | What it does |
|---|---|
| `apply_phase5b_recipe(model)` | MT every 4th layer + LoRA on q/k/v/o (default; what produced Track 1 results) |
| `apply_mt_only_recipe(model)` | MT adapters only, no LoRA |
| `apply_lora_only_recipe(model)` | LoRA only baseline |

### Train an adapter end-to-end on a base LM

```bash
python train_llama_mt_adapter.py \
    --model meta-llama/Llama-3.2-1B \
    --dataset wikitext --dataset_config wikitext-2-raw-v1 \
    --seq_len 512 --batch 1 --grad_accum 8 --steps 1000 \
    --mt_every 4 --lora
```

Adapter checkpoints are saved under `checkpoints/llama_mt_adapter/` (separate from the frozen base).

Eval and ablation:

```bash
python eval_llama_mt_adapter.py --model meta-llama/Llama-3.2-1B \
    --adapter checkpoints/llama_mt_adapter/llama_mt_adapter_001000.pt

python bench_llama_mt_ablation.py --model meta-llama/Llama-3.2-1B \
    --adapters checkpoints/llama_mt_adapter/llama_mt_adapter_001000.pt \
    --max_batches 50 --out_json benchmarks/llama_mt_ablation.json

python bench_llama_mt_needle.py --model meta-llama/Llama-3.2-1B \
    --adapters checkpoints/llama_mt_adapter/llama_mt_adapter_001000.pt \
    --context_lengths 1024 2048 4096 --depths 0.1 0.5 0.9 --samples 5
```

### Train MT-LNN from scratch (~125M)

```bash
python prepare_data.py                         # tokenize WikiText-103 to memmap binaries
python train.py --compile --wandb              # default: d_model=832 (=13×64), 12 layers
python demo.py --ckpt checkpoints/final.pt --prompt "The human brain"
```

Smoke test on dummy data (no dataset download):

```bash
python train.py --d_model 128 --n_layers 2 --n_heads 4 --n_kv_heads 2 \
                --batch 2 --seq_len 32 --steps 200 --dummy --vocab_size 200
```

### Reproduce the toy benchmarks

```bash
python benchmarks/compare_baselines.py    # Selective Copy head-to-head
python benchmarks/long_context.py         # T=37 / 101 / 229 sweep
python benchmarks/run_benchmark.py        # full benchmark suite
```

---

## Architecture

> Full design doc: [ARCHITECTURE.md](ARCHITECTURE.md). Visual walkthrough: [MT_LNN_ARCHITECTURE_VISUAL.md](MT_LNN_ARCHITECTURE_VISUAL.md). 3D interactive viewer: [llm-viz-QUICKSTART.md](llm-viz-QUICKSTART.md).

```
input_ids
   ↓
Token Embedding + RoPE
   ↓
─── × n_layers ──────────────────────────────────────────────────
MTLNNBlock  (pre-norm + residual at each sub-layer)
  • MicrotubuleAttention             [GQA, KV cache, SDPA / Flash-Attn]
       scalar polarity bias  +  GTP-cap ALiBi log-bias
       opt-in: low-rank bilinear polarity  σ(xWa)(xWb)ᵀ
  • MTLNNLayer                       [recurrent h_prev cache, parallel scan]
       d_model → 13 protofilaments  (d_proto = d_model/13; exact at 832)
       13 × 5-scale MultiScaleResonance  (geometric τ sweep, softmax blend)
         κ-gate         content-based scale activation
         LAVI rhythm    history-based slow/fast τ blend (use_rhythm=True)
       LateralCoupling (3-way):
         static W_lat (13×13, identity init)
         + nearest-neighbor torch.roll (ring topology)
         + RMC content-aware attention (σ(rmc_gate) ≈ 0.05 init)
         → all gated by exp(-γ · (t mod T_period))   [GTP-cap renewal]
       MAPGate (per-protofilament, fc2_bias=+2 → near-open at init)
       → d_model
  • [GWTBLayer / CompetitiveGWTBLayer]  optional per-block workspace
──────────────────────────────────────────────────────────────────
   ↓
[GlobalRhythmController]    aggregate per-layer LAVI → residual correction
   ↓
GWTBLayer or CompetitiveGWTBLayer
    compress d_model → d_gw → workspace SA → broadcast + γ·residual
    (CompetitiveGWTBLayer adds multi-source bidding from lnn / attn / coherence)
   ↓
GlobalCoherenceLayer        sparse top-k + Orch-OR collapse gate
   ↓
[PredictiveStateHead]       BYOL/V-JEPA online predictor vs EMA target (stop-grad)
   ↓
LayerNorm → lm_head (weight-tied)
   ↓
logits
```

### Two inference modes

| Mode | LNN behavior | Use case |
|---|---|---|
| `use_lnn_recurrence=False` | `h_prev = 0` each step (parallel) | Bit-exact match with full forward; matches training-time semantics |
| `use_lnn_recurrence=True` *(default at inference)* | `h_prev` threaded across steps | True RNN-style microtubule state accumulation |

The `ModelCacheStruct` carries per-layer `(attention KV, LNN h_prev, per-block GWTB KV)`, plus top-level GWTB and coherence KV caches.

### Opt-in v2.0 / v2.1 modules

All four default **OFF**, add zero overhead when disabled, and never change the `mt_lnn_layer.py` forward signature.

| Module | Flag | Biological prior | Mechanism |
|---|---|---|---|
| **Phase A** — `CompetitiveGWTBLayer` | `use_competitive_gwtb` | GWT (Baars / Dehaene): conscious content wins by competition | Multi-source bids → score → winner broadcast; `gwtb_competition_entropy` guards routing collapse |
| **Phase B** — `CausalConsistencyChecker` | inference object | PFC predictive-error monitoring | `cosine` or anisotropy-robust `subspace`-residual novelty → forced SELF_CRITIQUE below threshold |
| **Phase B+** — `CausalActivationSteerer` | inference object | error-driven re-stabilisation | STARS-inspired: on a detected break, orthogonally project the drifting state back onto the legal causal subspace (reuses the checker's `principal_subspace()` — no duplicated SVD) |
| **Phase C** — `PredictiveStateHead` | `use_world_model` | Predictive coding / Friston free energy | BYOL/V-JEPA online predictor + stop-grad EMA target (collapse-free); normalised surprise ∈ [0,1] feeds LAVI |
| **Phase D** — `HebbianRegularizer` | `use_hebbian` | Hebbian consolidation | LAVI-gated co-activation loss (training only) |

> Phase C `use_ema_target=False` is a SimSiam variant — provably collapse-free on its own (verified across 3 seeds, |cos| ≈ 0.33 vs naïve 1.000). EMA aids convergence, not collapse-prevention. Full analysis in [V2_REVIEW.md](V2_REVIEW.md) §8.

### EEG-inspired Rhythm Gate (2026-06-06)

Brain cortex maintains two oscillatory modes — **persistent** (theta/alpha, stable context) and **transient** (gamma bursts, rapid switching). MT-LNN implements this via the **LAVI** (Lag Angle Vector Index) estimator: per-protofilament cosine similarity between current input and `h_prev` shifts the τ-scale blend.

| Signal | Existing κ-gate | New LAVI rhythm gate |
|---|---|---|
| Source | Current input content | h_prev vs current input similarity |
| Effect | Which τ scales are active | How much to weight slow vs fast τ |

Enable:

```python
from mt_lnn.config import MTLNNConfig
cfg = MTLNNConfig(use_rhythm=True, rhythm_scale_init=0.1, global_rhythm=True)
```

Diagnostics surfaced via `model.get_mt_diagnostics()`: `lavi_mean / min / max`, `rhythm_scale_mean`, `global_rhythm_scale`.

### Operator Compression (state-only streaming)

Drop historical KV tensors during decode and keep only the recurrent `h_prev`:

- 1000 tokens, traditional KV stream: ~1020 KB → MT-LNN state-only: **4.1 KB**.
- Aimed at edge / always-on inference where context length is bounded by the recurrent state, not by KV memory.

### Sparse Resonance (top-$k$ scale routing)

The 5 timescales per protofilament are gated to top-$k$ at decode. Ablation: top-$k{=}2$ matches dense quality and lifts CPU single-batch throughput from ~3650 to ~6400 tok/s. See [`benchmarks/sparse_resonance_ablation.md`](benchmarks/sparse_resonance_ablation.md).

### Anesthesia Validation Protocol (AVP)

At inference, hooks progressively damp MT-DL outputs and the global-coherence broadcast by `(1 − level)` as `level` rises 0 → 1. We measure **Φ̂** (Kraskov kNN proxy for integrated information). Hooks attach only to `MTLNNLayer` and `GlobalCoherenceLayer`, so the baselines' Δ Φ̂ is exactly 0 by construction:

| Model       | Φ̂(κ=1)   | Φ̂(κ=10)  | Δ Φ̂                        |
|---          |---:       |---:       |---:                          |
| Transformer | -9.045    | -9.045    | 0.000 (no hooks)             |
| LNN         | -7.977    | -7.977    | 0.000 (no hooks)             |
| **MT-LNN**  | -18.673   | -11.096   | **+7.578 (responsive)**      |

> ⚠️ At ~200K toy scale, Δ Φ̂ sign is inverted vs. paper prediction. Architectural *responsiveness* is real; *direction* is expected to flip after 125M-scale training. See [BENCHMARKS.md](BENCHMARKS.md) §AVP.

---

## Optional research modules

Both off by default; package imports cleanly without their dependencies.

- **`mt_lnn.phi_iit`** — exact IIT 4.0 Φ via PyPhi (Tononi lab toolbox). `pip install pyphi`. Use for ≤8-node analysis; the kNN proxy `phi_hat` covers training-time monitoring.
- **`mt_lnn.quantum_coupling.QuantumLateralCoupling`** — drop-in replacement for `LateralCoupling`: P qubits in a ring with parameterised CNOT entanglers (mod-P), classical simulator default. `pip install pennylane`.

---

## File map

```
mt_lnn/
  config.py              MTLNNConfig — single source of truth for all hparams
  embedding.py           TokenEmbedding + RoPE (offset-aware)
  mt_attention.py        MicrotubuleAttention — GQA, KV cache, scalar+low-rank polarity
  mt_lnn_layer.py        MTLNNLayer, MultiScaleResonance, LateralCoupling, MAPGate
                         (fully vectorised over P; LAVI rhythm hook in scale gate)
  rhythm.py              LAVIEstimator + GlobalRhythmController
  parallel_scan.py       Blelloch / Mamba-style pscan (true recurrence on GPU)
  gwtb.py                GWTBLayer, CompetitiveGWTBLayer (Phase A multi-source bid)
  global_coherence.py    sparse top-k + Orch-OR collapse gate
  causality.py           CausalConsistencyChecker (Phase B; cosine + subspace methods; principal_subspace())
  causal_steering.py     CausalActivationSteerer (Phase B+; STARS-inspired subspace projection)
  world_model.py         PredictiveStateHead (Phase C; BYOL/V-JEPA EMA target)
  plasticity.py          HebbianRegularizer (Phase D; LAVI-gated consolidation)
  deliberation.py        DeliberationRouter (entropy 3-way + causal-consistency floor)
  router.py              DeliberationRouter mode plumbing
  reasoning_trace.py     Per-step trace logging for audit / replay
  capsule.py             SessionMemory serialization (.capsule format)
  session_state.py       Multi-turn session orchestration
  cloud_client.py        Awareness Cloud API client (factual blind-spot fallback)
  llama_adapter.py       Frozen-base + MT residual adapter for HF causal LMs
  recipes.py             apply_{phase5b, mt_only, lora_only}_recipe(model)
  streaming.py           streaming_inference, prefill_state_only
  observability.py       JsonlMetricWriter, cache_summary, v2 metric records
  anesthesia.py          AnesthesiaController + runtime AVP hooks
  phi_hat.py             Kraskov kNN Φ̂ proxy + anesthesia sweep
  phi_iit.py             Exact IIT 4.0 Φ (PyPhi); optional
  phi_spectral.py        Spectral Φ approximation
  quantum_coupling.py    QuantumLateralCoupling (PennyLane); optional
  multimodal.py          Multi-modal token codebook hooks
  spatial.py             Spatial frontends: GridCellEncoding, PlaceCellCode (DoG target), PointCloud/Voxel
  spatial_reasoning.py   SpatialReasoner (perception + deliberation; optional causal checker/steerer)
  spatial_memory.py      SpatialMemory (L2; place-indexed associative memory, Hebbian write / pattern-completion read, 0 params)
  memory.py              SessionMemory primitive
  meta_learning.py       Meta-learning helpers
  awareliquid_daemon.py  Long-running inference daemon
  export.py              ONNX / state-only export utilities
  utils.py               init_mt_params, schedulers, checkpointing, param groups
  model.py               MTLNNBlock + MTLNNModel + ModelCacheStruct (dual+GWTB cache)

prepare_data.py            Tokenise to uint16 .bin (numpy.memmap-friendly)
train.py                   From-scratch trainer (AMP, torch.compile, W&B)
train_llama_mt_adapter.py  Frozen-base + MT-adapter trainer for HF causal LMs
eval.py                    PPL, sliding-window long-context PPL, AVP CLI
eval_llama_mt_adapter.py   Adapter-only eval against base
demo.py                    KV-cached autoregressive streaming generation
demo_llama_mt_adapter.py   Streaming demo for adapter checkpoints
demo_mvp_loop.py           AwareLiquid full multi-turn loop with cloud fallback
demo_awareliquid_v2.py     v2 mode demo (CompetitiveGWTB + PredictiveState + Hebbian)
examples/demo_causal_spatial_steering.py  L3 causal spatial steering: a teleport
                           knocks the belief off its legal manifold; the steerer
                           detects the break and projects it back (before/after + --plot)

bench_llama_mt_ablation.py        One-shot ablation table over checkpoints
bench_llama_mt_needle.py          Needle-in-a-haystack retrieval benchmark
benchmarks/compare_baselines.py   Selective Copy head-to-head (toy scale)
benchmarks/long_context.py        T=37/101/229 sweep
benchmarks/run_benchmark.py       Full benchmark suite

kaggle/                    Cloud-ready notebooks (Qwen-1.5B, Qwen-3B, ablations)
scripts/                   Real-trace v3 (KV-cache O(N)) + cloud-inject helpers
tests/                     Full test suite (219 tests, all pass)
assets/                    decks/ (investor + paper), figures/ (architecture diagrams)
```

---

## Status

Research-grade code. All 219 tests pass (model · rhythm · causality · world-model · observability · GWTB · coherence · AVP). Highlights:

```
[ok] test_kv_cache_parity                 cached vs full diff < 1e-4
[ok] test_lnn_recurrence_active           h_prev verifiably flows
[ok] test_gwtb_cache_parity               GWTB cached vs full diff < 1e-4
[ok] test_anesthesia_validation_protocol  Φ̂ collapses monotonically with κ
[ok] test_protofilament_scaling           P=64 only ~1.2× slower than P=13
[ok] test_lavi_persistent_higher_for_similar_input
[ok] test_global_rhythm_identity_at_init  scale=0 → output = input exactly
[ok] test_model_no_regression_rhythm_off  use_rhythm=False: zero output impact
```

What's validated:

- ✅ Architectural priors (13 protofilaments, GTP renewal, parallel-scan recurrence, RMC, GWTB) yield ×42 advantage on long-range selective tasks at matched 200K params, growing with $T$.
- ✅ MT-residual adapter transfers across Llama and Qwen bases at 0.1–0.2 % trainable params, with PPL improvement that scales positively (−28 % at 1.1B → −34 % at 3B).
- ✅ Real $O(N)$ generation with `past_key_values`; state-only streaming reduces 1000-token decode footprint from ~1020 KB → 4.1 KB.

What's not yet shown (by design):

- ❌ Generic LM benchmarks (MMLU, HellaSwag, full WikiText-103 PPL) — requires 125M+ from-scratch training run; listed as future work.
- ⚠️ Δ Φ̂ direction at toy scale is inverted vs. paper prediction; expected to flip after 125M training.

---

## Documentation

| Doc | What's in it |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Full v2.0 architecture spec |
| [MT_LNN_ARCHITECTURE_VISUAL.md](MT_LNN_ARCHITECTURE_VISUAL.md) | Visual walkthrough |
| [BENCHMARKS.md](BENCHMARKS.md) | All benchmark numbers + AVP details |
| [RECIPES.md](RECIPES.md) | Recipe API reference |
| [ABLATIONS.md](ABLATIONS.md) | Ablation framework |
| [V2_REVIEW.md](V2_REVIEW.md) | v2.0/v2.1 module review (incl. SimSiam collapse proof) |
| [TECH_BLOG.md](TECH_BLOG.md) | Cross-architecture reproducibility story |
| [PRD.md](PRD.md) | Product requirements & roadmap |
| [SPEC.md](SPEC.md) | Component spec |
| [NEEDLE_FIX.md](NEEDLE_FIX.md) | Fixed needle-in-a-haystack harness |
| [CLOUD_RUN.md](CLOUD_RUN.md) / [KAGGLE_RUN.md](KAGGLE_RUN.md) | Cloud / Kaggle reproduction |
| [llm-viz-QUICKSTART.md](llm-viz-QUICKSTART.md) | 3D interactive architecture viewer |

---

## Design references

- **Closed-form LTC** — Hasani et al., *Closed-form continuous-time neural networks*, Nature MI 2022
- **Liquid Foundation Models** — Liquid AI LFM2 / LFM2.5 (2025–2026)
- **Orch-OR** — Penrose & Hameroff; experimental support: Wiest, *Neuroscience of Consciousness*, Oxford Academic, 2025
- **GWT** — Baars (1988); Dehaene & Changeux, *Neuron* 2011
- **Predictive coding** — Friston, *free energy principle*
- **GQA** — Ainslie et al., EMNLP 2023
- **RoPE** — Su et al., *RoFormer*

---

## License

MIT — see [LICENSE](LICENSE).
