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

> **Note — "M2" also names a document-QA adapter, not this architecture.**
> A separate repo, [`AwareLiquid/AwareLiquid-M2`](https://github.com/AwareLiquid/AwareLiquid-M2),
> is a retrieval + compression **adapter** that wraps a *frozen* API model for
> financial-document QA. Its ~91.7% eval score is the adapter's, not this MT-LNN
> architecture's — it trains no weights and shares no code with this repo. The
> "M2" in this project (`m2_final.pt`, the kaggle m2 pretrain kernel) refers to
> this architecture's 125M line.

**A streaming-state recurrent LLM architecture with two proven, hard-to-replicate results: cross-window associative recall that attention cannot express, and (in its attention-free variant) genuine O(1) inference memory.**

> **Evidence policy.** Every number in this README is backed by a reproducible table in [BENCHMARKS.md](BENCHMARKS.md) and reconciled in [RESULTS.md](RESULTS.md). RESULTS.md is the source of truth; if any other doc disagrees, RESULTS.md wins. Earlier "−28–34% adapter PPL" and consciousness/Φ̂ claims have been **retracted or reclassified as inspiration** — see [What is retracted](#what-is-retracted).

## What is proven

Results that survive convergence and a modern baseline:

1. **MT-LNN trains stably at 125M — but a modern Transformer still leads on language-modeling quality.** At **convergence** (20,000 steps, **3 seeds**, fp32, WikiText-103): MT-LNN reaches **88.93 ± 0.33** val PPL, beating the simple matched Transformer (**94.14 ± 0.78**, −5.5%) — but a **modern Transformer baseline (RoPE + RMSNorm + SwiGLU) reaches 78.86 ± 0.25, i.e. 11.3% *better* than MT-LNN**. All architectures train stably with no NaN, which answers the original "does a liquid-recurrent net converge when scaled 100×?" question. ⚠️ **An earlier 2,000-step run reported MT-LNN ahead by ~31% (257 vs 371). That gap did not survive**: it was an artifact of **undertraining plus a weak reference baseline**, and both effects vanish at convergence against a modern baseline. **Perplexity is therefore not currently an MT-LNN advantage** — the architecture's case rests on the memory/efficiency results below.

2. **Cross-window / cross-session associative recall that attention and LoRA get 0.000 on by construction.** The fast-weight state stores discrete key→value bindings across a **dropped KV cache**: **0.56 mean recall** (3-seed 0.62 / 0.43 / 0.62) where frozen attention and LoRA are structurally zero. Remove the fast-weight matrix and it collapses to 0.008 — the fast weight *is* the memory. Snapshotting that state to disk and restoring into a fresh process is **bit-exact lossless** — what turns "recall within a session" into "remembers you across sessions."

And, in the attention-free line:

3. **O(1) inference memory (O-series / ARR only).** Every attention block replaced by a recurrent mixer → carried state **flat at 0.381 MB** regardless of context, versus an O(T) KV cache. Measured at 125M scale across a **2048× context increase (512 → 1,048,576 tokens)** with the state unchanged to the decimal: **1008× smaller at 128k, 8063× at 1M** (where the KV cache alone would be 3 GB). This is the architecture's strongest surviving claim.

| Result | Number | Caveat |
|---|---|---|
| **LM quality at convergence** (20K steps, n=3, fp32) | modern Transformer **78.86 ± 0.25** < MT-LNN **88.93 ± 0.33** < simple Transformer **94.14 ± 0.78** | **MT-LNN loses to a modern baseline by 11.3%**; beats only the weak one (−5.5%) |
| ~~Native 125M "−31% vs Transformer"~~ (2K steps) | **Retracted as a headline claim** — undertrained + weak baseline; does not survive 20K convergence | superseded by the row above |
| Native 125M vs Mamba | −37.8% val PPL at 2K steps (257 vs 414) | **2K-step only, not re-run at convergence**; width/depth-mismatched external baseline |
| Cross-window recall (fast-weight) | 0.56 vs **0.000** (attention/LoRA) | discrete K→V bindings, not long-context LM |
| Cross-session snapshot/restore | bit-exact, controls at chance | recall task is high-variance |
| O(1) inference state (**O-series only**) | 0.381 MB flat → 1008× @128k → **8063× @1M** | attention-free ARR only, not the hybrid; inference carried-state, not training memory |

## Honest pain-point framing

| Transformer pain point | What MT-LNN actually offers |
|---|---|
| KV-cache memory grows O(T) | **O(1) constant state — O-series (ARR) inference only.** The hybrid M-series still has attention and is **not** O(1); its *training* memory is worse than a Transformer's. |
| Attention cannot recall across a dropped context window | **Fast-weight state carries K→V bindings across windows/sessions** (0.56 vs 0.000). |
| No durable per-user memory across sessions | **Lossless (F,z) snapshot/restore** — bit-exact round-trip. |

*(MT-LNN does **not** give long-context language-modeling gains — out-of-window LM is a measured null. The state is episodic key→value memory, not compressed distributed context.)*

## What is retracted

To keep this repo credible, the following earlier headline claims are **withdrawn** (full detail in [RESULTS.md](RESULTS.md) and the BENCHMARKS.md correction notes):

- **The "−28.5% / −27.7% / −34.4% PPL at 0.1–0.2% trainable params" adapter results are retracted.** Those runs froze the MT adapter (PEFT) and trained **LoRA only**; a controlled ablation shows the MT adapter adds **≈0 PPL beyond LoRA** (7.98 vs 7.92). The "0.1–0.2% trainable" figures were the LoRA-only param counts.
- **The hybrid is not O(1) and gives no long-context LM gain.** Both are measured nulls for the M-series.
- **The Orch-OR / Φ̂ / anesthesia "consciousness" results are inert in the trained path** (AVP failed; Φ̂ sign inverted vs theory). Inspiration, not evidence.
- **The five optional bio modules are PPL-neutral at 48M.** Shipped configs run the lean core.

## Product lines

- **M-series (hybrid, attention + liquid adapter):** cloud/GPU serving at full base quality; unique edge is cross-window/cross-session recall at ~1% param overhead.
- **O-series (ARR, attention-free):** edge/streaming; unique edge is genuine O(1) inference memory. Research preview at 2.15× teacher PPL.

See [docs/PRODUCT_LINES.md](docs/PRODUCT_LINES.md).

## Inspiration (not load-bearing)

MT-LNN's design *draws on* neuroscience — neuronal microtubules and the 13-protofilament count, Global Workspace Theory, Friston's predictive coding, and the Penrose–Hameroff Orch-OR collapse hypothesis. These shaped the initial architecture (e.g. the τ timescale ladder is initialized from biological priors, which the frozen-τ ablation shows is a genuinely good *starting point*: 0.285 vs 0.621 trained recall).

**None of this is load-bearing on the results above, and we do not present it as evidence.** The Orch-OR collapse gate, the Φ̂ "integrated information" proxy, the Anesthesia Validation Protocol, and the quantum-coupling module are **inert in the trained path** — the AVP fails and the Φ̂ response is even sign-inverted versus the theory's prediction. They remain behind flags as research scaffolding and biological motivation, not a working consciousness metric or a selling point. The credible story is the benchmarked engineering: recall, cross-session persistence, O(1) O-series inference, and 125M sample efficiency.

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
python benchmarks/run_all.py              # consolidated report -> benchmarks/report.md
```

### Serve / deploy

Two REST servers expose the same API (the frontend at `serve/static/index.html`
works against either):

```bash
# Native MT-LNN decoder (no HF download; SMALL=1 builds a tiny byte model)
SMALL=1 uvicorn serve.server:app --host 0.0.0.0 --port 8000

# HF base model + MT-LNN residual adapters (default Qwen2.5-0.5B-Instruct, chat mode)
uvicorn serve.server_hf:app --host 0.0.0.0 --port 8000
```

Both servers ship as containers off one CPU image (`command:` picks the surface):

```bash
docker build -t mtlnn .                                  # CPU image (build once)
docker run --rm -p 8000:8000 -e SMALL=1 mtlnn            # smoke (no checkpoint)

# A) Native MT-LNN decoder, persistent demo on :8088
docker compose -f docker-compose.demo.yml up -d

# B) HF base + MT adapter + cognitive modules, on :8089
docker compose -f docker-compose.hf.yml up -d
```

Surface **B** (`docker-compose.hf.yml`) runs `server_hf` with the wired
cognitive modules: self-thinking decode (`THINKING=1`), declarative
knowledge memory (`KB_PATH` on a writable `kb` volume), and the
causal-consistency check (`CAUSAL_CHECK=0` by default -- honest: it is
non-discriminative on a Transformer base, see `serve/server_hf.py`). The `kb`
named volume persists `/v1/memory/write` facts across restarts and redeploys
(verified by `scripts/diagnostics/_test_kb_persist.py`); an `hf_cache` volume downloads
Qwen2.5-0.5B once, so first boot needs internet but restarts are offline.

```bash
# write a fact, then query it (memory is a strict no-op on an empty store)
curl -s localhost:8089/v1/memory/write -H 'content-type: application/json' \
     -d '{"content":"The 2026 launch codename is Project Halcyon."}'
curl -s localhost:8089/v1/completions  -H 'content-type: application/json' \
     -d '{"prompt":"What is the 2026 launch codename?","use_memory":true}'
```

`scripts/docker_autoverify.sh` builds + boots + verifies the image; the HTTP
contract for both servers is pinned by `tests/test_server.py` /
`tests/test_server_hf.py` (offline).

---

## Architecture

> Full design doc: [ARCHITECTURE.md](ARCHITECTURE.md). Visual walkthrough: [MT_LNN_ARCHITECTURE_VISUAL.md](docs/specs/MT_LNN_ARCHITECTURE_VISUAL.md). 3D interactive viewer: [llm-viz-QUICKSTART.md](llm-viz-integration/llm-viz-QUICKSTART.md).

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
| `use_lnn_recurrence=False` | `h_prev = 0` each step (parallel) | Cached decode matches full forward within `1e-4` (FP reduction order; `test_kv_cache_parity`); matches training-time semantics |
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

> Phase C `use_ema_target=False` is a SimSiam variant — provably collapse-free on its own (verified across 3 seeds, |cos| ≈ 0.33 vs naïve 1.000). EMA aids convergence, not collapse-prevention. Full analysis in [V2_REVIEW.md](docs/reviews/V2_REVIEW.md) §8.

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
- The $O(1)$ vs $O(T)$ contrast is pinned as a regression in [`tests/test_long_context_memory.py`](tests/test_long_context_memory.py): state-only cache bytes stay flat for **T = 20× the RoPE window** (320 steps over a 16-token window, i.e. 20 wraps) while the KV cache grows a constant +bytes/token. Reproduce the sweep with `python benchmarks/state_only_streaming.py --steps 512 --max_seq_len 64 --fixed_window` (8 wraps: KV 330 KB vs state-only 2.6 KB, a 127× gap).

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

Off by default; package imports cleanly without their dependencies.

- **`mt_lnn.phi_iit`** — exact IIT 4.0 Φ via PyPhi (Tononi lab toolbox). `pip install pyphi`. Use for ≤8-node analysis; the kNN proxy `phi_hat` covers training-time monitoring.

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
  causal_decoding.py     CausalDecodeSteerer (L3; generate() step_callback that steers the live recurrent cache)
  world_model.py         PredictiveStateHead (Phase C; BYOL/V-JEPA EMA target)
  imagination.py         LatentImagination (L4; rolls the world model's 1-step map forward into a multi-step imagined trajectory; 0 params, no backbone coupling)
  spatial_ops.py         Composable geometric operators (distance/direction/containment/proximity graphs/connected components/REACHABILITY; pure functions, 0 params, compose into spatial-reasoning queries)
  physics_ops.py         Composable Newtonian dynamics operators (symplectic integration — semi-implicit Euler AND 2nd-order time-reversible velocity Verlet (integrate_verlet, selectable via rollout(integrator="verlet") for O(dt^2)-bounded energy on long imagination rollouts) / gravity/N-body/collision impulse/wall reflection/conservation probes/ROLLOUT; pure functions, 0 params, compose into "what happens next" physics simulation)
  salience_events.py     SalienceEventDetector (global-workspace ignition; adaptive-baseline z-score + Schmitt hysteresis + refractory; 0-param read-only observer of world-model surprise; the dual-speed engine's wake-up tripwire)
  failsafe.py            BlindRolloutGuard (confidence-gated blind rollout: coast on the world-model imagination through input dropouts, go dark when untrusted) + CircuitBreaker (model-external output safety: unconditional NaN/bounds/slew clamp + debounced trip-to-fallback with bumpless transfer; 0-param, no model.py coupling) + TopologyBreaker (a zero-param tripwire on the SHAPE of the live representation: latch a healthy reference cloud, then per-tick watch SRTD H0-barcode drift + optional Betti-0 component count, trip/close through the same debounced FSM as CircuitBreaker; composes topology_ops, no model.py coupling)
  acoustic_ops.py        Composable binaural-hearing operators (propagation delay / 1-over-r spreading / ITD / ILD / Doppler / phasor interference; localize_azimuth inverse readout + binaural_scene composition; 0-param analytic, no model.py coupling)
  geometry_ops.py        Composable Fisher-Rao information-geometry operators on the probability simplex (the space PlaceCellCode's softmax actually emits): Bhattacharyya overlap / Fisher-Rao distance / geodesic interpolation / exp-log maps / parallel transport / Fisher metric / Karcher barycenter; pure functions, 0 params, no model.py coupling. The correct curved-manifold ruler for L2 place codes that spatial_memory currently reads with the wrong (Euclidean) metric (P0#1)
  topology_ops.py        Composable TDA operators for a state/code point cloud: minimum spanning tree / exact H0 persistent homology (the barcode = sorted MST edge weights) / Betti-0 + Betti-0 curve (reusing spatial_ops' union-find connected components) / total persistence / SRTD (Symmetric Relative-Topology Divergence, a zero-param topology-drift tripwire); pure functions, 0 params, no model.py coupling. Counts attractor/cluster structure and flags topology change (P0#2)
  stdp_ops.py            Composable spike-timing-dependent-plasticity operators: the asymmetric exponential STDP window (stdp_kernel / window_integral) + the all-to-all pairwise sum (pairwise_stdp, the definition) + the O(T) online eligibility-trace update (stdp_trace_update, how it runs on hardware) — provably equal; pure functions, 0 params, no model.py coupling. Local, BACKPROP-FREE learning from spike timing alone, PARALLEL to plasticity.py: the loss-level Hebbian governs the smooth continuous-time LTC core, event-driven STDP governs the discrete salience-ignition / L2 place-code event streams that plasticity.py explicitly says STDP does NOT fit at the core (P0 learning)
  attractor_ops.py       Composable attractor / self-stabilization operators: fixed point / spectral radius / asymptotic rate / is_contraction for a linear map (closed form) + relax rollout of any step map + empirical settling_time / convergence_rate / lyapunov_descent from an observed trajectory + basin_radius (bisection probe for the basin half-width); analyze_linear_attractor composes them and cross-checks empirical==analytic; pure functions, 0 params, no model.py coupling. Measures where the settling core converges, how fast, and how large a shock its basin absorbs (P0 learning)
  ingest_ops.py          Sensor-ingestion / stream-alignment operators: resample a jittered, timestamped sensor stream onto the core's fixed-dt grid (resample_uniform: linear exact-on-ramp / ZOH) + coverage_mask flags steps deep inside a dropout (hand off to BlindRolloutGuard to coast); align_stream flagship -> AlignedStream; closes the gap between the fixed-dt LTC discretization and a real, irregular sensor clock; 0-param analytic input-side front-end, no model.py coupling
  slow_layer.py          SlowThreatAssessor — the slow half of the dual-speed engine, woken ONLY on a salient ignition: a multi-step ballistic forecast (physics_ops.rollout + spatial_ops.in_ball) -> time-to-breach ETA + closest approach + CLEAR/WATCH/ENGAGE threat level; 0-param, no model.py coupling, paid for only when a real state change earns it
  pipeline.py            DualSpeedSentry — the commercial loop wiring every layer in its intended role: perceive (acoustic+spatial) -> predict (physics surprise) -> ignite + wake the slow layer (real multi-step threat assessment) on salience -> coast through dropout (blind rollout) -> clamp actuator (circuit breaker); orchestrator 0 new params, zero model.py coupling
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
  multimodal.py          Multi-modal token codebook hooks
  sensory_frontend.py    SensoryFrontend (P1 closed-loop): the temporal front that turns a raw, jittered, dropout-prone sensor stream into backbone-ready tokens -- composes ingest_ops.align_stream (resample onto the core's fixed-dt grid + flag dropout steps) with a trainable multimodal.ModalityProjector (project to d_model); emits a SensoryEncoding (inputs_embeds + coverage/pad mask, the trust signal a BlindRolloutGuard coasts on); trainable nn.Module, but never imports model.py (feeds the backbone via inputs_embeds)
  model.py top_down      Top-down modulation (P1 closed-loop ②): MTLNNModel.forward(top_down=...) threads a high-level goal/context vector to every block, folded in as a zero-init-gated residual (x = x + tanh(gate)*proj(LayerNorm(top_down))) after attention, before the liquid core -- cortical top-down feedback. Gated by config.use_top_down (default OFF -> no new params, bit-identical). At init gate=0 -> strict no-op (bit-exact); the gate keeps a live gradient so the model LEARNS to open and aim it. Optional config.top_down_to_gwtb additionally offers the goal as an external bid in the global-workspace competition (reuses the world-model bid pathway)
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
examples/demo_causal_decoding.py  L3 steering INSIDE real decoding: inject a belief
                           jump into the live recurrent cache during generate(); the
                           CausalDecodeSteerer hook pulls the trajectory back toward
                           baseline while leaving a healthy run untouched (--plot)
examples/demo_causal_spatial_steering.py  L3 causal spatial steering: a teleport
                           knocks the belief off its legal manifold; the steerer
                           detects the break and projects it back (before/after + --plot)
examples/demo_spatial_memory.py  L2 spatial sequence memory: walk a path writing
                           content at landmarks, then recall by a noisy place cue;
                           graceful pattern completion under positional noise (--plot)
examples/demo_imagination.py  L4 latent mental-simulation rollout: attach the
                           imagination driver to a live model's hidden state, then
                           show the composed multi-step rollout tracks the true
                           future far better than a "nothing changes" baseline
examples/demo_spatial_ops.py  Composable geometry: an agent on stepping-stones
                           split by a gap; compose distance -> radius_graph ->
                           reachable_from to compute (not memorise) whether the
                           goal is reachable for a given stride (ASCII map)
examples/demo_physics_ops.py  Composable physics: a ball launched at a wall over
                           a bouncy floor; compose gravity -> integrate ->
                           reflect_in_box to compute (not memorise) whether it
                           clears the wall; sweep restitution to flip the verdict
                           (ASCII side-view)
examples/demo_salience_events.py  Global-workspace ignition: a surprise stream
                           with a slow drift then a regime change; the detector
                           adapts to the drift and ignites only on the real
                           change, then quiesces and re-arms (ASCII timeline) --
                           the dual-speed engine's wake-up tripwire
examples/demo_failsafe.py  Two safety reflexes: (1) the input feed stutters and
                           BlindRolloutGuard coasts on the world-model imagination,
                           going dark once the dream is no longer trusted; (2) the
                           output goes NaN / out-of-bounds / slews too fast and a
                           model-external CircuitBreaker hard-clamps it into the
                           physical red-lines and trips to a safe fallback (ASCII)
examples/demo_acoustic_ops.py  Binaural hearing from geometry: (1) a drone flies
                           past a head and is localized from its ITD (bearing sweeps
                           left->right) while the Doppler falls through the rest tone
                           ("where is it, coming or going?"); (2) two coherent
                           speakers + a sliding mic show constructive/destructive
                           wavefront interference (ASCII)
examples/demo_geometry_ops.py  Fisher-Rao geometry on the simplex: take two real
                           place-cell codes (softmax outputs) and walk between them
                           two ways -- the Euclidean chord L2 currently assumes vs
                           the Fisher-Rao geodesic it should use; show they DISAGREE
                           (the chord midpoint is the wrong "halfway"), the geodesic
                           is constant-speed and its midpoint is the Karcher
                           barycenter -- making the metric inconsistency concrete
examples/demo_topology_ops.py  TDA on a state population: build 3 clean clusters,
                           show Betti-0 COMPUTES the cluster count (3) and its
                           barcode measures the inter-cluster gaps; then stress it
                           -- a benign jitter leaves the topology intact (SRTD~0)
                           while a cluster collapse drops Betti-0 and spikes SRTD
                           (a zero-param topological health-check / failsafe signal)
examples/demo_stdp_ops.py  Backprop-free STDP on discrete event streams: drive a
                           synapse with two spike trains and watch the weight move
                           from TIMING alone -- a causal stream (pre before post)
                           potentiates with zero depression, the role-swapped stream
                           depresses with zero potentiation, a coincident pair (dt=0)
                           does nothing, and the O(T) eligibility-trace update equals
                           the all-to-all pairwise window sum to machine precision
examples/demo_attractor_ops.py  Self-stabilization diagnostics: a stable linear map's
                           measured convergence rate matches the closed-form -log(rho)
                           and its Lyapunov energy descends every step; a divergent
                           map (rho>1) is flagged with a negative rate and growing
                           energy; and on the nonlinear xdot=-x+x^3 the basin probe
                           recovers the stability edge at 1.0 -- where it settles, how
                           fast, and how big a shock the basin absorbs
examples/demo_symplectic_integrator.py  Why a long imagination rollout should
                           integrate with velocity Verlet, not Euler: fly a circular
                           Kepler orbit for ~32 orbits and compare -- semi-implicit
                           Euler leaks energy and spirals outward (radius drift ~2.6e-2)
                           while 2nd-order Verlet holds the orbit (~1.3e-3, ~1600x less
                           energy drift); a fixed-time dt-halving shows Euler's position
                           error dropping ~2x (1st order) vs Verlet's ~4x (2nd order)
examples/demo_topology_failsafe.py  TopologyBreaker as a tripwire on the SHAPE of the
                           representation: stream clouds of 3 clusters through it --
                           benign jitter keeps SRTD~0 (never trips), a mode collapse
                           fuses them (Betti-0 3->1, SRTD spikes) and trips the
                           debounced FSM on the 2nd bad tick, then recovery closes it
                           on the 3rd clean tick -- a zero-param health signal, not a
                           representation edit
examples/demo_sensory_frontend.py  Close the perception loop: drive SensoryFrontend
                           with a raw 12-channel stream whose clock jitters and that
                           drops a burst of frames mid-way -- the stream lands on the
                           core's fixed-dt grid, the coverage mask flags exactly the
                           dropout steps (what a BlindRolloutGuard coasts on), and the
                           projected (B, M, d_model) tokens feed MTLNNModel via
                           inputs_embeds to finite logits -- zero model.py coupling
examples/demo_top_down_modulation.py  Top-down feedback, closed until opened: a
                           high-level goal vector biases every block. With the
                           gate closed (init) the goal is a STRICT no-op
                           (bit-identical to no goal -> a trained checkpoint is
                           untouched); open the gate and two distinct goals steer
                           the same input to different next-token distributions --
                           the pathway is live, default-off, zero regression
examples/demo_pipeline.py  Dual-speed sentry, all layers as one loop: a drone makes a
                           steady approach (wakes nobody), a sharp evasive turn (one
                           salient ignition wakes the slow layer, which forecasts the
                           threat: ENGAGE / breach imminent), a 2-tick sensor dropout
                           (the raw feed is first aligned to the fixed-dt grid by
                           ingest_ops.align_stream, so the dropout emerges as a
                           coverage gap and is coasted on the world model) and a zone
                           breach; ASCII top-down map + per-tick log + verdict, every
                           aim command bounded +/-90 deg and slew-limited <=20 deg/tick
examples/demo_cognitive_agent.py  The whole brain as one loop: an agent in a 2-D
                           obstacle scene PERCEIVES every object as a grid-cell place
                           code (L1), REMEMBERS what-is-where in a place-indexed
                           cognitive map (L2 SpatialMemory), is STEERED by a top-down
                           goal (closed gate = strict no-op, open = the goal biases the
                           backbone), IMAGINES a latent rollout (L4, 0 params, confidence
                           decays) while physics_ops validates candidate paths for
                           collisions + kinematics (the straight line hits the wall, a
                           detour is committed), SURVIVES a sensor blackout by coasting
                           on the world model then going DARK ("lost perception, stop"),
                           and CLAMPS its steering command to a safe envelope; ASCII map
                           + per-stage report + verdict
examples/demo_streaming_continual.py  Streaming continual learning on the real
                           MTLNNModel: the SAME task sequence trained twice -- naive
                           sequential training catastrophically forgets (final accuracy
                           collapses to ~1/n_tasks, forgetting measure ~1.0), while
                           interleaving a bounded reservoir-replay buffer (replay.py)
                           preserves every old task (forgetting ~0, accuracy ~1.0) and
                           adds ZERO model params; scored with the catastrophic-forgetting
                           metrics (continual_eval.py); ASCII accuracy matrices + verdict

bench_llama_mt_ablation.py        One-shot ablation table over checkpoints
bench_llama_mt_needle.py          Needle-in-a-haystack retrieval benchmark
benchmarks/compare_baselines.py   Selective Copy head-to-head (toy scale)
benchmarks/long_context.py        T=37/101/229 sweep
benchmarks/run_benchmark.py       Full benchmark suite

kaggle/                    Cloud-ready notebooks (Qwen-1.5B, Qwen-3B, ablations)
scripts/                   Real-trace v3 (KV-cache O(N)) + cloud-inject helpers
tests/                     Full test suite (967 tests, all pass)
assets/                    decks/ (investor + paper), figures/ (architecture diagrams)
```

---

## Status

Research-grade code. All 967 tests pass (model · rhythm · causality · world-model · observability · GWTB · coherence · AVP · operator layers + dual-speed sentry + autonomous cognitive agent + streaming continual learning + the three default-on Phase-1 brain mechanisms: predictive-coding loss, O(1) decay working memory, dynamic-kappa scale gating). Run the full suite with `python -m pytest tests/`, or the fast smoke path `python -m pytest tests/ -m "not slow"` (955 tests in ~99s, deselecting the 12 `slow` tests that train a model or download CLIP weights — the full run is ~8 min). Highlights:

```
[ok] test_kv_cache_parity                 cached vs full diff < 1e-4
[ok] test_state_only_memory_is_flat...    O(1) cache flat at T=20x window (vs KV O(T))
[ok] test_ingest_ops_properties           Hypothesis: linear resample exact on any affine signal
[ok] test_physics_ops_properties          Hypothesis: momentum/energy conservation laws hold
[ok] test_spatial_ops_properties          Hypothesis: metric/rigid-motion/reachability invariants hold
[ok] test_acoustic_ops_properties         Hypothesis: ITD/ILD/Doppler/superposition laws hold
[ok] test_salience_events_properties      Hypothesis: ignition hysteresis/refractory/DC-invariance hold
[ok] test_geometry_ops_properties         Hypothesis: Fisher-Rao metric/geodesic/exp-log/transport laws hold
[ok] test_topology_ops_properties         Hypothesis: MST/Betti-0/persistence/SRTD topology laws hold
[ok] test_stdp_ops / _properties          STDP window + trace==pairwise equivalence + causal-only-potentiates laws
[ok] test_attractor_ops / _properties     fixed point/spectral radius/rate/settling/Lyapunov + basin-probe stability laws
[ok] test_physics_ops (Verlet block)      velocity Verlet kick-drift-kick + 2nd-order energy/reversibility vs Euler
[ok] test_failsafe (TopologyBreaker)      SRTD/Betti tripwire ignores jitter, debounced trip on collapse + close on recovery
[ok] test_sensory_frontend               raw jittered stream -> fixed-dt grid + dropout coverage mask -> d_model tokens feed the backbone, grads only to projector
[ok] test_top_down_modulation            goal biases every block: closed gate == strict no-op (bit-exact), open gate steers logits, gate has live gradient, cache parity holds
[ok] test_lnn_recurrence_active           h_prev verifiably flows
[ok] test_gwtb_cache_parity               GWTB cached vs full diff < 1e-4
[ok] test_anesthesia_validation_protocol  Φ̂ collapses monotonically with κ
[ok] test_protofilament_scaling           P=64 only ~1.2× slower than P=13
[ok] test_lavi_persistent_higher_for_similar_input
[ok] test_global_rhythm_identity_at_init  scale=0 → output = input exactly
[ok] test_model_no_regression_rhythm_off  use_rhythm=False: zero output impact
```

What's validated:

- ✅ Architectural priors (13 protofilaments, GTP renewal, parallel-scan recurrence, RMC, GWTB) yield a modest but consistent edge on whole-sequence recall in long-range selective tasks at matched 200K params (×1.3 over a vanilla Transformer at $T{=}32$; ratio grows at longer $T$).
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
| [MT_LNN_ARCHITECTURE_VISUAL.md](docs/specs/MT_LNN_ARCHITECTURE_VISUAL.md) | Visual walkthrough |
| [BENCHMARKS.md](BENCHMARKS.md) | All benchmark numbers + AVP details |
| [docs/PRODUCT_LINES.md](docs/PRODUCT_LINES.md) | M-series (hybrid) vs O-series (attention-free ARR): honest capability cards |
| [RECIPES.md](docs/guides/RECIPES.md) | Recipe API reference |
| [ABLATIONS.md](ABLATIONS.md) | Ablation framework |
| [V2_REVIEW.md](docs/reviews/V2_REVIEW.md) | v2.0/v2.1 module review (incl. SimSiam collapse proof) |
| [TECH_BLOG.md](https://github.com/AwareLiquid/AwareLiquid-Web/blob/main/blog/TECH_BLOG.md) | Cross-architecture reproducibility story (moved to AwareLiquid-Web) |
| [PRD.md](PRD.md) | Product requirements & roadmap |
| [SPEC.md](SPEC.md) | Component spec |
| [mt_lnn_operator_algebra_whitepaper.tex](mt_lnn_operator_algebra_whitepaper.tex) | Operator-algebra math white paper: exact definitions + property-pinned invariants for all 11 zero-parameter operators wrapping the LTC core (honest "identity vs property-pinned vs statistical" grading) |
| [NEEDLE_FIX.md](docs/reviews/NEEDLE_FIX.md) | Fixed needle-in-a-haystack harness |
| [CLOUD_RUN.md](docs/guides/CLOUD_RUN.md) / [KAGGLE_RUN.md](docs/guides/KAGGLE_RUN.md) | Cloud / Kaggle reproduction |
| [llm-viz-QUICKSTART.md](llm-viz-integration/llm-viz-QUICKSTART.md) | 3D interactive architecture viewer |

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
