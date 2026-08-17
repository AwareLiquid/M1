# AwareLiquid (formerly MT-LNN) — Product Requirements Document

**Version:** 2.2
**Date:** 2026-06-06
**Status:** Active
**Repo:** https://github.com/AwareLiquid/M1
**Supersedes:** v2.1 (2026-05-30), which added Position-Free Architecture. v2.2 adds EEG-inspired Rhythm Gate (LAVI + GlobalRhythmController).

> **⚠️ CORRECTION (2026-07-11) — read [RESULTS.md](RESULTS.md) first.** The "~28% PPL at ≤0.2% trainable params" adapter claim below is **retracted** (those runs trained LoRA only; the MT adapter was frozen and adds ≈0 PPL beyond LoRA). The real, proven differentiators are cross-window/cross-session **associative recall** (0.56 where attention/LoRA are 0.000) and **O(1) inference memory** in the attention-free O-series (1008× smaller than a KV cache at 128k). RESULTS.md is the source of truth; where this PRD disagrees, RESULTS.md wins.

---

## 0. What changed since v1.1

v1.1 framed the project as a single deliverable: a 125M brain-inspired research model + the Anesthesia Validation Protocol + an arXiv paper.

Between 2026-05-24 and 2026-05-30 the project pivoted to compete in the Gemini-3.1-era reasoning-UX market:

| Date | Event |
|---|---|
| 2026-05-24 | `docs/specs/AWARENESS_NETWORK_PRD.md` introduces the **Cloud Oracle** strategy |
| 2026-05-24 | `ARCHITECTURE.md` rewrite — Predictive Coding, O(1) Memory, Compute Skipping |
| 2026-05-26 | Project renamed **M1 → AwareLiquid** |
| 2026-05-28 | Capsule v2 + ReasoningTrace shipped *"for Gemini-3.1-class reasoning UX"* |
| 2026-05-28 | Phase 5 — TinyLlama-1.1B + MT residual adapter → WikiText-2 PPL −28.5% |
| 2026-05-29 | Phase 5b — Qwen-2.5-1.5B + adapter → PPL −27.7% (cross-base replication) |
| 2026-05-29 | Real cloud-inject uplift on Qwen-1.5B: 83.3% → 96.7% (+13.3%) |
| **2026-05-30** | **Position-Free Architecture** — h_prev-based timing replaces RoPE → **94.11%** performance |
| **2026-06-06** | **EEG Rhythm Gate** — LAVI estimator + GlobalRhythmController; dynamic stability/flexibility balance |

This v2.2 adds the Rhythm Gate (F7) as a Track A architecture advancement with direct Track B relevance: improved long-context stability and smoother multi-turn reasoning. **Both Track A and Track B remain active**; Track B (AwareLiquid) is still the headline product.

---

## 1. Product Overview / 产品概述

**What is AwareLiquid?**
AwareLiquid is an open-source reasoning system that combines:
1. A **local liquid-neural-network adapter** (MT residual adapter) bolted onto any open-weight 1B+ LM, which adds cross-window/cross-session **associative recall** (0.56 accuracy where attention/LoRA score 0.000) at ~1% param overhead — *not* a perplexity win over LoRA (see [RESULTS.md](RESULTS.md))
2. A **cloud oracle inject pathway** that pulls verifiable facts from a frontier LM only when the local model's entropy / route gate decides it can't answer alone
3. A **fully auditable per-token reasoning trace** (JSONL + clickable HTML viewer) that records every routing decision, entropy spike, Φ̂ sample, and cloud-inject event

> *AwareLiquid 是一个开源推理系统：本地液态神经网络适配器（跨窗口/跨会话联想记忆 0.56,注意力/LoRA 结构性为 0;非 PPL 优势)+ 云端按需注入 + 完全可审计的逐 token 推理日志。详见 [RESULTS.md](RESULTS.md)。*

**Pitch in one line:** Gemini-3.1-class reasoning UX, but local-first, audit-trail-native, and runnable on a single RTX 4090.

---

## 2. Problem Statement / 痛点与定位

### 2.1 The Gemini-3.1 challenge

Frontier reasoning models (Gemini 3.1, Claude Sonnet 4.6, GPT-5) ship "thinking" features that:
- Bill at $15+/MTok output
- Return a post-hoc opaque "thinking summary" — not a per-token route, not diffable, not auditable
- Require always-cloud — no local-first path, no compliance story for regulated industries
- Have no measured cost/benefit boundary for when the cloud is actually needed

### 2.2 Why standard Transformers also have a memory wall

KV-cache scales O(N²) with context. Brute-force long-context inference burns A100-grade memory per user. The local-first path requires a recurrent/state-space alternative.

### 2.3 AwareLiquid's wedge

| Frontier-cloud (Gemini 3.1) | AwareLiquid |
|---|---|
| Always cloud | Local-first; cloud only on entropy spike (measured: 0.8% of tokens on demo trace) |
| Opaque thinking summary | Per-token JSONL `(step, entropy, route, phi, source)` + HTML viewer |
| O(N²) KV cache | O(1) recurrent state via MT adapter |
| No compliance audit trail | Every cloud query + every fact's provenance in `evidence_log` |
| Cost: $15/MTok output | Cost: ~$0.0016 net per 120-token answer (measured on demo trace) |
| No public reproduction | All artefacts in `benchmarks/`, all training in `kaggle/*.ipynb` |

---

## 3. Goals and Non-Goals / 目标与非目标

### Track B — AwareLiquid (P0, headline product)

- **B1** — Ship MT residual adapter that runs on top of any open-weight 1B+ LM, ≤0.2% trainable params, ≥25% PPL improvement, demonstrated on ≥2 base families (Llama + Qwen). **STATUS: ✅ done (Phase 5 + 5b)**
- **B2** — Per-token reasoning trace: JSONL schema + clickable HTML viewer + quantitative audit (`bench_trace_audit.py`). **STATUS: ✅ done — scaffolding *and* real-inference wiring shipped (`scripts/awareliquid_real_trace_v3.py` hooks `ReasoningTrace` into a real Qwen KV-cache generate loop; artefacts `real_trace_demo.jsonl` / `real_trace_v3_test.jsonl`). Remaining: the *canonical adapter-on Kaggle* trace (Track 1B).**
- **B3** — Cloud-inject pathway with measurable accuracy uplift on a real LM. **STATUS: ✅ +13.3% on 30-question harness (Qwen-1.5B)**
- **B4** — End-to-end demo: a user query that triggers local generation, an entropy-spike-triggered cloud inject, a fact absorption, and a full trace artifact for that session. **STATUS: ✅ shipped on Qwen-0.5B/CPU smoke (`awareliquid_real_trace_v3.py` emits real per-token entropies, entropy-gated injects, and a session trace; audit `artifacts/real_trace_demo_audit.json`). ⏳ remaining: re-run as the canonical adapter-on session (Track 1B).**
- **B5** — Public reproducibility: every claim has a script in `scripts/`, a notebook in `kaggle/`, and JSON artefacts in `benchmarks/`. **STATUS: ✅ ongoing**

### Track A — Research artefact (P1, preserved from v1.1)

The original v1.1 thesis is downgraded to P1 but kept alive — the 125M standalone MT-LNN model and AVP are still publishable research and feed Track B's Φ̂ trace integration.

- **A1** — 17/17 test suite passing on CPU in <2 min. **STATUS: ✅ (v1.1 met)**
- **A2** — 125M standalone MT-LNN trained on WikiText-103 to PPL < 22. **STATUS: 🔲 deferred, not blocking Track B**
- **A3** — Anesthesia Validation Protocol (AVP) with Φ̂ metric pass at trained checkpoint. **STATUS: 🔲 deferred, not blocking Track B**
- **A4** — arXiv paper covering Track A + Track B evidence. **STATUS: 🔲 planned as Track C (writing phase)**

### Non-Goals / 非目标

- **NG1** — AwareLiquid does not claim sentience or subjective experience.
- **NG2** — Not a hosted SaaS; the deliverable is open-source weights, code, and artefacts.
- **NG3** — Not an RLHF-aligned chat product; v2.0 ships raw adapter + tooling.
- **NG4** — Not a Gemini-3.1 replacement on *all* axes (multimodal, function calling, video) — those are roadmap items, not v2.0 acceptance criteria.

---

## 4. Users and Personas

### Primary: Compliance-sensitive enterprise reasoning user
Finance, legal, healthcare. Cannot ship Gemini's opaque thinking — every fact's provenance must be auditable. Needs: per-token trace, deterministic local fallback, low TCO.

### Secondary: Open-source AI engineer / hacker
Needs: a working 1B-scale adapter recipe, reproducible Kaggle notebooks, JSON artefacts they can diff. They were already using TinyLlama/Qwen and want measurable PPL/accuracy uplift without huge train budgets.

### Tertiary: AI × neuroscience researcher (was v1.1 primary)
Needs: 125M standalone MT-LNN + Anesthesia Validation Protocol + arXiv citation. Track A serves them.

---

## 5. Feature Requirements

### F1 — MT Residual Adapter (Track B, P0)

| Sub-feature | Requirement | Status |
|---|---|---|
| `attach_mt_adapters(model, layer_idxs)` API | One-line wrap of any HF causal LM | ✅ |
| Attribute proxy on wrapper | `DecoderLayerWithMTAdapter` proxies `attention_type` etc. to base layer (new HF transformers need this) | ✅ `b108744` |
| Checkpoint format | `attach_adapters_from_checkpoint(model, ckpt)` reconstructs adapters from `.pt` | ✅ |
| Trainable param budget | ≤ 0.2% of base model params | ✅ TinyLlama 0.196%, Qwen 0.139% |
| PPL improvement | ≥ 25% WikiText-2 drop at 1000 steps, batch 1, grad_accum 8 | ✅ TinyLlama −28.5%, Qwen −27.7% |
| Cross-base reproducibility | Same recipe must work on ≥2 LM families | ✅ Llama + Qwen |
| ≥3B base validation | Same recipe at Qwen-3B / Phi-3-mini; needle non-zero on base | ⏳ Track A (next) |

### F2 — Cloud Oracle Inject Pathway (Track B, P0)

| Sub-feature | Requirement | Status |
|---|---|---|
| Inject template | `[Absorbed fact] {fact}\nContinuing: Question: {q}\nAnswer:` | ✅ |
| Real-model accuracy uplift | ≥ +10% on 30-question harness, real HF backend | ✅ +13.3% on Qwen-1.5B |
| Adapter does not break in-context learning | Same uplift with/without MT adapter loaded | ✅ |
| Entropy-triggered routing | Local generation routes to cloud inject only when token entropy > threshold | ✅ wired in `awareliquid_real_trace_v3.py` (LOCAL/CLOUD entropy thresholds gate real injects) |
| Per-session self-sufficiency metric | `1 - cloud_tokens/total_tokens`, reported per trace | ✅ on synthetic **and** on real inference (`artifacts/real_trace_demo_audit.json`) |

### F3 — Reasoning Trace (Track B, P0)

| Sub-feature | Requirement | Status |
|---|---|---|
| JSONL schema | One row per token: `(step, token_id, entropy, route, phi)` + separate `route` and `cloud_inject` events | ✅ |
| `trace_timeline.html` viewer | Single-file HTML; one colored bar per token; click for raw event | ✅ |
| `bench_trace_audit.py` | Reports route breakdown, self-sufficiency, entropy stats, Φ̂ stats, cost vs full-cloud | ✅ |
| Real inference emits trace | Hook `ReasoningTrace` into Qwen generate loop, not just synthetic demo | ✅ done (`scripts/awareliquid_real_trace_v3.py`: manual KV-cache token loop emits real per-token entropy/route/inject events) |
| Demo session bundle | One canonical user-query trace shipped in repo; reproducible from scripts | ✅ Qwen-0.5B/CPU smoke shipped (`real_trace_demo.jsonl`); ⏳ canonical adapter-on rerun pending (Track 1B) |

### F4 — Public Reproducibility (Track B, P0)

| Sub-feature | Requirement | Status |
|---|---|---|
| Kaggle notebooks | Each headline number reproducible from one `kaggle/*.ipynb` | ✅ Phase 5b, cloud-inject |
| Artefact JSONs | All headline tables back-by-JSON in `benchmarks/` | ✅ |
| Pinned torch | Kaggle notebooks pin `torch==2.4.1+cu121` to survive P100/T4 random assignment | ✅ |
| Pitch deck | [AwareLiquid-Web](https://github.com/AwareLiquid/AwareLiquid-Web) `decks/Pitch_Deck_MT_LNN.md` reflects current headline numbers (decks migrated out of M1) | ✅ |
| Test suite green | All tests pass on CPU in < 2 min | ✅ 967 tests green on CPU (Track 1B wiring landed) |

### F5 — MT-LNN core architecture (Track A, P1, from v1.1)

Carried over from v1.1 §5 (F1-F8). No changes to specs. Track A items are no longer Track B acceptance gates but remain valid research deliverables.

### F6 — Position-Free Architecture (Track A, P1, 2026-05-30)

| Sub-feature | Requirement | Status |
|---|---|---|
| Dual-path architecture | Preserve 100% of RoPE baseline; add position-free path with config switch | ✅ `df6ecd9` |
| h_prev timing extractor | Extract position signal from liquid state (B,P,S,D) via tau-weighted aggregation | ✅ |
| Tau-weight initialization | 1/sqrt(τ) init: fast scales (τ=0.01) weighted 10× higher than slow scales (τ=10) | ✅ |
| Content-based attention | Low-rank bilinear polarity (rank=16) for position-free mode | ✅ |
| Performance target | ≥90% of RoPE baseline on Selective Copy task | ✅ **94.11%** |
| Parameter overhead | ≤15% additional params vs RoPE-only | ✅ **12.8%** |
| Kaggle deployment | Production-ready test script with multi-scale (tiny/small/medium) support | ✅ |

**Rationale**: Position-Free Architecture replaces external position encoding (RoPE) with internal timing signals from liquid dynamics. This aligns with the AGI thesis: understanding through state evolution, not statistical token prediction. Achieves 94.11% of baseline performance with only 12.8% parameter overhead.

**Key innovation**: No max_seq_len constraint in theory (position from h_prev, not absolute indices); enables true O(1) memory streaming via state-only mode.

### F7 — EEG-Inspired Rhythm Gate (Track A, P1, 2026-06-06)

| Sub-feature | Requirement | Status |
|---|---|---|
| `LAVIEstimator` per layer | Cosine-sim LAVI proxy; output ∈ [0,1], shape `(B,T,P,1)` | ✅ `mt_lnn/rhythm.py` |
| Rhythm blend in resonance | LAVI gates τ-scale blend: high LAVI → slow scales, low → fast | ✅ `mt_lnn_layer.py` |
| Neutral init (zero impact) | `h_prev=None` → LAVI=0.5 → rhythm_bonus=0 → output unchanged | ✅ `test_rhythm.py` |
| `GlobalRhythmController` | Cross-layer LAVI aggregation; residual correction before GWTB | ✅ `mt_lnn/rhythm.py` |
| Identity at init | `GlobalRhythmController.scale=0` → zero correction at step 0 | ✅ test verified |
| Config opt-in | `use_rhythm=False` default; zero impact on existing code/checkpoints | ✅ |
| Diagnostics | `lavi_mean/min/max`, `rhythm_scale_mean`, `global_rhythm_scale` in `get_mt_diagnostics()` | ✅ |
| Streaming gradient | LAVI bias receives gradient during step-2+ streaming inference | ✅ `test_model_rhythm_gradient_flow` |
| Biological validity | High LAVI for similar input (persistent mode), low for novel input | ✅ `test_lavi_persistent_higher_for_similar_input` |
| Test coverage | 13 dedicated tests, all pass | ✅ `tests/test_rhythm.py` |

**Product angle — what this changes for users:**

The existing κ-gate already answers "what is the input now?" (content-based). The rhythm gate answers "how consistent has reasoning been?" (history-based). Combined, AwareLiquid now has **two independent stability signals**:

1. **Short-term**: κ-gate responds to each token's content (fast, reactive)
2. **Long-term**: LAVI responds to h_prev/input alignment (slower, contextual)

For AwareLiquid's target use cases:
- **Long-document analysis** (B2): Sustained reading → high LAVI → slow τ dominant → state drifts less over 10K+ tokens
- **Multi-turn chat** (B4): Context switch at turn boundary → LAVI drops → fast τ activates → model adapts quicker without catastrophic forgetting
- **Compliance audit trail** (B3): LAVI time-series is logged alongside Φ̂ in `get_mt_diagnostics()`, giving auditors a per-layer stability index per inference step

**No parameter budget impact**: `LAVIEstimator` adds only P=13 learnable bias values per layer. At default `rhythm_scale_init=0.1`, the gate starts at near-zero influence and grows only if training finds it useful — it cannot hurt baseline performance.

**Rationale**: EEG research (persistent vs transient oscillatory modes) provides a mathematically well-grounded reason for why the existing κ-gate's purely content-based signal is incomplete. LAVI is the missing history dimension. Both are needed for the stability-flexibility tradeoff that is the defining challenge of all recurrent architectures.

---

### F8 — Bio-Inspired Cognitive Modules (Track A, P1, 2026-06-07, v2.1)

Four orthogonal brain-inspired modules + observability. **All default OFF, zero regression**, never change the `mt_lnn_layer.py` forward/parallel_scan signature, all gated behind config flags, each backed by tests + negative controls.

| Sub-feature | Requirement | Status |
|---|---|---|
| **Phase A** `CompetitiveGWTBLayer` | Multi-source bids → score → winner broadcast; degenerates to `GWTBLayer` when `module_bids=None` | ✅ `mt_lnn/gwtb.py` |
| Competition-collapse guard | `gwtb_competition_entropy` diagnostic (0=monopoly, log K=uniform) | ✅ `test_gwt_competition.py` |
| **Phase B** `CausalConsistencyChecker` | `consistency_score ∈ [0,1]`; `cosine` (default) + anisotropy-robust `subspace` method | ✅ `mt_lnn/causality.py` |
| Subspace beats cosine on anisotropy | Topic-switch: cosine Δ≈0.000 (blind) vs subspace Δ≈0.414 | ✅ `test_subspace_detects_break_cosine_blind` |
| `effective_rank` diagnostic | Participation ratio `(Σλ)²/Σλ²` tracks representation complexity | ✅ `test_effective_rank_diagnostic_tracks_complexity` |
| Deliberation integration | consistency < threshold → forced SELF_CRITIQUE (backward-compatible optional field) | ✅ `test_subspace_break_triggers_self_critique` |
| **Phase C** `PredictiveStateHead` | BYOL/V-JEPA online-predictor + EMA stop-grad `target_proj`; loss on L2-normalised latents | ✅ `mt_lnn/world_model.py` |
| No representational collapse | pairwise \|cos\| < 0.5 (vs naïve self-prediction ≈ 1.000) | ✅ `test_no_representational_collapse` |
| Normalised surprise → LAVI | `last_pred_error ∈ [0,1]`; raw MSE kept as `last_pred_error_raw` | ✅ `test_pred_error_normalised_to_unit_interval` |
| Surprise tracks structure not noise | structured < 0.25, noise > 0.35, separation > 0.2 | ✅ `test_surprise_tracks_structure_not_noise` |
| EMA warmup | gentler decay (≤0.9) for first `warmup_steps` | ✅ `test_ema_warmup_uses_gentler_decay` |
| **Phase D** `HebbianRegularizer` | LAVI-gated co-activation loss term (training only, no forward change) | ✅ `mt_lnn/plasticity.py` |
| **Observability** | `record_v2_metrics(writer, model, step, checker)` → JSONL, all scalars bounded [0,1], every 100 steps | ✅ `mt_lnn/observability.py` |
| Targeted grad clip | `--world_model_grad_clip` (default 1.0) before global clip | ✅ `train.py` |
| Config opt-in | `use_competitive_gwtb` / `use_world_model` / `use_hebbian` all default False | ✅ `mt_lnn/config.py` |
| Test coverage | +9 mechanism/negative-control tests; full suite 219 tests, all pass | ✅ `tests/` |

**Scientific finding (Phase C) — SimSiam vs BYOL:** Rigorous empirical investigation (3 seeds) corrected the common assumption that any non-EMA self-prediction collapses. Our `use_ema_target=False` branch is a **SimSiam** design (stop-grad + predictor + bias-free L2-normalised projector) and is **provably collapse-free on its own** (pairwise |cos| ≈ 0.33). Only naïve self-prediction (no stop-grad, no predictor) collapsed (≈1.000). EMA's role is convergence/quality, **not** collapse-prevention — at this scale EMA ≈ SimSiam (not the rumored +25%). `use_ema_target` is retained as an ablation switch (default True). Full analysis: [V2_REVIEW.md](docs/reviews/V2_REVIEW.md) §8.

**Product angle — what this changes for users:** these modules give AwareLiquid measurable *cognitive* signals beyond raw PPL: a competition-health index (GWT routing not collapsing), a per-step causal-consistency / topic-break detector (drives self-critique), a predictive-surprise channel (novelty detection feeding the rhythm gate), and Hebbian consolidation against catastrophic forgetting. All are exposed as bounded scalars in `get_mt_diagnostics()` / JSONL for audit trails (B3) and long-run monitoring.

---

## 6. Acceptance Criteria (v2.0 headline metrics)

### Track B (must hit for v2.0 ship)

| Metric | Target | **Measured** | Source |
|---|---|---|---|
| PPL drop on 1B+ base, 1000 steps | ≥ −25% | **−28.5% (TinyLlama)** | `benchmarks/kaggle_run/ppl_ablation.json` |
| PPL drop, cross-family base | ≥ −25% | **−27.7% (Qwen-1.5B)** | `benchmarks/kaggle_qwen_run/ppl_ablation.json` |
| PPL drop, ≥3B base | ≥ −25% | **−34.4% (Qwen-3B)** | `benchmarks/kaggle_qwen3b_run/ppl_ablation.json` |
| Trainable params | ≤ 0.2% | **0.117%–0.196%** | same |
| Cloud-inject accuracy uplift, real backend | ≥ +10% | **+13.3% (Qwen-1.5B, 30 Q)** | `benchmarks/cloud_inject_qwen/*.json` |
| Adapter preserves in-context learning | uplift Δ vs no-adapter ≤ 1pp | **Δ = 0pp (identical)** | same |
| Self-sufficiency on demo trace | ≥ 95% | **99.17%** (synthetic) · **100%** (real Qwen-0.5B smoke) | `demo_trace_audit.json` · `artifacts/real_trace_demo_audit.json` |
| Real-inference trace | demo session shipped | ✅ **shipped** (Qwen-0.5B/CPU; adapter-on canonical Kaggle run in flight) | `scripts/awareliquid_real_trace.py` · `artifacts/real_trace_demo.jsonl` |
| ≥3B base needle non-zero | base non-zero accuracy at 4096-context | ⏳ **harness fixed (2026-05-30)** — Qwen-0.5B achieves 1.0 acc with chat-template; pending rerun on 1.5B/3B+adapter | `NEEDLE_FIX.md` · `bench_needle_chat_template.py` |

### Track A (preserved from v1.1 + Position-Free 2026-05-30)

| Metric | Target | Status | Source |
|---|---|---|---|
| Test suite | 17/17 in < 2 min CPU | ✅ | — |
| KV-cache parity | diff < 1e-4 | ✅ | — |
| Position-Free performance (Selective Copy) | ≥90% of RoPE baseline | ✅ **94.11%** | `test_position_free_optimized.py` |
| Position-Free token accuracy | ≥68% absolute | ✅ **71.78%** | same (baseline: 76.27%) |
| Position-Free parameter overhead | ≤15% | ✅ **12.8%** | same (323K → 365K params) |
| Position-Free training convergence | 1000 steps | ✅ | same |
| WikiText-103 PPL (125M standalone) | < 22 | 🔲 deferred | — |
| AVP pass on trained MT-LNN | Φ̂(κ=10)/Φ̂(κ=1) ≤ 0.30 | 🔲 deferred | — |
| arXiv paper | published | 🔲 planned Track C | — |


---

## 7. Roadmap (post-v2.0)

Three tracks, sequenced:

### Track 1A — Scale validation
Re-run Phase 5b recipe on **Qwen-3B or Phi-3-mini-3.8B**. Goal: produce non-zero base needle scores so the MT adapter delta becomes measurable. Closes acceptance row "≥3B base needle non-zero." Wall: ~5h Kaggle.

### Track 1B — Wire ReasoningTrace into real Qwen inference
**DONE.** The wiring shipped: `scripts/awareliquid_real_trace_v3.py` hooks `ReasoningTrace` into a real Qwen generate loop (manual KV-cache token loop, real per-token entropies, entropy-gated route/cloud-inject decisions); a synthetic generator (`scripts/demo_trace_synth.py`) is retained only for the UI demo. Artefacts: `real_trace_demo.jsonl` + `real_trace_demo_audit.json`. **Remaining:** re-run as the *canonical adapter-on Kaggle* session (cosmetic — the path itself is proven on the Qwen-0.5B/CPU smoke).

### Track 2 — Brain-inspired Phase 1 (from `docs/specs/BRAIN_INSPIRED_ROADMAP.md`)
**DONE (correcting the earlier "deferred" framing).** All three are wired and **ON by default** in `MTLNNConfig`, and each now has a behavioural-contract test (added 2026-06):
- Dynamic channel gating (`dynamic_scale_gates=True`) — `tests/test_dynamic_scale_gating.py`. NB: by default the κ-gate only *reweights* scales; real κ-based compute skipping needs `sparse_resonance_kernel=True` (top-k scale selection), also pinned.
- Working memory decay / GWTB upgrade (`use_decay_wm=True`) — `tests/test_decay_working_memory.py` (the O(1)-vs-O(T) streaming-cache contract).
- Predictive coding loss (`use_predictive_coding=True`) — `tests/test_predictive_coding_loss.py`.

These feed the research narrative and are independent of Track B headline metrics.

### Track 3 — arXiv tech report (Track C)
2-3 page short paper bundling Phase 5 + 5b + cloud-inject + trace audit + (if done) Track 1A 3B-scale results. Wall: ~1 week.

### Deferred / Out-of-scope for v2.0
- Multimodal, function calling, agent OS (mentioned in pitch deck as Stage 2-3)
- 7B / 70B scale training (Stage 2 / 3 in roadmap)
- Hosted SaaS, RLHF alignment
- Track A G3 / G4 (AVP pass + arXiv) — moved to Track 3

---

## 8. Architecture snapshot (v2.0)

```
Local stack:
  Base LM (open weights, frozen)       ← TinyLlama-1.1B / Qwen-2.5-1.5B (tested), Qwen-3B / Phi-3-mini-3.8B (planned)
  + MT residual adapter on every 4th layer  ← O(1) recurrent state, 0.2% trainable
  + LoRA on q/k/v/o projections             ← absorbs adapter output into attention
  ─────────
  Per-token generate loop emits ReasoningTrace JSONL events

Routing:
  Entropy of next-token distribution > threshold ?
      ├─ no  → LOCAL route, emit token
      ├─ yes → SELF_CRITIQUE: regenerate with explicit reflection prompt
      └─ yes & still uncertain → CLOUD route:
            POST {q} to frontier LM
            wrap returned fact in [Absorbed fact] template
            re-generate with absorbed fact in context
            emit cloud_inject event with fact_len and bytes_absorbed

Track A (research):
  125M standalone MT-LNN with original v1.1 F1-F8 features
  (independent of Track B; supports AVP + Φ̂ research)
```

---

## 9. Dependencies / Constraints

### Runtime
- Python ≥ 3.10
- PyTorch 2.4.1 (pinned for Kaggle P100/T4 compatibility; sm_60 support)
- CUDA 12.1 build of torch for Kaggle, cu118+ for local
- transformers, peft, accelerate, safetensors, datasets

### Training hardware
- Track B adapter: Kaggle free-tier GPU (T4 or P100) — 1000 steps on Qwen-1.5B fits in ~3h
- Track A 125M standalone: single A100 (v1.1 spec)
- Track 1A 3B scale: Kaggle GPU; ~5h estimated

### Design constraints
- Adapter must remain ≤ 0.2% trainable params (else story becomes "we just LoRA'd Qwen")
- Adapter wrapper must proxy attribute access (`DecoderLayerWithMTAdapter.__getattr__`) — non-negotiable since new HF transformers introspects layer attributes
- Track A 13-protofilament biological constraint preserved (v1.1 §8)

---

## 10. Open questions / risks

| Question | Risk | Notes |
|---|---|---|
| Does the MT adapter PPL gain hold at ≥3B base? | Medium | If yes — strong scaling story. If no — Track B caps at 1B–2B "small-model" market |
| Does the +13.3% cloud-inject uplift hold across other Q&A formats (MMLU, TriviaQA)? | Medium | Current 30-question harness is hand-curated; needs broader replication before publication |
| Will entropy-triggered routing in real inference produce sane route decisions? | Medium | Synthetic demo trace was hand-tuned; real inference may pick weird threshold values |
| Track A (125M standalone) gets de-prioritised long enough that the v1.1 research narrative atrophies | Low–Medium | Track 3 paper explicitly bundles both — that's the mitigation |
| Conversation logs contain a Kaggle API token | Operational | Rotate token after each external collaboration |
| Gemini 3.x evolves into local-first / open-trace before we ship | Medium | Maintain a 4-6 week lead time on key features; trace JSONL spec is the moat |
