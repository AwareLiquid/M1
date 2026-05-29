# AwareLiquid (formerly MT-LNN) — Product Requirements Document

**Version:** 2.0
**Date:** 2026-05-29
**Status:** Active
**Repo:** https://github.com/everest-an/M1
**Supersedes:** v1.1 (2026-05-12), which anchored on the standalone 125M research artefact only. v1.1 goals are preserved under Track A (Research) below; the headline product is now Track B (AwareLiquid).

---

## 0. What changed since v1.1

v1.1 framed the project as a single deliverable: a 125M brain-inspired research model + the Anesthesia Validation Protocol + an arXiv paper.

Between 2026-05-24 and 2026-05-29 the project pivoted to compete in the Gemini-3.1-era reasoning-UX market:

| Date | Event |
|---|---|
| 2026-05-24 | `AWARENESS_NETWORK_PRD.md` introduces the **Cloud Oracle** strategy |
| 2026-05-24 | `ARCHITECTURE.md` rewrite — Predictive Coding, O(1) Memory, Compute Skipping |
| 2026-05-26 | Project renamed **M1 → AwareLiquid** |
| 2026-05-28 | Capsule v2 + ReasoningTrace shipped *"for Gemini-3.1-class reasoning UX"* |
| 2026-05-28 | Phase 5 — TinyLlama-1.1B + MT residual adapter → WikiText-2 PPL −28.5% |
| 2026-05-29 | Phase 5b — Qwen-2.5-1.5B + adapter → PPL −27.7% (cross-base replication) |
| 2026-05-29 | Real cloud-inject uplift on Qwen-1.5B: 83.3% → 96.7% (+13.3%) |

This v2.0 reflects the pivoted scope. **The research artefact is preserved**, not retired — it now lives as Track A. The new headline is Track B.

---

## 1. Product Overview / 产品概述

**What is AwareLiquid?**
AwareLiquid is an open-source reasoning system that combines:
1. A **local liquid-neural-network adapter** (MT residual adapter) bolted onto any open-weight 1B+ LM, which drops perplexity ~28% with ≤0.2% trainable parameters
2. A **cloud oracle inject pathway** that pulls verifiable facts from a frontier LM only when the local model's entropy / route gate decides it can't answer alone
3. A **fully auditable per-token reasoning trace** (JSONL + clickable HTML viewer) that records every routing decision, entropy spike, Φ̂ sample, and cloud-inject event

> *AwareLiquid 是一个开源推理系统：本地液态神经网络适配器（28% PPL 下降，0.2% 可训练参数）+ 云端按需注入（self-sufficiency 99%+）+ 完全可审计的逐 token 推理日志（替代 Gemini 黑盒 thinking summary）。*

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
- **B2** — Per-token reasoning trace: JSONL schema + clickable HTML viewer + quantitative audit (`bench_trace_audit.py`). **STATUS: ✅ scaffolding done, ⏳ real-inference wiring pending (Track 1B below)**
- **B3** — Cloud-inject pathway with measurable accuracy uplift on a real LM. **STATUS: ✅ +13.3% on 30-question harness (Qwen-1.5B)**
- **B4** — End-to-end demo: a user query that triggers local generation, an entropy-spike-triggered cloud inject, a fact absorption, and a full trace artifact for that session. **STATUS: ⏳ pending (Track 1B + 2 below)**
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
| Entropy-triggered routing | Local generation routes to cloud inject only when token entropy > threshold | 🔲 wiring pending (B4) |
| Per-session self-sufficiency metric | `1 - cloud_tokens/total_tokens`, reported per trace | ✅ on synthetic; ⏳ on real inference |

### F3 — Reasoning Trace (Track B, P0)

| Sub-feature | Requirement | Status |
|---|---|---|
| JSONL schema | One row per token: `(step, token_id, entropy, route, phi)` + separate `route` and `cloud_inject` events | ✅ |
| `trace_timeline.html` viewer | Single-file HTML; one colored bar per token; click for raw event | ✅ |
| `bench_trace_audit.py` | Reports route breakdown, self-sufficiency, entropy stats, Φ̂ stats, cost vs full-cloud | ✅ |
| Real inference emits trace | Hook `ReasoningTrace` into Qwen generate loop, not just synthetic demo | 🔲 pending (Track 1B below) |
| Demo session bundle | One canonical user-query trace shipped in repo; reproducible from scripts | 🔲 pending (B4) |

### F4 — Public Reproducibility (Track B, P0)

| Sub-feature | Requirement | Status |
|---|---|---|
| Kaggle notebooks | Each headline number reproducible from one `kaggle/*.ipynb` | ✅ Phase 5b, cloud-inject |
| Artefact JSONs | All headline tables back-by-JSON in `benchmarks/` | ✅ |
| Pinned torch | Kaggle notebooks pin `torch==2.4.1+cu121` to survive P100/T4 random assignment | ✅ |
| Pitch deck | `assets/decks/Pitch_Deck_MT_LNN.md` reflects current headline numbers | ✅ |
| Test suite green | All tests pass on CPU in < 2 min | 🔲 verify after Track 1B wiring |

### F5 — MT-LNN core architecture (Track A, P1, from v1.1)

Carried over from v1.1 §5 (F1-F8). No changes to specs. Track A items are no longer Track B acceptance gates but remain valid research deliverables.

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

### Track A (preserved from v1.1)

| Metric | Target | Status |
|---|---|---|
| Test suite | 17/17 in < 2 min CPU | ✅ |
| KV-cache parity | diff < 1e-4 | ✅ |
| WikiText-103 PPL (125M standalone) | < 22 | 🔲 deferred |
| AVP pass on trained MT-LNN | Φ̂(κ=10)/Φ̂(κ=1) ≤ 0.30 | 🔲 deferred |
| arXiv paper | published | 🔲 planned Track C |

---

## 7. Roadmap (post-v2.0)

Three tracks, sequenced:

### Track 1A — Scale validation
Re-run Phase 5b recipe on **Qwen-3B or Phi-3-mini-3.8B**. Goal: produce non-zero base needle scores so the MT adapter delta becomes measurable. Closes acceptance row "≥3B base needle non-zero." Wall: ~5h Kaggle.

### Track 1B — Wire ReasoningTrace into real Qwen inference
Currently traces are synthetic (`scripts/demo_trace_synth.py`). Hook `ReasoningTrace` into the Qwen + adapter generate loop, emit real per-token entropies, route decisions, optional cloud injects. Ship one canonical demo session. Closes B4 + F3 last row. Wall: ~1-2 days code.

### Track 2 — Brain-inspired Phase 1 (from `BRAIN_INSPIRED_ROADMAP.md`)
Currently deferred items that are "首选 / 高优" per roadmap:
- Dynamic channel gating (κ-based compute skipping)
- Working memory decay (GWTB upgrade)
- Predictive coding loss

These are independent of Track B headline metrics but feed the research narrative. Sequence after 1A/1B.

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
