# MT-LNN Results — Canonical Evidence Base

**This file is the single source of truth for what MT-LNN has and has not demonstrated.** Every row maps to a reproducible table in [BENCHMARKS.md](BENCHMARKS.md). If a marketing doc, slide, or README states a number that is not in the "Proven" section below (or is contradicted by the "Retracted / Null" section), that doc is wrong and this file wins.

Last reconciled: 2026-07-16.

---

## The one-paragraph honest summary

MT-LNN is a streaming-state recurrent architecture whose **one independently proven, hard-to-replicate result** is that its fast-weight state supports **cross-window / cross-session associative recall** (0.56 mean accuracy) that attention and LoRA score **exactly 0.000** on by construction. ⚠️ **The former headline "−31% validation PPL vs a matched Transformer" is RETRACTED as of 2026-07-19**: it was measured at 2,000 steps (undertrained) against a simple-reference baseline. At convergence (20,000 steps, 3 seeds) the gap shrinks to 5.5% (88.93 ± 0.33 vs 94.14 ± 0.78), and a **modern Transformer baseline (RoPE/RMSNorm/SwiGLU) beats MT-LNN by 11.3%** (78.86 ± 0.25). Language-modeling perplexity is **not** currently an MT-LNN advantage — see below. A separate attention-free variant (the O-series / ARR) has genuinely **O(1) inference memory** (flat 0.381 MB vs an O(T) KV-cache, up to 1008x smaller at 128k context). The earlier headline "−28–34% PPL adapter" wins were **retracted** (they were plain LoRA — the MT adapter was frozen and adds ≈0 PPL), out-of-window language modeling is a **null result**, and the Orch-OR / Φ̂ / consciousness modules are **inert in the trained path** (with the AVP sign inverted vs theory). We do not sell those.

---

## PROVEN — reproducible, and the only claims allowed in public docs

| Claim | Number | Status | BENCHMARKS.md section |
|---|---|---|---|
| ~~From-scratch native 125M MT-LNN beats a **matched** Transformer on val PPL~~ | ~~fp16 AMP: 299.5 vs 435.6 (−31%); fp32: 257.5 vs 370.8 (−30.6%)~~ | ❌ **RETRACTED 2026-07-19** — both runs 2K-step (undertrained) vs a weak simple-ref baseline. Superseded by the convergence row below. | "20K convergence + modern baseline (2026-07-19)" |
| **LM quality at convergence** (20K steps, n=3, fp32, WikiText-103) | modern_transformer **78.86 ± 0.25** (144.1M) **<** mt_lnn **88.93 ± 0.33** (126.0M) **<** transformer **94.14 ± 0.78** (142.1M). All 9 runs stable. | ⚠️ **honest negative**: mt_lnn beats the simple baseline by 5.5% but **loses to a modern Transformer by 11.3%**. PPL is not an MT-LNN advantage. | "20K convergence + modern baseline (2026-07-19)" |
| 125M recurrent/liquid model trains **stably at scale** (the review's central "does it converge at 100x?" fear) | No NaN / no divergence for transformer / lnn / mt_lnn (fp16, 2026-07-05) and transformer / mt_lnn / mamba (fp32, 2026-07-16) at 125M | ✅ proven | "Scaling to ~125M → --mode train" + "fp32 reconfirmation + Mamba baseline (2026-07-16)" |
| First Mamba baseline at matched 125M-class param count | fp32, **2000 steps only**, seed 0: MT-LNN 257.5 vs Mamba 414.0 PPL | ⚠️ **not re-run at convergence** — given that the Transformer comparison reversed at 20K steps, this 2K-step Mamba result cannot be assumed to hold either. Treat as provisional; width/depth-mismatched external reference. | "fp32 reconfirmation + Mamba baseline (2026-07-16)" |
| Cross-window associative recall through fast-weight state | **0.56 mean** (3-seed 0.621 / 0.434 / 0.621, ±0.09; in-window 0.99–1.00 every seed) | ✅ proven; attention/LoRA are **0.000 by construction** (structural zero-channel) | "Cross-window associative recall" |
| The fast-weight matrix **is** the memory (not incidental) | Remove fast-weight → cross-window collapses **0.553 → 0.008**; v1's 62.8M EMA state manages only 0.002 | ✅ proven | "Cross-window associative recall" |
| Cross-session snapshot → disk → fresh process → restore is **lossless** | Round-trip Δ **+0.008 / +0.000**; unit test **bit-exact (max\|diff\| 0.0)**; wrong-session restore = chance; no-restore = chance | ✅ proven | "Cross-session persistence" |
| O(1) inference memory — **O-series (ARR) only** | ARR state **flat 0.381 MB** vs KV-cache O(T): 3.9x @512 → 252x @32k → 1008x @128k → **8063x @1M** (KV would be 3072 MB). Verified across a **2048x context increase** with the state unchanged to the decimal. | ✅ proven (attention-free O-series only; conservative — GQA=1 already shrinks KV 13x). ARR figure is a measured snapshot-byte sum; KV figure is exact analytic. | "Scaling to ~125M → --mode decode" |
| Adapter (M-series SFT) is **capability-neutral** — recall machinery costs no core ability | LAMBADA +0.7pt, ARC-easy +0.7/+1.2, HellaSwag −0.7/−0.9, PIQA 0.0/+0.4 (all ±1pt, noise) | ✅ proven (deployment-safety) | "Capability evals — v2s SFT is ability-neutral" |
| Bio prior is a good **initialization**, not the endpoint | Frozen-τ cross-window **0.285** vs trained **0.621** (bio init ≈46% of the effect, 285x chance; training doubles it) | ✅ proven | "Cross-window recall → Bio-prior (frozen-τ) ablation" |
| Cloud-inject prompt template lifts factual accuracy on a real model | **+13.3%** (25/30 → 29/30 on Qwen-1.5B) — **and is identical with vs without the MT adapter** | ✅ proven for the **template**; the adapter contributes 0 to this (see "what we do NOT claim") | "Real cloud-inject numbers on Qwen-2.5-1.5B" |
| Selective Copy (toy, ~200K params, matched budget, fair decode) | MT-LNN seq-exact **0.895** vs Transformer 0.676 (×1.32) / LNN 0.727; ratio widens to ×2.0 at T=229 | ✅ proven **at toy scale only**; LNN baseline close behind — most gain is the liquid component | "Selective Copy" / "Long-context sweep" |
| **Irregular-sampling robustness on a real BMS task** (NASA battery, held-out cell, 10 seeds) | At 80% of timesteps dropped: mt_lnn degrades **+7.7%** vs lstm **+31.1%** / gru **+32.8%**. Welch **t=+2.40 vs lstm, +2.63 vs gru** — significant. Δt supplied to every arch, so the RNNs are not handicapped. | ✅ proven — this is the liquid core's one demonstrated architectural edge over discrete RNNs | "A2 edge pilot: battery SoH on real NASA cells" |
| **Constant streaming state on the same task** | mt_lnn flat at **2.6 KB** across a 256× stream-length increase; transformer KV reaches **34 MB** at 32K = **13,107× larger** | ✅ proven — but note O(1) state is a property of *any* RNN: lstm (1,040 B) and gru (520 B) are also flat and smaller | "A2 edge pilot → streaming memory" |
| Battery SoH accuracy, regular sampling | transformer 0.0972 / **mt_lnn 0.0986** / lstm 0.1034 / gru 0.1048 RMSE (Ah), 10 seeds; baseline 0.1572 | ⚠️ **all four statistically tied** (every pairwise \|t\| < 1). MT-LNN reaches parity with 2.6× fewer params than the transformer — that, not accuracy, is the claim | "A2 edge pilot → accuracy" |
| Real recurrence (pscan) does real work | pscan vs legacy broadcast: seq-exact **0.965 vs 0.883** (+8.2pp), tok-acc 0.983 vs 0.942 | ✅ proven | "Parallel scan ablation" |

---

## RETRACTED / NULL / INERT — must not appear as selling points anywhere

| Old claim | The real number | Status | BENCHMARKS.md section |
|---|---|---|---|
| "MT adapter drops PPL −28.5% / −27.7% / −34.4% at 0.117–0.196% trainable params" (TinyLlama / Qwen-1.5B / Qwen-3B) | The MT adapter was **frozen by PEFT** — only LoRA trained. Controlled ablation: **lora_only 7.984 vs mt_lora 7.920** (v1) / **mt_v2_lora 7.918** — MT adds **≈0 PPL** (−0.064 for +62.8M params, within noise). The "0.1–0.2% trainable" figures ARE the LoRA-only param counts. | ❌ **RETRACTED** (2026-07-04) | Correction note (2026-07-04) + "Attribution results" table |
| "First end-to-end evidence the MT-LNN inductive bias transfers to a real pretrained LM" / "gain grows with base size (−28% → −34%)" | The trend measures **plain LoRA fine-tuning**; the MT adapter transferred nothing measurable on in-window PPL. | ❌ **RETRACTED** | Correction note (2026-07-04) |
| "MT-LNN has O(1) working memory / long-context compression" (as a property of the **hybrid** M-series) | O(1) holds **only for the attention-free O-series**. The hybrid **still contains attention** → KV cache still grows → **not O(1)**. Its **training memory is a NEGATIVE** (uses MORE than a plain Transformer at every length, OOMs at 4096 too, ~1.6x slower). | ❌ **RETRACTED for the hybrid** (real for O-series only) | "Scaling to ~125M → --mode profile" (negative) + "--mode decode" (O-series positive) |
| "State compresses long context / out-of-window LM gains" | **NULL ×2**: standard chunked-streaming **−0.006 / −0.000**; TBPTT state-carry **+0.004 (noise)**. Full attention gains 0.89–0.94 PPL from 512→2048 that the state does **not** capture. The state is an **episodic key→value memory**, not compressed distributed context. | ❌ **NULL** | "Out-of-window streaming" + "State-carry (TBPTT) training" |
| "Orch-OR collapse gate / Φ̂ integration / anesthesia validation is a working consciousness biomarker and a unique selling point" | Modules are **INERT in the trained path**. AVP **FAILED**; Φ̂ **rises** under anesthesia (sign **inverted** vs theory). Φ̂ moves +8.499 signed only as a "hooks fire" artifact on toy activations with no real-data baseline (Kraskov bias at N=148, Lord et al.). | ❌ **INERT / net liability** — inspiration only, never load-bearing | "Anesthesia Validation Protocol" + "What this does NOT show" |
| "Optional bio modules (predictive coding, GWTB, world model, rhythm, Hebbian) improve quality" | All 5 are **PPL-neutral at 48M** (within ±0.3–0.5 noise band); the full stack costs **5.6% throughput** for nothing; predictive coding (the one ON by default) trends **negative**. Lean core trunk is best. | ❌ **NULL** (archived behind flags) | "O1 module switch-matrix" |
| "Adapter improves long-context / needle retrieval" | Within the 2048 window base and adapter both ~0.87–1.0 (**parity, inconclusive**); at 4096 both 0.000 (base RoPE limit, not adapter). Attribution now shows the adapter adds nothing anyway. | ⚠️ **INCONCLUSIVE / parity** | "Needle-in-a-haystack (CORRECTED)" |
| "ARR attention-free student matches teacher" | PPL **25.4 = 2.15× teacher** (11.8), still falling — converging with tokens, **not at parity**. ARR cross-window recall is **negative** at current budget (curriculum retry queued). | ⚠️ **research preview, not parity** | "Round 2/3 distillation" + "ARR-student recall — negative" |

---

## What we do NOT claim

- **We do NOT claim the MT adapter beats LoRA on perplexity.** On in-window LM PPL it adds ≈0 beyond LoRA. The old −28/−34% numbers are retracted.
- **We do NOT claim the hybrid (M-series) flagship is O(1).** It contains attention; its KV cache grows and its **training** memory is worse than a Transformer's. O(1) is an **inference** property of the **O-series (ARR) only**.
- **We do NOT claim long-context language-modeling gains.** Out-of-window LM is a double null. The state carries **discrete addressable key→value bindings**, not compressed distributed context.
- **We do NOT claim a working consciousness / integrated-information / Orch-OR result.** Those modules are inert in the trained path and the AVP Φ̂ sign is inverted. Microtubule/Orch-OR framing is **inspiration only**, not evidence.
- **We do NOT claim the optional bio modules improve quality.** They are PPL-neutral; shipped configs run the lean core.
- **We do NOT claim a 125M SOTA result.** The 125M win is vs this repo's simple-reference Transformer and (as of 2026-07-16) a width/depth-mismatched Mamba baseline, single seed each, undertrained — a real, consistent, budget-limited signal, not a converged or SOTA-competitive result.
- **fp16 robustness: root-caused and FIXED (2026-07-19).** MT-LNN's loss previously went non-finite mid-training under fp16 AMP (step 629 on Colab T4; reproduced locally at step 875/896 — the step drifts with data order). Instrumented diagnosis ruled out gradient explosion (grad-norm ~1.1 at failure) and activation overflow (peak 67.4 vs fp16's 65504 ceiling). **Actual cause:** `global_coherence.py` computed `(Q @ K^T) / scale`, doing the `d_head=64` accumulation *before* the down-scaling — with measured q/k projection peaks of 52.8/60.8 the intermediate product hits ~2e5 and overflows fp16 *inside the matmul*, after which `Inf * 0` against the causal mask in `_gate_energy` yields NaN that `sigmoid()` spreads through the layer. **Fix:** pre-scale (`(Q/scale) @ K^T`, 4 sites), select-instead-of-multiply in the gate reduction, fp32 accumulation, and a `clamp_min(1e-6)` replacing a `1e-9` epsilon that underflows to exactly 0 in fp16. **Verified:** the identical 2000-step recipe that died at step 875 now completes in full, `stable: true`, val PPL **257.91 vs Infinity** — and matches fp32's 257.48 on the same recipe to within 0.17% (inside the ±4.89 seed variance), i.e. fp16 is back at fp32 parity, not just non-crashing. Audit: every other attention surface uses `F.scaled_dot_product_attention`; `global_coherence.py` was the only hand-rolled one. Tool: `benchmarks/diagnose_fp16_divergence.py`. Long-horizon (20K-step) fp16 confirmation is still outstanding.
- **We do NOT claim the cloud-inject +13.3% as MT-adapter value.** It is 100% the `[Absorbed fact]` prompt template; the adapter row is identical to baseline.

---

## Product positioning (M-series vs O-series)

Two checkpoints, one codebase, deliberately different trade-offs — kept split so every claim stays attributable ([docs/PRODUCT_LINES.md](docs/PRODUCT_LINES.md)):

- **M-series — hybrid (attention + liquid adapter).** Cloud/GPU serving where full base quality matters. Its **unique** edge is cross-window / cross-session associative recall (0.56 vs 0.000 structural for attention/LoRA) at ~1% parameter overhead, plus lossless cross-session state snapshot/restore. It is **not** a perplexity win over LoRA and **not** O(1).
- **O-series — pure recurrent (ARR, attention-free).** Edge / CPU / low-power / unbounded-stream where KV-cache growth is disqualifying. Its **unique** edge is genuine **O(1) inference memory** (flat 0.381 MB, 1008x smaller than KV at 128k). It is a **research preview** at 2.15× teacher PPL — a token-budget gap, not a stability problem.

Everything else — the from-scratch 125M sample-efficiency result and the stability-at-scale result — is a shared architectural signal that motivates both lines.
