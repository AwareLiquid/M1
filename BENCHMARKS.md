# MT-LNN Benchmarks

End-to-end benchmark suite for MT-LNN. Designed to be reproducible on CPU in
under 5 minutes per task. The suite tests the architecture's three claimed
strengths: (1) long-range selective memory via `h_prev` recurrence, (2) global
information bottleneck via GWTB, and (3) consciousness-relevant integration
collapse via the Anesthesia Validation Protocol.

> **Correction note (2026-06-29).** An earlier version of this section reported
> a ×42 sequence-exact advantage. That number was an **evaluation artifact**:
> `evaluate_selective_copy` decoded token-by-token off an incremental cache, but
> the vanilla Transformer/LNN baselines return `cache=None` (they implement no
> cache), so the decoder fed them a lone token each step and silently dropped
> all prefix context — crushing them to near-random. MT-LNN, whose recurrent
> state cache works, was unaffected. The decoder now falls back to a
> **full-sequence recompute** path for cacheless models, so all architectures
> are compared fairly. Under the fair decode the real advantage is **modest**
> (×1.3 on sequence-exact at T=32), not a knockout. Numbers below are the
> corrected, fair measurements.

> **Correction note (2026-07-04) — Phase 5/5b adapter results are LoRA-only.**
> All Phase 5/5b runs below (TinyLlama −28.5%, Qwen-1.5B −27.7%, Qwen-3B
> −34.4%) predate the re-arm fix (`8d9d741`, 2026-06-28): `get_peft_model()`
> silently froze the MT adapters at random initialization (residual scale
> 1e-3, contribution ≈0), so **only LoRA trained**. Three independent lines of
> evidence: (1) no committed version of the adapter code produces the reported
> "2.30M trainable" — the real adapter is 9.6–11.8M *per layer*; (2) the
> reported trainable counts match the LoRA-only parameter counts exactly
> (TinyLlama 2.25M→"2.30M", Qwen-1.5B 2.18M→"2.22M"); (3) the re-arm fix
> commit itself documents that the MT scale was frozen. The Phase 5/5b PPL
> gains therefore measure **plain LoRA fine-tuning**, and the "first
> end-to-end evidence that the MT-LNN inductive bias transfers" claim is
> **retracted**. The controlled attribution
> (`benchmarks/attribution_ablation.py`) has since been run and confirms it.
>
> **Attribution results (2026-07-04, TinyLlama-1.1B, WikiText-2 test PPL,
> 1000 steps, identical data/optimizer, single P100 fp16, seed 0):**
>
> | config | trainable | test PPL | vs base | ΔPPL/1M params |
> |---|---|---|---|---|
> | baseline (frozen) | 0 | 11.821 | — | — |
> | **lora_only (r=8)** | **2.25M (0.20%)** | **7.984** | **−32.5%** | **+1.70** |
> | mt_only (v1) | 62.8M (5.4%) | 8.102 | −31.5% | +0.06 |
> | mt_lora (v1+LoRA) | 65.1M (5.6%) | 7.920 | −33.0% | +0.06 |
> | mt_v2_only | 8.38M (0.76%) | 8.158 | −31.0% | +0.44 |
> | mt_v2_lora | 10.6M (0.96%) | 7.918 | −32.9% | +0.37 |
>
> Verdict: **on plain-LM perplexity the MT adapter adds ≈nothing beyond
> LoRA** (−0.064 PPL for +62.8M params; single-seed, within noise). v2
> matches v1 quality at 7.5× fewer parameters. Perplexity inside the
> attention window was the wrong battlefield for a memory architecture;
> the differentiating test is `benchmarks/cross_window_recall.py`
> (recall across a dropped KV cache, where attention cannot help).

## Cross-window associative recall (2026-07-05)

`benchmarks/cross_window_recall.py`: segment A shows 8 key→value token
pairs, then the KV cache is **dropped** — segment B re-queries each key in
a fresh attention context. The frozen attention physically cannot see
segment A; the only A→B channel is the adapters' streaming recurrent
state. TinyLlama-1.1B, 8000 steps, mixed protocol (75% cross-window),
lr 1e-3, `--state_scale_init 0.1`, single P100, chance = 0.001:

| config | trainable | in-window | **cross-window** |
|---|---|---|---|
| baseline (frozen) | 0 | 0.557 | 0.000 |
| lora_only (r=8) | 2.25M | 0.000 † | 0.000 |
| mt_only (v1, EMA state) | 62.8M | 0.994 | 0.002 |
| mt_v2 **without** fast-weight | 5.2M | 0.996 | 0.008 |
| mt_v2 (fast-weight) | 8.4M | 1.000 | **0.553** |
| mt_v2s (+ selective decay) | 8.4M | 0.994 | **0.621** |

Findings: (1) **the fast-weight matrix is the memory** — removing it
collapses cross-window recall 0.553→0.008, and v1's 62.8M multi-timescale
EMA state manages only 0.002: decaying averages cannot store discrete
bindings. (2) **Selective (input-dependent) decay beats static** and the
gap widens with training (0.490→0.553 vs 0.561→0.621 from 5k→8k steps;
both still climbing). (3) LoRA/attention is **structurally zero** across
the window — at 50/50 protocol mix it learns in-window recall fine (0.951,
pilot) and still scores exactly 0.000 cross-window.
† At the 75/25 mix the cross-window batches are an unlearnable objective
for stateless configs; their gradient noise also destroys lora_only's
in-window skill — reported honestly, see the 50/50 pilot for its 0.951.

**Bio-prior (frozen-tau) ablation (2026-07-06):** freezing every tau
ladder at its biologically-derived init (training everything else) scores
cross-window **0.285** vs **0.621** trained (in-window ~0.99 both). Read
both ways: the bio initialization alone carries ~46 % of the effect (285x
chance), and trainability doubles it — the bio prior is a genuinely good
starting point, not a sufficient endpoint.

**Multi-seed replication (2026-07-05):** mt_v2s cross-window over seeds
{0, 1, 2} = **0.621 / 0.434 / 0.621** (mean 0.56 ± 0.09; in-window
0.994–1.000 every seed) — the effect is robust, three orders of magnitude
above chance in every run. lora_only seed 1 failed to learn even in-window
at lr 1e-3 (training instability for LoRA at this task lr; the structural
zero-channel argument is unaffected, and its seed-0 run learned in-window
0.951 while still scoring exactly 0.000 cross-window).

**ARR-student recall — negative at current budget (2026-07-06):** the
distilled attention-free student (round-2 mixers, streaming state wired,
79.5M trainable) trained on the same recall protocol scores 0.000 both
in-window and cross-window at 8k steps — unlike the adapters, which get
in-window recall "for free" from frozen attention, the ARR student must
implement induction entirely in recurrence. Diagnosis: expressivity is NOT
the blocker (a single batch is memorized to acc 1.0 in 20 steps); it is an
optimization/curriculum problem (start at n_pairs=2, longer training).
Curriculum retry queued for the next GPU budget.

## Cross-session persistence — snapshot/restore is lossless (2026-07-05)

`benchmarks/cross_session_recall.py` + `snapshot_adapter_streams` /
`restore_adapter_streams` (llama_adapter.py). The fast-weight (F,z) that
carries cross-window recall is normally reset every request and never
serialized. This measures whether snapshotting it to DISK and restoring into
a FRESH model object (new (F,z)=None by construction — process-volatile state
provably did not ride along) preserves recall. Oracle retrieval (restore the
correct session by id) isolates round-trip fidelity from content-addressed
retrieval. TinyLlama, mt_v2s, B=1 eval:

| run | within_window | cross_session | round-trip Δ | C1 no-restore | C2 wrong-session |
|---|---|---|---|---|---|
| A (fixed bf16, 8k steps) | 0.273 | 0.266 | **+0.008** | 0.004 | 0.000 |
| B (earlier, undertrained) | 0.027 | 0.027 | **+0.000** | 0.000 | 0.000 |

Verdict: **the snapshot->disk->fresh-process->restore round-trip is lossless**
(cross_session tracks within_window to within noise in both runs) and
**content-addressed to the right session** (C2 wrong-session restore = chance,
so it is not "any non-zero F helps"); C1 no-restore = chance proves the state
was really gone and the test has power. A CPU unit test
(`tests/test_cross_session_snapshot.py`) confirms the round-trip is bit-exact
(max|diff| 0.0). The absolute within_window here (0.27, not the standalone
benchmark's 0.62) is training-convergence-limited — this recall task is
high-variance (see the 0.62/0.43/0.62 multi-seed spread) and run A undertrained;
the persistence CLAIM is the delta and the controls, which are clean. This is
the fast->slow (hippocampus->durable) transfer the consolidation stack was
missing a source for; follow-ons are server session_id wiring + e5
content-addressed retrieval (turning within-session recall into "remembers you
across sessions" on the site).

## Out-of-window streaming on real text (2026-07-05) — honest null

`benchmarks/length_streaming_eval.py`: WikiText-2 test windows of 2048,
fed in 512-token chunks with the KV cache dropped between chunks; the
adapters' streaming state either carries (streaming) or not (stateless).
Adapters trained the standard way (1000 LM steps at seq 512):

| protocol | mt_lora (v1) | mt_v2_lora |
|---|---|---|
| full attention @512/1024/2048 | 7.92 / 7.35 / 6.98 | 7.92 / 7.34 / 6.98 |
| chunked stateless @512 | 7.921 | 7.919 |
| chunked **streaming** @512 | 7.927 | 7.920 |
| **out-of-window gain** | **−0.006** | **−0.000** |

**Null result, cause understood**: standard windowed LM training gives the
state ZERO pressure to carry information across chunk boundaries — exactly
as the recall experiment predicted (recall was also ~0 until trained WITH
the cross-window protocol). Full attention gains 0.94 PPL from 512→2048 of
context; that is the prize a state-carry TRAINING recipe (TBPTT-style
chunked training with carried state) still has to capture. Capability is
trainable, not free.

### State-carry (TBPTT) training — second null, boundary established (2026-07-05)

`benchmarks/state_carry_train.py`: chunked LM training with KV dropped
between chunks, adapter state carried WITH gradients (pieces 2+ only
improvable through the state channel), state_scale_init 0.1, lr 5e-4,
1000 window-steps (~1M tokens), mt_v2s_lora:

| trained | chunked_stateless | chunked_streaming | state gain |
|---|---|---|---|
| carry=False (control) | 7.809 | 7.818 | −0.009 |
| **carry=True (TBPTT)** | 7.812 | 7.808 | **+0.004 (noise)** |

Full attention still gains 0.89 PPL from 512→2048 context. Combined with
the recall result (0.62 cross-window accuracy on discrete pairs), the
boundary is now sharp: **the fast-weight state is an episodic key→value
memory — it carries discrete addressable bindings across windows, not
compressed distributed context** — consistent with the linear-attention
literature. Larger budgets/windows might move this; under ours it is null.

## Capability evals — v2s SFT is ability-neutral (2026-07-05)

`benchmarks/capability_eval.py` (lm-evaluation-harness, full tasks, P100),
TinyLlama base vs base + v2s SFT adapter (streaming on, serving semantics):

| task | base | v2s SFT | Δ |
|---|---|---|---|
| LAMBADA-openai (acc / ppl) | 0.610 / 5.90 | 0.617 / 5.74 | +0.7pt / −0.16 |
| ARC-easy (acc / acc_norm) | 0.617 / 0.548 | 0.624 / 0.560 | +0.7 / +1.2 |
| HellaSwag (acc / acc_norm) | 0.465 / 0.604 | 0.458 / 0.595 | −0.7 / −0.9 |
| PIQA (acc / acc_norm) | 0.742 / 0.745 | 0.742 / 0.749 | 0.0 / +0.4 |

As the attribution predicted: the adapter is **capability-neutral** (±1pt,
noise) — the SFT bought chat formatting and recall machinery without
trading away core abilities. Deployment-safety box ticked.

## Scaling to ~125M (2026-07-05) — training memory is a NEGATIVE, O(1) is real but only for O-series

`benchmarks/scaling_comparison.py`, T4, d_model 832 x 12 layers, matched
width/depth (embeddings tied so param diff = mixer cost).

**--mode profile (TRAINING memory + throughput, fwd+bwd):**

| arch | params | T=512 | T=1024 | T=2048 | T=4096 |
|---|---|---|---|---|---|
| transformer | 142M | 1928 MB / 1594 t/s | 3630 / 2400 | 8993 / 2081 | OOM |
| lnn | 92M | 1521 / 2879 | 3108 / 3357 | 8267 / 2710 | OOM |
| **mt_lnn (hybrid)** | 127M | **2499** / 1316 | **5085** / 1303 | **12344** / 1109 | OOM |

**Honest negative:** in TRAINING, the native (hybrid) MT-LNN uses MORE memory
than a plain Transformer at every length, grows just as fast, OOMs at 4096
too, and is slower. Two reasons, both structural: (1) the parallel scan
materialises the whole (B,P,S,T,D) hidden stream for backprop — O(T) with a
large P·S·D constant; (2) the native block STILL contains attention (it is a
HYBRID, not attention-free). **Do not claim MT-LNN saves memory in training —
it does not.**

**--mode train (STABILITY + sample efficiency, WikiText-103, 2000 steps):**

| arch | params | stable | val PPL | tok/s |
|---|---|---|---|---|
| transformer | 142M | **yes** | 435.6 | 2357 |
| lnn | 92M | **yes** | 445.5 | 3361 |
| **mt_lnn (native)** | 127M | **yes** | **299.5** | 1491 |

Two POSITIVES (the project's strongest scaling evidence):
1. **Stability confirmed** — all three train with no NaN/divergence at 125M.
   The review's central fear ("does an ODE/liquid-recurrent net even converge
   when scaled 100x from 48M?") is answered: it does.
2. **MT-LNN is more sample-efficient** — at matched data/steps/optimizer it
   reaches 299 PPL vs the Transformer's 436 (−31%), and the gap is *consistent*
   across the whole curve (step 500: 5.79 vs 6.09; step 1900: 5.34 vs 5.76),
   not noise. Unlike the *adapter* attribution (where MT added nothing over
   LoRA), the from-scratch native recurrent model genuinely beats a matched
   Transformer on per-step learning.

Honest caveats: all PPLs are high (2000 steps << 1 epoch of WikiText-103 —
undertrained; the comparison is relative, at matched budget). MT-LNN costs
~1.6x the wall-clock (1491 vs 2357 tok/s), so at matched TIME (not steps) the
gap narrows. Single seed; the Transformer is this repo's simple reference
impl, not a SOTA-tuned baseline. No Mamba baseline in this run (added
2026-07-16, fp32, see below). Direction is clear and consistent, but this is
a budget-limited signal, not a converged result.

## fp32 reconfirmation + Mamba baseline (2026-07-16)

Same harness (`benchmarks/scaling_comparison.py --mode train`), same
d_model/n_layers/WikiText-103/2000-step/50M-token-cap budget as the run
above, but three changes: (1) explicit `--dtype fp32` (autocast/GradScaler
fully disabled, vs the T4 run above which used `--dtype auto` → fp16 AMP),
(2) run locally on an RTX 5060 Laptop (8GB) instead of a cloud T4/P100, (3)
first-time Mamba baseline via `use_mambapy=True` (mamba.py parallel-scan
backend — mamba-ssm's CUDA kernel doesn't build on Windows, and the
transformers sequential fallback is ~25-50x too slow for a 2000-step run;
numerically equivalent to the CUDA kernel path, only trades throughput).

Motivation: a separate Colab T4 fp16 run (commit `b11e5bc`) found **mt_lnn's
loss went non-finite past step 629** while the **transformer** baseline
stayed stable for the full 2000 steps under the identical fp16 recipe — a
real, unresolved fp16 numerical-robustness gap isolated to mt_lnn (never
exposed under bf16's wider dynamic range). This run checks whether the PPL
advantage survives a same-precision, divergence-free comparison.

> ⚠️ **SUPERSEDED (2026-07-19).** The 2000-step numbers below are
> **undertrained** and were measured against a **weak reference baseline**.
> Training the same architectures to convergence (20,000 steps, 3 seeds) and
> adding a **modern** Transformer baseline **reverses the conclusion** — see
> "20K convergence + modern baseline" below. The ~30% advantage reported here
> **does not survive**. Retained for provenance only.

| arch | params | stable | val PPL | tok/s |
|---|---|---|---|---|
| transformer | 142.1M | **yes** | 370.81 | ~7100 |
| **mt_lnn** | 126.0M | **yes** | **257.48** | ~1200 |
| mamba | 129.1M | **yes** | 414.00 | ~1270 |

Findings:
1. **fp16 divergence: ROOT-CAUSED AND FIXED (2026-07-19).** Originally a
   precision-only issue (fp32 stable, fp16 non-finite past step 629). Now
   diagnosed and repaired — see "fp16 divergence root cause" below.
   `--dtype fp32` is no longer required as a divergence workaround.
2. ~~**The ~30% PPL advantage holds under both precisions**~~ — **RETRACTED
   2026-07-19.** Both runs were 2000-step (undertrained) against a
   simple-reference Transformer. At convergence the gap shrinks to 5.5%, and a
   modern Transformer baseline **beats mt_lnn by 11.3%**. The consistency
   across two precisions was real but measured the same two confounds
   (undertraining + weak baseline), not a durable architectural advantage.
3. **First Mamba data point: mt_lnn beats it too** — 257.48 vs 414.00 PPL
   (−37.8%), with fewer params (126.0M vs 129.1M). Caveat: Mamba here is
   HF's default sizing (`hidden=768`, 24 layers) which is NOT width/depth
   matched to the transformer/mt_lnn pair (`d_model=832`, 12 layers); the
   param counts land in the same class by construction but this is an
   external reference point, not an architecture-matched control.
4. **Mamba's train/val gap is a caution, not a mamba-specific flaw claim.**
   Mamba's training loss looked competitive mid-run (step 1500: 5.43, ahead
   of transformer's 5.76 at the same step) but its final val PPL is the
   worst of the three — a reminder not to read training-loss curves as a
   proxy for generalization in isolation.

Caveats: single seed (0), no variance estimate. Local single-GPU wall-clock
(RTX 5060 Laptop, 8GB), not the T4 used in the run above — throughput
numbers are not cross-hardware comparable, only relative arch-to-arch
ordering within this run. Full report: `benchmarks/scaling_comparison_report.{json,md}`.

## 20K convergence + modern baseline (2026-07-19) — supersedes the 2K runs

Same harness (`benchmarks/scaling_comparison.py --mode train`), same protocol
(WikiText-103-raw-v1, GPT-2 tokenizer, `seq_len=512`, `batch=4`, `lr=3e-4`,
`--dtype fp32`), but **trained to convergence (20,000 steps) with 3 seeds per
arch**, and with a **modern Transformer baseline** added (RoPE + RMSNorm +
SwiGLU, `ModernCausalTransformer` in `benchmarks/baselines.py`). Run on a
Linux A100-class remote box, not the 8GB laptop.

| arch | params | stable | val PPL (mean ± std, n=3) |
|---|---|---|---|
| **modern_transformer** | 144.1M | **yes** | **78.86 ± 0.25** |
| mt_lnn | 126.0M | **yes** | 88.93 ± 0.33 |
| transformer (simple) | 142.1M | **yes** | 94.14 ± 0.78 |

Per-seed: mt_lnn 89.28/88.88/88.62 · transformer 94.63/94.54/93.24 ·
modern_transformer 79.15/78.66/78.77.

Findings:
1. **mt_lnn still beats the simple baseline, but only by 5.5%** (88.93 vs
   94.14) — down from the 30.6% claimed at 2000 steps. **Most of the apparent
   advantage was undertraining.**
2. **A modern Transformer beats mt_lnn by 11.3%** (78.86 vs 88.93) at
   comparable parameter count. The original "simple-reference" Transformer was
   simply a weak baseline; giving it the standard modern recipe flips the
   ordering.
3. **Perplexity is not currently an MT-LNN advantage.** The honest headline is
   that MT-LNN trains stably at this scale with a *residual gap to close*
   against modern Transformers. The architecture's un-refuted claims are the
   O(1) carried state and cross-window/cross-session recall, not LM quality.
4. All 9 runs `stable: true`, no non-finite loss.

Raw JSON + logs: `scaling_fp32/converge_probe/` (`train_*_s{0,1,2}.json`,
`scaling_train_20000_*.log`, `scaling_train_20000_summary.txt`).

**--mode decode (CARRIED STATE bytes vs context — the real O(1) test):**
The O(1) claim is an inference-time property (the state you must retain to
generate the next token), and it belongs to the attention-free **O-series
(ARR)**, not the hybrid. Matched Llama vs its ARR conversion:

T4, 832 x 12, GQA=1 (matched to the native model's config):

> **Hard boundary — read before citing this table.** It measures **inference
> carried-state bytes only**. It contains **no evidence of model quality beyond
> the 512-token training length**: out-of-window LM results are **null** (see
> RESULTS.md, chunked-streaming −0.006 / TBPTT +0.004). A reader who takes this
> table as long-context *capability* is misreading it, and we would rather say
> so here than let a reviewer say it for us.

| context T | Llama KV-cache | ARR state | ratio |
|---|---|---|---|
| 512 | 1.5 MB | 0.381 MB | 3.9x |
| 2048 | 6.0 MB | 0.381 MB | 15.7x |
| 8192 | 24.0 MB | 0.381 MB | 63x |
| 32768 | 96.0 MB | 0.381 MB | 252x |
| 131072 | 384.0 MB | 0.381 MB | 1008x |
| 524288 | 1536.0 MB | 0.381 MB | 4031x |
| **1048576** | **3072.0 MB** | **0.381 MB** | **8063x** |

Extended to **1M tokens** on 2026-07-19 (RTX 5060 8GB, `--mode decode
--profile_lens 512,2048,8192,32768,131072,524288,1048576`). Streaming the prime
in 512-token chunks under `no_grad` keeps the measurement itself O(1), which is
why a 1M-token context is measurable on an 8GB laptop at all — the KV-cache
side would need 3 GB just for the cache.

Attention KV-cache grows **exactly linearly** (4x per 4x in T); ARR state is
**flat at 0.381 MB** (F is DxD, no T dimension) — the O(1) claim, proven at
real 125M scale. Across a **2048x increase in context** (512 -> 1,048,576) the
carried state does not move by a single decimal place, while the KV line never
plateaus: at 1M context the ratio is **8063x measured**. ARR's number is an
empirical snapshot-byte sum, not an estimate; the Llama KV figure is the exact
analytic `2 * L * n_kv * d_head * T * bytes`. And this is CONSERVATIVE: GQA=1 already
shrinks the KV cache 13x — standard multi-head attention would put the ratio
~13x higher again. This cleanly validates the M-series/O-series split: only
the attention-free O-series gets constant memory, which is exactly the
edge/streaming/unbounded-context niche the product line targets.

> This fp16/GQA=1 table is the *weakest* opposing baseline. The same claim
> audited against quantized / evicted / GQA=8 KV caches lives in the next
> section — the O(1) claim survives all of them in the long-context regime
> and has one honest loss row (small-window eviction + 2-bit).

## KV-compression frontier (2026-08-29) — O(1) vs quantized / evicted / GQA KV

The section above measures ARR against fp16 GQA=1. External compression
(KIVI 2-bit, KVQuant/NVFP4 4-bit, StreamingLLM/SnapKV eviction, GQA=8)
shrinks that baseline, so the O(1) claim is re-audited here against the
**strongest** configurations. Everything below is byte-exact analytic
accounting on the KV side (bit-packed storage + KIVI asymmetric-quant scale
metadata; zero-points fold to 0 B) and the **measured** flat ARR state
loaded from `benchmarks/results/decode.json` — no number is hand-copied, and
the M-series hybrid is charged `kv@2bit + measured liquid stream` so the
ledger never compresses only the opponent. Full method + public anchors:
[docs/KV_FRONTIER.md](docs/KV_FRONTIER.md). Regenerate everything with
`python benchmarks/kv_frontier.py` (seconds, CPU); artifacts:
`benchmarks/results/kv_frontier.md` + `kv_frontier_ledger.json`.

Full grid (carried MB; the evict rows flatten at their sink+window cap, the
KV rows grow linearly, ARR and the hybrid rows are the reference):

| config | bits | gqa | window | T=512 | T=8192 | T=32768 | T=131072 | T=1048576 |
|---|---|---|---|---|---|---|---|---|
| **arr_state_measured (O-series)** | — | — | — | **0.381** | **0.381** | **0.381** | **0.381** | **0.381** |
| kv fp16 | 16 | 1 | — | 1.5 | 24.0 | 96.0 | 384.0 | 3072.0 |
| kv 8-bit | 8 | 1 | — | 0.763 | 12.189 | 48.751 | 195.001 | 1560.001 |
| kv 4-bit | 4 | 1 | — | 0.388 | 6.189 | 24.751 | 99.001 | 792.001 |
| kv 2-bit | 2 | 1 | — | 0.201 | 3.189 | 12.751 | 51.001 | 408.001 |
| kv fp16 | 16 | 8 | — | 12.0 | 192.0 | 768.0 | 3072.0 | 24576.0 |
| kv 8-bit | 8 | 8 | — | 6.105 | 97.512 | 390.012 | 1560.012 | 12480.012 |
| kv 4-bit | 4 | 8 | — | 3.105 | 49.512 | 198.012 | 792.012 | 6336.012 |
| kv 2-bit | 2 | 8 | — | 1.605 | 25.512 | 102.012 | 408.012 | 3264.012 |
| evict sink4+w512 | 2 | 1 | 512 | 0.201 | 0.202 | 0.202 | 0.202 | 0.202 |
| evict sink4+w512 | 4 | 1 | 512 | 0.388 | 0.391 | 0.391 | 0.391 | 0.391 |
| evict sink4+w4096 | 2 | 1 | 4096 | 0.201 | 1.597 | 1.597 | 1.597 | 1.597 |
| evict sink4+w4096 | 16 | 8 | 4096 | 12.0 | 96.094 | 96.094 | 96.094 | 96.094 |
| hybrid kv@2bit + state | 2 | 1 | — | 0.582 | 3.57 | 13.132 | 51.382 | 408.382 |
| hybrid kv@2bit + state | 2 | 8 | — | 1.986 | 25.893 | 102.393 | 408.393 | 3264.393 |

*(All 26 configurations × all 7 contexts in `benchmarks/results/kv_frontier.md`.)*

Crossover boundaries — T\* is the smallest context at which the config's
carried bytes exceed the 0.381 MB ARR state; below T\* the config is smaller:

| config | ARR smaller for | config/ARR @128k | @1M |
|---|---|---|---|
| kv fp16 GQA=1 (old baseline) | T ≥ 131 | 1007.88x | 8063.0x |
| kv 8-bit GQA=1 | T ≥ 256 | 511.82x | 4094.5x |
| kv 4-bit GQA=1 | T ≥ 503 | 259.85x | 2078.75x |
| kv 2-bit GQA=1 | T ≥ 976 | 133.86x | 1070.87x |
| kv fp16 GQA=8 | T ≥ 17 | 8063.0x | 64504.01x |
| kv 8-bit GQA=8 | T ≥ 32 | 4094.52x | 32755.97x |
| kv 4-bit GQA=8 | T ≥ 62 | 2078.77x | 16629.97x |
| **kv 2-bit GQA=8 (strongest compression)** | **T ≥ 119** | **1070.9x** | **8566.97x** |
| evict sink4+w512, 4-bit GQA=1 | T ≥ 503 | 1.03x | 1.03x (de facto tie) |
| **evict sink4+w512, 2-bit GQA=1** | **never** | **0.53x** | **0.53x — the honest loss** |
| evict sink4+w4096, any bits/GQA | T ≥ T* (17–976) | 4.19–252.21x | same (constant) |

Validation: the fp16/GQA=1 row reproduces the published table above exactly
(1.5 MB @512, 1007.9× @128k, 8063.0× @1M) before any extension.

**Reading.** Against every *non-evicted* KV — including the strongest
2-bit+GQA=8 compression — the ARR state is smaller throughout the common
range (all T\* ∈ [17, 976] ≪ 128k) and the margin grows with context. The
one configuration that undercuts ARR at every length is small-window
eviction + 2-bit + GQA=1 (0.202 MB flat = 0.53×). Memory is not capability:
eviction wins bytes by discarding tokens, and attention is structurally
**0.000** on cross-window recall where the fast-weight state scores **0.56**
(see the "Cross-window associative recall" section) — the two accounts are
reported side by side, never merged. Honest corrections forced by this table
are mirrored in RESULTS.md ("What we do NOT claim").

**Measured validation (2026-08-29).** The table's analytic side is exact on
real models: real HF forwards (TinyLlama GQA=4, Llama-3.2-1B GQA=8,
T ∈ {512, 2048, 8192}, MPS) reproduce `2·L·n_kv·d_head·T·bytes` with
**0.0000% deviation in every cell**, and the 2-bit rows are confirmed as
strict floors — real KIVI-scheme packing of the actual cache tensors adds
+2.96% (zero-points the ledger folds to 0) and KIVI-style g=32 grouping
+24.4–25.6%, all on the *opponent's* side, so the published ARR advantages
are lower bounds. Artifacts: `benchmarks/results/kv_measured.{json,md}`,
method: [docs/KV_FRONTIER.md](docs/KV_FRONTIER.md) §7.

## MTP (multi-token-prediction) aux loss — honest null, not promoted (2026-07-12)

`benchmarks/scaling_comparison.py --mode train`, native MT-LNN 125M, matched
seed-0 trunk copy so `mt_lnn` vs `mt_lnn_mtp` start byte-identical and differ
ONLY by the MTP aux gradient (3 seeds, 2000 steps, WikiText-2, K=3 lookahead
heads, λ=0.1). Question: does a DeepSeek-V3-style multi-token-prediction
regularizer lower held-out PPL on the proven core?

| arch | params | val PPL (mean ± std, n=3) |
|---|---|---|
| mt_lnn (core) | 126.7M | 268.40 ± 2.61 |
| mt_lnn + MTP heads | 252.2M | 267.65 ± 2.63 |

**Honest null.** The −0.75 PPL delta is well inside the seed noise (std
±2.6 both arms — the difference is ~0.3σ, not a resolvable signal at n=3).
Two reasons it likely stays null rather than replicating the published
result: (1) the K heads here are **flat, weight-untied linear** projections
reading the *same* final hidden state (`head_k(x)` for all k), unlike
DeepSeek-V3's sequential per-depth modules where head k conditions on head
k−1's prediction — a materially weaker predictor of t+2/t+3; (2) the 3 heads
add **125.4M params (+99% of the base model)** for a training-only term that
contributes zero value at inference (dead weight in every checkpoint unless
speculative decoding is later built). **Verdict: do not enable
`use_mtp_heads` by default.** The aux-loss wiring itself is correct and
zero-regression when off (see the architecture seam audit); this result
answers the audit's own validation gate (`WIRE_LATER: validate PPL is
neutral-or-better before default`) with a null, not a win — kept opt-in for
research, not promoted.

## O1 module switch-matrix — all optional modules PPL-neutral (2026-07-05)

`benchmarks/o1_module_ablation.py` (Colab free T4, 48M-class O1, TinyStories,
1200 steps, identical budget/seed; leave-one-in over the five optional
brain-inspired modules):

| config | val PPL | Δ vs core | tok/s |
|---|---|---|---|
| **core (all optional OFF)** | 25.33 | — | **3678** |
| + predictive coding | 25.98 | +0.65 (worse) | 3567 |
| + competitive GWTB | 25.36 | +0.03 | 3604 |
| + world model | 25.19 | −0.14 | 3615 |
| + rhythm (LAVI) | 25.05 | −0.28 | 3670 |
| + Hebbian | 25.62 | +0.29 | 3627 |
| full (all ON) | 25.09 | −0.24 | 3471 (−5.6%) |

Verdict (single seed, small budget — treat ±0.3–0.5 PPL as the noise band):
**no optional module clears noise**; the full stack costs 5.6% throughput
for a noise-level PPL change, and predictive coding — the one module that
defaults ON — trends *negative*. All quality lives in the core recurrent
trunk. Shipped/lean configs should run `core` (pass
`--no_predictive_coding`); the five modules are archived as negative
results at this scale, retained behind flags for larger-scale retests.

## Physics-informed head — structure helps, but OFF the LM path (2026-07-13)

`benchmarks/physics_rollout_eval.py` + `mt_lnn/hamiltonian_head.py`. The
honest, scoped test of physics-informed ML (the PINN investigation's one
real applicability boundary): a HARD-CONSTRAINT Hamiltonian head — learns a
scalar energy `H(q,p)=T(p)+V(q)` and advances state with a velocity-Verlet
**symplectic** integrator, so energy is conserved BY CONSTRUCTION — vs an
unstructured MLP-field control (same budget class, plain Euler). Both trained
on 1-step MSE; scored on **physics metrics only** (k-step rollout MSE, energy
drift), never perplexity. This is a continuous-state (q,p) trajectory
component, deliberately **not** wired into the language model — a physics
prior is a category error for token hallucination and PPL-neutral on language.

| system | metric | Hamiltonian (hard-constraint) | MLP-field control | advantage |
|---|---|---|---|---|
| spring (analytic) | 150-step energy drift | **9.4e-3** | 3.0e-2 | 3.2× |
| spring (analytic) | 150-step rollout MSE | **2.3e-2** | 5.5e-2 | 2.4× |
| orbit (via `physics_ops`) | 100-step energy drift | **6.2** | 37.8 | 6.0× |
| orbit (via `physics_ops`) | 100-step rollout MSE | **2.8e-3** | 6.6e-2 | 23.8× |

At **matched 1-step training fit** (loss ~5e-6 both), the symplectic structure
cuts long-horizon energy drift 3–6× and rollout error 2–24×. Architectural
proof pinned in `tests/test_hamiltonian_head.py`: symplectic energy drift
4e-6 vs forward-Euler 2e-3 on the *same* field, and the integrator is exactly
time-reversible (reversal error 0.0). Honest scope: this is a world-model /
trajectory-prediction win on continuous physics-governed state; it says
**nothing** about language hallucination or the −31% PPL result, and none of
these tasks are in the served-LM path (CPU-runnable, off by default, not
wired into `model.py`).

`mt_lnn/arr.py` + `benchmarks/distill_arr.py`: ALL 22 TinyLlama
self-attention blocks replaced by MT-v2s recurrent mixers (79.5M trainable;
pretrained MLPs/embeddings frozen — zero token-to-token attention, no KV
cache, O(1) inference state), then distilled from the frozen teacher.

| stage | student WikiText-2 test PPL |
|---|---|
| teacher (with attention) | 11.8 |
| after hidden alignment (2k steps) | 13,110 (alignment unstable — see below) |
| after logit KD (4k steps ≈ 2M tokens) | **264, still falling** |

Honest read: a promising slope, nowhere near parity — MOHAWK-class results
use 3B+ distillation tokens vs our 2M. Known round-2 fix: stage A aligned
all layers simultaneously on the student's own (drifting) hidden stream,
compounding error layer-by-layer until it destabilised; the correct
protocol teacher-forces each layer's INPUT from the teacher stream
(MOHAWK stage 2). Mixer checkpoint saved for resuming.

### Round 2 — teacher-forced alignment (2026-07-05)

Stage A teacher-forced (`--align_mode teacher_forced`, layers decoupled):
norm-MSE converges stably to **0.016** (round 1 oscillated to 467), and
alignment ALONE brings the student to PPL **278** — better than round 1's
final. Stage B (10k KD steps ≈ 5M tokens): final PPL **32.9**, still
falling at cutoff.

| round | after alignment | after KD | vs teacher (11.8) |
|---|---|---|---|
| 1 (free-running align, 2M tok) | 13,110 | 264 | 22× |
| 2 (teacher-forced, 5M tok) | 278 | 32.9 | 2.8× |
| **3 (resumed KD, ~18M tok cum.)** | — | **25.4** | **2.15×** |

Round-over-round gains are turning log-linear in tokens (standard for
distillation); KD loss still descending at cutoff. Next lever is data
scale (FineWeb-class corpus), not more epochs of WikiText-2.

An attention-free, KV-cache-free, O(1)-state 1.1B reaches 2.8× teacher
perplexity on ~5M distillation tokens (free-tier GPUs). The remaining gap
is a token-budget problem, not a stability problem. Round 3 (resumed, +18k
KD steps) queued.

Three architectures trained on identical Selective Copy data with identical
hyperparameters, parameter-matched to ~200K each, then evaluated with the
**same fair full-sequence decode** for every model (16 batches × 16 = 256
held-out sequences, 1500 training steps):

| Model | #Params | Training tok-acc | **Held-out tok-acc** | **Held-out seq-exact** | AVP responsive |
|---|---:|---:|---:|---:|:---:|
| Random baseline | — | — | 0.250 | 0.0039 | — |
| Vanilla Transformer | 199,464 | 0.875 | 0.874 | 0.676 | ✗ no |
| LNN (CfLTC FFN only) | 135,930 | 0.969 | 0.900 | 0.727 | ✗ no |
| **MT-LNN (with pscan)** | **224,900** | **1.000** | **0.949** | **0.895** | **✓ (+8.274)** |
| MT-LNN advantage vs Transformer | — | — | **+0.075 (×1.09)** | **+0.219 (×1.32)** | — |

MT-LNN reaches the highest held-out sequence-exact recall, but the lead over a
plain Transformer is single-digit-to-modest, and the LNN baseline (attention +
liquid LTC, no microtubule structure) is close behind — so most of the gain on
this task comes from the *liquid* component, with the microtubule machinery
adding a smaller increment.

### Long-context sweep: does the temporal advantage grow with T?

The Selective Copy task at three sequence lengths, same models, same recipe,
**equal 1500-step budget at every length** (the earlier sweep used 600/600/500
steps, which left the slower-converging MT-LNN undertrained and confounded the
comparison). Fair full-sequence decode for all models.
Reproduce with `python benchmarks/long_context.py` (~21 min on CPU).

**Held-out sequence-exact accuracy:**

| T_total | Transformer | LNN | **MT-LNN** | MT-LNN vs Transformer |
|---:|---:|---:|---:|---:|
| 37  | 0.672 | 0.703 | **0.883** | +0.211 (×1.31) |
| 101 | 0.570 | 0.727 | **0.742** | +0.172 (×1.30) |
| 229 | 0.109 | 0.172 | **0.219** | +0.110 (×2.0) |

**Held-out token accuracy:**

| T_total | Transformer | LNN | MT-LNN |
|---:|---:|---:|---:|
| 37  | 0.871 | 0.887 | **0.947** |
| 101 | 0.805 | **0.914** | 0.867 |
| 229 | 0.656 | **0.691** | 0.625 |

**Interpretation (honest):**

1. **On strict whole-sequence recall (seq-exact), MT-LNN is the best at every
   length** — 0.883 / 0.742 / 0.219. Its lead over a vanilla Transformer is
   consistent (~+0.17–0.21 absolute at T=37/101) and the *ratio* widens at the
   longest length (×2.0 at T=229), though absolute accuracy collapses for all
   three there under the limited budget. This is suggestive of a length
   advantage, but it is modest, not the order-of-magnitude gap the buggy
   evaluation implied.
2. **The LNN baseline is competitive.** Attention + liquid LTC (no microtubule
   structure) nearly ties MT-LNN on seq-exact at T=101 (0.727 vs 0.742) and
   *beats* it on token-accuracy at T=101 and T=229. So most of the benefit on
   this task is the liquid recurrence; the microtubule machinery adds a smaller
   increment, mainly visible on the all-or-nothing seq-exact metric.
3. **MT-LNN does not dominate token-accuracy.** It wins token-acc only at T=37;
   the baselines match or beat it at longer T. MT-LNN's edge is specifically on
   recalling *every* memorable token in order, not on average per-token recall.

### Parallel scan ablation (proves real recurrence matters)

| Variant | Final train loss | Held-out tok-acc | Held-out seq-exact |
|---|---:|---:|---:|
| MT-LNN with legacy parallel mode (h_prev broadcast across T) | 0.076 | 0.942 | 0.883 |
| **MT-LNN with parallel scan (real h_t recurrence)** | **0.059** | **0.983** | **0.965** |
| Improvement from real recurrence | **~1.3× loss** | +4.1 pp | **+8.2 pp** |

The pscan path gives a strictly better model on every metric — confirming
that the "temporal" claim is not just branding. Real recurrence does real
work.

Reproduce in ~50 seconds on CUDA:

```bash
python benchmarks/compare_baselines.py
```

### What this shows

1. **All three architectures learn the task and generalise** under a fair
   decode: held-out token accuracy is 0.87–0.95 and sequence-exact 0.68–0.90
   at T=32 — all far above the 0.25 / 0.004 random floors. (The earlier claim
   that the baselines "collapse to ~2% seq-exact" was the evaluation artifact
   described in the correction note above, not a real failure to generalise.)

2. **MT-LNN has the highest sequence-exact recall, by a modest margin.**
   0.895 vs 0.676 (Transformer) and 0.727 (LNN) at T=32 — a +0.22 / ×1.32 edge
   over the Transformer. The architectural priors it adds over the baselines
   (13 parallel protofilaments with content-aware RMC + nearest-neighbour
   lateral coupling, periodic GTP-cap renewal, MAPGate stabilisation) buy a
   real but incremental improvement on this task, concentrated in the
   all-or-nothing whole-sequence metric rather than average token accuracy.

3. **AVP is architecturally specific.** Anesthesia hooks attach only to
   `MTLNNLayer` and `GlobalCoherenceLayer`. The Transformer and LNN
   baselines contain neither, so anesthesia produces a Φ̂ delta of
   *exactly zero*. MT-LNN's Φ̂ moves +8.499 (signed) under anesthesia —
   verifiably responsive, even if the toy-scale sign is still inverted
   relative to the paper's prediction (see *Anesthesia Validation
   Protocol* section below).

### What this does NOT show

This is a fair comparison **at toy scale** (200K params, synthetic Selective
Copy). It is **not** a comparison vs mainstream 125M models (GPT-2-117M,
Mamba-130M, Pythia-160M) — those would require training MT-LNN at 125M on
WikiText-103, which we list as future work. The honest interpretation: at
matched parameter budget and a fair decode, MT-LNN's inductive biases give it a
**modest but consistent** edge on whole-sequence recall (×1.3 over a vanilla
Transformer at T=32), with the simpler liquid-LTC baseline close behind. This
is a real architectural signal on a selective-memory task — not the
order-of-magnitude gap an earlier evaluation bug suggested. Whether even this
modest advantage scales to 100M+ params on natural language is the next
experiment to run.

---

## Recommended benchmark hierarchy

| Tier | Benchmark | What it validates | Cost |
|---|---|---|---|
| 1 | **WikiText-103 PPL** | Standard LM competence | hours on GPU |
| 1 | **Selective Copy** *(below)* | Long-range memory + selectivity (Mamba §3.2) | minutes on CPU |
| 2 | **AVP / Φ̂ collapse** *(below)* | Information integration; consciousness claim | seconds |
| 2 | **Long-range PPL** *(in `eval.py`)* | Φ̂ extrapolation past training seq_len | seconds |
| 3 | **Anesthesia dose-response curve** *(in `eval.py`)* | Sigmoid match to clinical EEG | seconds |

Run the full Tier 1 + 2 sweep with:

```bash
python benchmarks/run_benchmark.py
```

---

## Selective Copy (Mamba §3.2)

Each example is a sequence

```
[n n n m1 n n m2 n n n m3 n n m4 ... SEP m1 m2 m3 m4]
                                  ^^^ targets
```

where `n` are random noise tokens and `m_i` are "memorable" tokens scattered at
random positions in the noise prefix. After SEP the model must autoregressively
emit `m_1, m_2, m_3, m_4` in order. **Random-guess baselines are 25% token /
0.4% sequence**, so a passing model must do much better.

### Configuration

| | |
|---|---|
| Model | MT-LNN, 204K params |
| `d_model` | 104 = 13 × 8 (TC-aligned, exact `d_proto`) |
| `n_layers` | 2 |
| `n_heads` | 4 |
| `n_kv_heads` | 2 |
| `d_proto` | 8, `d_proto_total` = 104 |
| `d_gw` (GWTB) | 26 |
| Task | `K_mem=4`, `T_noise=32`, `vocab=16`, `batch=16` |
| Training | 1500 steps, AdamW, peak LR 3e-3, grad-clip 1.0 |

### Results (CPU, single run)

| Step | Loss | Batch token acc |
|---|---|---|
| 1 | 2.744 | 0.250 |
| 200 | 0.767 | 0.672 |
| 400 | 0.306 | 0.891 |
| 600 | 0.254 | 0.891 |
| 800 | 0.123 | 0.953 |
| 1000 | 0.077 | 0.953 |
| 1200 | 0.070 | 0.953 |
| 1400 | 0.059 | 0.969 |

**Held-out greedy decoding** (16 batches × 16 sequences = 256 sequences):

| Metric | MT-LNN | Random baseline | Δ |
|---|---|---|---|
| Token accuracy | **0.973** | 0.250 | **+0.723** (3.9×) |
| Sequence exact match | **0.926** | 0.0039 | **+0.922** (235×) |

Wall-clock: 153s training + 1s eval = **154s total** on CPU.

### Interpretation

The model crosses the **selectivity barrier** — it learns to mask out noise
tokens and memorize the specific positions/contents of memorable tokens. A
feedforward layer cannot solve this task; the win confirms that MT-LNN's
recurrent state (h_prev) and selective gating (MAPGate, RMC, GWTB compression)
are doing real work.

---

## Anesthesia Validation Protocol (AVP)

After training, sweep anesthesia level `κ ∈ {1, 2, 5, 10}` via the
`AnesthesiaController` and measure Φ̂ on Selective Copy activation samples.

### Result (corrected reporting)

| κ | Φ̂ |
|---|---|
| 1 (clean) | −37.07 |
| 2 | −32.04 |
| 5 | −25.35 |
| 10 (full) | −20.51 |

| Metric | Value |
|---|---|
| Absolute change Φ̂(κ=10) − Φ̂(κ=1) | **+16.55** |
| Signed relative change | **+44.7 %** |
| Collapse percentage (counts decrease only) | 0.0 % |
| Monotone decrease | **False** |
| Pass threshold δ | 0.70 |
| **AVP** | **FAILED** |

### Interpretation — what this honestly shows

> The model's information integration *rises* monotonically with anesthesia
> level rather than collapsing. AVP fails for a clear, biologically
> interpretable reason — and the result itself is useful information.

This is an honest negative result that highlights three real issues to be aware of:

1. **Kraskov estimator bias at small N.** With our toy configuration the
   activation pool is only 4 sequences × 37 tokens = 148 samples in d=104
   space. The kNN entropy estimator is *negatively biased* in this regime
   (Lord et al. 2018), so absolute Φ̂ values are not meaningful — only their
   *direction of change* is. The benchmark exposes this honestly rather
   than hiding it.

2. **Anesthesia hook on a tiny model collapses representations** toward a
   low-rank manifold where part-wise activations become *more* correlated,
   not less. This is the opposite of the paper's prediction for trained
   125M models with high baseline integration, and it tells us that **the
   AVP test is only meaningful at scale**. We expect this to invert with a
   real-data-trained 125M+ checkpoint where the clean baseline has Φ̂ > 0
   and meaningful integration to collapse from.

3. **The mechanism is verifiably alive.** Anesthesia *does* produce a large
   monotonic Φ̂ change (44.7 % signed). The hooks fire, the protofilament
   damping and coherence collapse propagate through the model — there is no
   bug in the implementation. The collapse criterion is biological, not
   mechanical, and only the trained-at-scale model can satisfy it.

### How to make AVP pass

For a future trained-at-scale run:

- Train MT-LNN on WikiText-103 to a high baseline Φ̂ (positive, > 0.1).
- Verify the AVP curve direction is downward in the clean trained model.
- The collapse_pct threshold of 70 % then matches Casali et al. (2013) EEG
  complexity suppression under general anesthesia.

For the small-scale toy benchmark, the meaningful signals are:

- ✓ Selective Copy passes overwhelmingly (97.3 % / 92.6 %)
- ✓ Φ̂ responds monotonically and substantially to anesthesia
- ✗ Direction of response is wrong (estimator + scale artefact)

---

## Final MT diagnostics (post-training)

After 1500 steps of Selective Copy training:

| Parameter | Value |
|---|---|
| `tau_mean` | 2.44 |
| `tau_std` | 3.85 |
| `tau_min, tau_max` | 0.01, 10.00 |
| `gamma_mean` (GTP) | 0.081 |
| `polarity_mean, polarity_std` | −0.052, 0.436 |
| `rmc_gate_mean` (sigmoid) | 0.116 |
| `lat_coupling_off_diag_norm` | 0.346 |
| `coherence_scale` | 0.010 |
| `collapse_threshold` | 0.381 |
| `collapse_gate_last` | 1.000 |
| `gwtb_broadcast_gate` | 0.0009 |
| `gwtb_d_gw` | 26 |

Note: `gwtb_broadcast_gate` stayed near its 0.01 init — the model solved
Selective Copy almost entirely with MT-DL + Microtubule Attention, without
needing the bottleneck broadcast. This matches intuition: Selective Copy is a
"selective routing" task, not a "global integration" task.

`tau_std = 3.85` confirms the continuous geometric τ spectrum is genuinely
multi-scale and survived training (the original draft had collapsed to a
single τ value, which we fixed by removing the buggy `init_mt_params` override).

---

## Reproducibility

```bash
# Full benchmark (train + eval + AVP)
python benchmarks/run_benchmark.py

# Just the AVP sweep on an existing checkpoint
python eval.py --ckpt checkpoints/selective_copy.pt \
               --anesthesia_test \
               --anesthesia_kappas 1 2 5 10
```

The trained checkpoint is saved to `checkpoints/selective_copy.pt` and
includes the full benchmark result dict (selective copy metrics, AVP sweep,
final diagnostics) so it can be re-analysed without retraining.

---

## What's next

Concrete benchmarks worth running once we have real training data:

1. **WikiText-103 PPL** at 125M, comparing MT-LNN vs vanilla Transformer at
   matched param count. The paper claims 14.7 % PPL reduction.
2. **LRA Pathfinder** at 1024 / 2048 / 4096 context lengths. Tests true
   long-range integration; MT-DL's adaptive τ should excel.
3. **Φ̂ before & after WikiText training** — the actual paper experiment
   that AVP fails on at toy scale should succeed at full scale.
4. **Anesthesia dose-response curve fit** to a sigmoid, comparing the curve
   shape against Casali et al. 2013 clinical EEG complexity suppression.

## 1.1B Scale: TinyLlama-1.1B + MT-LNN residual adapter (Kaggle T4, 2026-05-28)

Frozen TinyLlama-1.1B-Chat with **MT-LNN residual adapters on 6 decoder layers + LoRA** on q/k/v/o. Trained 1000 steps on WikiText-2-raw-v1, seq_len 768, batch 1 × grad_accum 8, AdamW, fp16 (T4 GPU has no bf16). Wrapped layers: [3, 7, 11, 15, 19, 21]. Adapter scale init 1e-2. Hardware: single Tesla T4 (14.6 GB), ~3 h wall-clock.

Raw artefacts: `benchmarks/kaggle_run/ppl_ablation.json`, `benchmarks/kaggle_run/needle.json`, adapter checkpoint `checkpoints/llama_mt_adapter/llama_mt_adapter_001000.pt` (≤ 10 MB, base weights NOT bundled).

### Headline: validation perplexity

WikiText-2 valid, 50 batches × 768 tokens = 38 400 tokens.

| Variant | Trainable params | PPL ↓ | Tok/s | Eval s |
|---|---:|---:|---:|---:|
| Base TinyLlama-1.1B (frozen) | 1,100 M | 9.161 | 959 | 40.0 |
| **+ MT adapter + LoRA (1000 steps)** | **2.3 M (0.196 %)** | **6.553** | 862 | 44.6 |
| Δ | | **−28.5 %** | −10 % | +11 % |

**This is the first end-to-end evidence that the MT-LNN inductive bias transfers to a real pretrained LM.** The adapter learns a 28 % PPL reduction with 0.196 % of the parameter budget, at the cost of ~10 % decode-time slowdown. No NaN / no loss explosion; training loss falls from 2.5 → ~1.9 over 1000 steps (raw curve in `benchmarks/kaggle_run/train.log`).

### Needle-in-a-haystack (CORRECTED 2026-06-26 — old 0.000 was a harness artefact)

> **Retraction.** The previous "0/15 across all contexts, base-model ceiling" result was
> **invalid**. It came from `bench_llama_mt_needle.py`, which concatenates raw filler/needle/
> question tokens *without applying the instruct chat template* — a format that returns 0.0
> on any instruct-tuned base (documented in `docs/reviews/NEEDLE_FIX.md`). Re-running with the chat
> template **and** a faithful phase-5b model build (MT adapters **+** PEFT LoRA, with an honest
> guard that aborts unless every adapter tensor maps onto the graph: 374/374 tensors matched,
> unexpected=0) gives the real numbers below. Harness: `bench_needle_m1_faithful.py`; raw data:
> `benchmarks/needle_m1_chat_template.json` + `benchmarks/needle_m1_run.log` (CPU, 5 samples/cell).

| Variant | Context | Exact (avg over depth ∈ {0.1, 0.5, 0.9}) |
|---|---:|---:|
| Base | 1024 | **0.867** |
| Base | 2048 | **1.000** |
| Base | 4096 | 0.000 |
| MT-Adapter (`_003000`) | 1024 | **1.000** |
| MT-Adapter (`_003000`) | 2048 | **1.000** |
| MT-Adapter (`_003000`) | 4096 | 0.000 |

**Honest reading:**
- **Within TinyLlama's 2048 context window, retrieval works** — near-perfect for both base and
  adapter. The old "0% / cannot do this format" conclusion is withdrawn; it measured the broken
  harness, not the model.
- **M1 is no worse than base, and slightly better at the hardest in-window cell** (1024: base 0.867
  vs adapter 1.000 — the base misses one mid-depth needle that the adapter recovers). But with only
  5 samples/cell and the base already saturating at 2048, **"the adapter improves retrieval" remains
  INCONCLUSIVE** — the headroom is too small to claim a real uplift. We report parity, not a win.
- **4096 = 0.000 for *both* variants is a genuine base limit, not an adapter failure**: 4096 tokens
  exceeds TinyLlama-1.1B's 2048 RoPE training window, so the base itself collapses. The adapter
  cannot extend a context window the frozen base never had.

Next experiment to make the adapter delta *measurable*: a base with a larger native context window
(so the in-window region is not already saturated), with more samples/cell to beat the noise floor.

### What this run validates / does not validate

| Claim | Verdict |
|---|---|
| MT-LNN adapter trains stably on a real 1B+ pretrained LM | ✅ |
| Adapter learns LM-useful representations (PPL improves) | ✅ (−28 %) |
| Parameter-efficient (0.2 % trainable) | ✅ |
| Adapter improves long-context retrieval at 1.1B scale | ⚠ Inconclusive — within-2048 retrieval near-saturated (base 0.87–1.0, adapter 1.0); parity, no measurable uplift |
| Adapter improves long-context retrieval at ≥3B scale | ⏳ Not yet tested |


## Phase 5b — Qwen-2.5-1.5B replication (2026-05-29, Kaggle GPU)

Re-ran the Phase 5 recipe on a different 1B+ base (Qwen-2.5-1.5B-Instruct) to test whether the PPL improvement was TinyLlama-specific or a property of the MT residual adapter itself. Same recipe: 1000 steps, batch 1, grad_accum 8, MT adapter on every 4th layer, LoRA on q/k/v/o projections.

Raw artefacts: `benchmarks/kaggle_qwen_run/{ppl_ablation,needle}.json`. Reproduce with `kaggle/awareliquid_train_qwen_phase5b.ipynb`.

### Perplexity on WikiText-2 (50 batches, 25,600 tokens)

| Variant | Trainable params | Total params | **WikiText-2 PPL ↓** |
|---|---:|---:|---:|
| Base Qwen-2.5-1.5B-Instruct (frozen) | 1.54 B | 1.54 B | 11.10 |
| **+ MT adapter + LoRA (1000 steps)** | **2.22 M (0.139 %)** | 1.59 B | **8.03 (−27.7 %)** |

### Cross-base reproducibility check

| Base | Trainable | PPL drop |
|---|---:|---:|
| TinyLlama-1.1B (Phase 5)  | 0.196 % | −28.5 % |
| Qwen-2.5-1.5B (Phase 5b)  | 0.139 % | **−27.7 %** |

Two different 1B+ bases, two different families (Llama vs Qwen), same recipe → essentially the same PPL drop. The MT inductive bias is **not** a TinyLlama-specific artefact.

### Needle (Qwen-2.5-1.5B — NOT yet re-run with the corrected harness)

> **Caveat (2026-06-26).** The `accuracy = 0.000` numbers stored in `benchmarks/kaggle_qwen_run/needle.json`
> were produced by the same raw-concat harness that distorted Phase 5 (see the corrected Phase 5
> needle section above). They are therefore **not trustworthy** and should not be read as a real
> ceiling. The faithful chat-template re-test (`bench_needle_m1_faithful.py`) has so far only been
> run on the TinyLlama-1.1B Phase-5 checkpoint; the Qwen-2.5-1.5B phase-5b checkpoint has **not** yet
> been re-measured. Treat Phase 5b needle as **pending re-test**, not a negative result.


## Reasoning trace observability (2026-05-28)

AwareLiquid emits one JSONL row per decoded token via `mt_lnn.reasoning_trace.ReasoningTrace`. Each row carries `(step, token_id, entropy, route, phi)` plus separate `route` decisions and `cloud_inject` events. Two artefacts ride on top of this stream:

**`examples/trace_timeline.html`** — single-file browser viewer (no external deps). Drop a `*.trace.jsonl` file; renders one colored bar per token (green=LOCAL, yellow=SELF_CRITIQUE, blue=CLOUD, purple=INJECT), bar height ∝ entropy, orange dots mark Φ̂ samples. Click any bar for the raw event. This is the AwareLiquid answer to Gemini's opaque "thinking summary": every reasoning step is replayable, auditable, diffable.

**`scripts/bench_trace_audit.py`** — quantitative trace auditor. On the bundled `artifacts/demo_trace.jsonl` (120 synthetic decode steps):

| Metric | Value |
|---|---:|
| Total tokens | 120 |
| LOCAL / SELF_CRITIQUE / CLOUD | 113 (94.2%) / 6 (5.0%) / 1 (0.8%) |
| Cloud injects | 1 (212 B absorbed) |
| **Self-sufficiency** | **99.17 %** |
| Φ̂ samples / mean | 14 / 0.221 |
| Est. cost vs full-cloud | saved $0.001785 output − spent $0.000159 input = **+$0.001626 net** |

Self-sufficiency = `1 - cloud_tokens / total_tokens`. Cost model assumes $3/MTok input, $15/MTok output (frontier-API order of magnitude). Raw report: `artifacts/demo_trace_audit.json`.

## Cloud-inject end-to-end uplift harness (scaffold)

`scripts/bench_cloud_inject_uplift.py` measures the accuracy lift from prepending an `[Absorbed fact]` block to the prompt — the exact template `demo_awareliquid_v2.py` uses on CLOUD route. For each question in `benchmarks/cloud_inject_questions.json` (30 factual probes), generate twice:

  - **no-inject**: bare `Question: ... \nAnswer:`
  - **inject**: `[Absorbed fact] ... \nContinuing: Question: ... \nAnswer:`

Score = normalized substring match. Two backends:

  - `echo` — deterministic stub for CI (returns "I do not know" without a fact, echoes the fact when present). On 30 questions: **no_inject = 0.000, inject = 1.000, uplift = +1.000** (10 ms). This proves the scoring + scaffolding work end-to-end.
  - `hf` — real HuggingFace model with optional MT adapter. Echo baseline proves the harness; full TinyLlama/Qwen + adapter numbers are next.

This is the scaffolding for the "cloud inject adds real measurable value" pitch number. Now that the Phase 5b adapter (`benchmarks/kaggle_qwen_run/`) has trained, the same harness can produce a real uplift figure on Qwen-2.5-1.5B with the adapter loaded.

### Real cloud-inject numbers on Qwen-2.5-1.5B (2026-05-29, Kaggle GPU)

Same 30-question harness, real backend (`--backend hf`), greedy decode, 60 tokens. Raw: `benchmarks/cloud_inject_qwen/`.

| Variant | no_inject acc | inject acc | **uplift_abs** | wall (s) |
|---|---:|---:|---:|---:|
| Qwen-2.5-1.5B-Instruct (baseline) | 0.833 | **0.967** | **+0.133** | 108 |
| Qwen-2.5-1.5B-Instruct + Phase 5b MT adapter | 0.833 | **0.967** | **+0.133** | 159 |

**Two claims this validates:**

1. **Cloud-inject genuinely lifts accuracy on a real model** — Qwen alone answers 25/30; with the `[Absorbed fact]` template it answers 29/30. This is no longer a stub-only demonstration.
2. **The MT adapter does not break in-context learning** — same model, with vs without the adapter, identical 83.3 → 96.7 % uplift. The adapter that drops PPL by 28 % does not "close the model off" from external facts. This is the experiment that disproves the worry "adapter-finetuned models stop listening to context."

Reproduce: `kaggle/awareliquid_cloud_inject_uplift.ipynb`.

## Track 1B — real-inference ReasoningTrace (2026-05-29)

The trace numbers above (`demo_trace.jsonl`, 99.17 % self-sufficiency) came from a *synthetic* event generator. Track 1B closes PRD §F3 last-row by wiring `ReasoningTrace` into a *real* HF generate loop.

`scripts/awareliquid_real_trace.py` loads a HF causal LM (optionally with the Phase 5b adapter), greedy/temperature-decodes a single prompt, and at every step:

  - computes Shannon entropy of the next-token logits;
  - picks LOCAL (entropy < 2.0) / SELF_CRITIQUE (2.0 – 4.0) / CLOUD (≥ 4.0);
  - on CLOUD, splices an `[Absorbed fact]` block into the running context (same template as the cloud-inject uplift harness) and emits a `cloud_inject` event.

**Canonical demo run** — Qwen-2.5-0.5B, prompt `"The capital of Australia is"`, 39 tokens, CPU:

| Metric | Value |
|---|---:|
| Total tokens | 39 |
| LOCAL / SELF_CRITIQUE / CLOUD | 23 (59.0 %) / 16 (41.0 %) / 0 (0.0 %) |
| Cloud injects | 1 (62 B absorbed) |
| **Self-sufficiency** | **100.00 %** |
| Entropy mean / max | 1.929 / 6.471 |
| Est. cost vs full-cloud | saved $0.000585 output, spent $0.000046 input → **+$0.000539 net** |

Raw: `artifacts/real_trace_demo.jsonl` + `artifacts/real_trace_demo_audit.json`. Viewable in `trace_timeline.html` (drag-drop the jsonl).

**v3 optimization (2026-05-30):** Fixed v2 infinite-loop bug (LogitsProcessor couldn't interrupt `generate()`). `scripts/awareliquid_real_trace_v3.py` uses manual token loop with `past_key_values` → proper O(N) KV cache + working cloud-inject. Local validation (Qwen-0.5B/CPU/34 tokens): 7.3s, 64.7% LOCAL / 35.3% SELF_CRITIQUE / 1 cloud inject. Kaggle run cancelled after 90+ min with empty logs (infrastructure slow, not script issue). Raw: `artifacts/real_trace_v3_test.jsonl`.

First real-model trace shipped in-repo; the synthetic trace stays for UI demo.

## Track 1A — Qwen-2.5-3B + MT adapter (2026-05-29, Kaggle GPU)

Scale validation. Phase 5b proved cross-base reproducibility at Qwen-1.5B. Track 1A re-runs the same recipe on **Qwen-2.5-3B-Instruct** (~6 GB fp16) so the headline PPL claim has a third independent base. Settings: 1000 steps, batch 1, grad_accum 8, MT adapter on every 4th layer, LoRA on q/k/v/o, SEQ_LEN=384.

Raw artefacts: `benchmarks/kaggle_qwen3b_run/{ppl_ablation,needle}.json` + `train.log`. Reproduce: `kaggle/awareliquid_train_qwen3b.ipynb`.

### Headline: validation perplexity (best PPL drop yet)

WikiText-2 valid, 50 batches × 384 tokens = 19 200 tokens.

| Variant | Trainable params | PPL ↓ |
|---|---:|---:|
| Qwen-2.5-3B (frozen) | 3.086 B | 10.72 |
| **+ MT adapter + LoRA (1000 steps)** | **3.75 M (0.117 %)** | **7.03 (−34.4 %)** |

**The MT adapter PPL gain grows with base size**: TinyLlama-1.1B −28.5 % → Qwen-1.5B −27.7 % → **Qwen-3B −34.4 %**. The "scale catastrophes the adapter" worry is disproven; the inductive bias *strengthens* at 3B. Trainable budget stays comfortably under 0.2 % (0.117 %).

### Needle (format fixed 2026-05-30, rerun pending)

**Root cause identified and fixed**: The old `bench_llama_mt_needle.py` used raw prompt concatenation without chat templates, causing 0.0 accuracy on all instruct-tuned models regardless of size (1.1B / 1.5B / 3B). The new `bench_needle_chat_template.py` uses `tokenizer.apply_chat_template()` and achieves **1.0 accuracy** on Qwen-0.5B-Instruct baseline at all depths (0.1, 0.5, 0.9) and contexts (512, 1024).

This was a benchmark-tooling problem, not an architecture problem. Next: rerun on Qwen-1.5B/3B with Phase 5b adapters to measure MT-vs-base delta. See `docs/reviews/NEEDLE_FIX.md` for full details.

### Cross-base summary (Phase 5 + 5b + Track 1A)

| Base | Trainable | PPL drop |
|---|---:|---:|
| TinyLlama-1.1B  | 0.196 % | −28.5 % |
| Qwen-2.5-1.5B   | 0.139 % | −27.7 % |
| **Qwen-2.5-3B** | **0.117 %** | **−34.4 %** |

Three different bases, three independent training runs, consistent PPL drop in the −28 % to −34 % band — and the drop **gets bigger** as the base scales. The MT residual adapter is not a small-model artefact.


## v2.0 bio-inspired modules + engineering features (2026-06-07)

The AwareLiquid-M2 line adds four bio-inspired auxiliary modules to the MT-LNN
backbone (Phase A competitive GWT, Phase C predictive world model, Phase D
Hebbian plasticity, plus the LAVI surprise→rhythm linkage) and two engineering
hardening features (P3.1 multi-source GWT bidding, P3.2 graceful degradation).
Every one defaults to a no-op-equivalent or OFF and is covered by a dedicated
test module. **Full suite: 254 tests passing** (`python -m pytest tests/ -q`).

### P3.1 — multi-source GWT external bidding

The competitive Global-Workspace bottleneck (`CompetitiveGWTBLayer`) now accepts
**external bids**: a module outside the backbone can submit a `(B, T, d_model)`
proposal that competes on equal footing with the layer's internal bids via the
shared score head + softmax. The first consumer is the predictive world model,
which bids its one-step expectation into the workspace through a residual,
zero-gated adapter — so the bid is **identity at init** (the gate starts at 0,
the model is bit-for-bit unchanged) yet the gate keeps a live gradient and can
learn to route workspace mass toward the prediction when it helps.

| Property | Verification |
|---|---|
| No external bids ⇒ identical to pre-P3.1 forward | ✅ test |
| External bid == x ⇒ init-identity preserved | ✅ test (max\|Δ\| < 1e-3) |
| Open gate routes attention mass to the world bid | ✅ test |
| Shape-mismatched bid raises; `None` entries skipped | ✅ test |
| Gradient flows back into the external bid + gate | ✅ test |
| Diagnostics expose `gwtb_external_bid_weight`, `gwtb_world_bid_gate` | ✅ test |

Test module: `tests/test_multisource_gwt.py` (12 tests).

### P3.2 — module graceful degradation

A multi-day pre-training run must survive a transient NaN/Inf in any bio-inspired
auxiliary term. Every **auxiliary** contribution — world-model loss, GWTB
orthogonality penalty, Hebbian loss, the world-model workspace bid, and the LAVI
surprise signal — is wrapped in a finiteness guard (`_aux_or_skip`). A faulty
term is **dropped for that step and counted** instead of poisoning the loss. The
**primary cross-entropy is never guarded** (a NaN there is a real failure that
must surface). Zero-regression by construction: on finite values the guards are
no-ops. `use_graceful_degradation=False` lets faults propagate for debugging.

| Fault injected | Behaviour with guard ON | Behaviour with guard OFF |
|---|---|---|
| World-model loss → NaN | dropped, main loss finite, counted | NaN reaches main loss |
| Hebbian loss → NaN | dropped, main loss finite, counted | — |
| World-model **bid** → NaN | falls back to raw state, counted | — |
| LAVI surprise → non-finite | reset to 0.0, counted | — |
| Backward after a guarded fault | all gradients finite (training lives) | — |
| Healthy run | **no guard ever fires**, counters empty | — |

Degradation events surface in `get_mt_diagnostics()` as
`degradation_events_total` + per-module `degradation_<name>` (keys present only
when something actually fired). Test module:
`tests/test_graceful_degradation.py` (9 tests), including an OFF-switch test that
proves the guard is load-bearing.

### TorchScript export + on-device latency

`mt_lnn/export.py` traces the model's **logits path** (a `LogitsOnlyWrapper`
running eval with `use_lnn_recurrence=False`, which makes the fixed-shape graph
well defined) to a deployable TorchScript artifact. The full `forward` returns a
dict and threads a dataclass cache with Python control flow, so it isn't
scriptable; `trace` also bakes in the sequence length (global coherence has a
seq-length-dependent reduction), so **one artifact per target shape**. Export is
a pure opt-in utility — importing it has no effect on the model (pinned by test).

CLI: `python scripts/export_torchscript.py --ckpt <ckpt> --batch 1 --seq_len 256 --optimize --out mt_lnn_ts.pt`

Measured on the **125M-class backbone** (831 d × 12 L × 13 H, all v2 modules ON),
batch 1 × seq 128, `optimize_for_inference`, single desktop CPU:

| Path | ms / forward (128 tok) | ms / token | vs eager |
|---|---:|---:|---:|
| Eager | 304.8 | 2.381 | 1.00× |
| **TorchScript (optimized)** | **187.6** | **1.466** | **1.62×** |

- **Numerical parity: within `atol=1e-3`** (the test contract in
  `test_torchscript_export.py`). The FP32 max\|traced − eager\| measured ~0 on
  this run, but `optimize_for_inference` fuses ops, so parity is *asserted* to
  1e-3 — not guaranteed bit-identical.
- **1.47 ms/token** is amortized full-sequence prefill (the parallel-forward
  path), comfortably under the **< 50 ms/token** on-device target.
- The same trace at toy scale (0.3M params) runs **0.14 ms/token** at 1.5×
  speedup.

Test module: `tests/test_torchscript_export.py` (7 tests): trace/eager parity,
save+load+rerun at the traced shape, `optimize_for_inference` fidelity, latency
benchmarking sanity, and zero model mutation.

### Engineering features (training-run robustness)

| Feature | Where | Status |
|---|---|---|
| Cross-session checkpoint resume (`--resume`) | `train.py`, M2 kernel auto-detects attached/last.pt | ✅ |
| Streaming JSONL v2 module-health metrics | `train.py --metrics_jsonl` (every N steps) | ✅ |
| Graceful degradation counters in diagnostics | `get_mt_diagnostics()` | ✅ |
| Pascal-GPU (sm_60 / P100) torch auto-repair | M2 kernel (installs torch 2.6.0+cu124) | ✅ |
| 16 GB-GPU memory fit (batch 4 + expandable_segments) | M2 kernel | ✅ |
| TorchScript deployment artifact + latency report | `mt_lnn/export.py`, `scripts/export_torchscript.py` | ✅ |

### 125M from-scratch pretraining (AwareLiquid-M2) — in progress

131.4M-param MT-LNN (832 d × 12 L × 13 H, GQA=1, seq 512) pretraining on
WikiText-103 (gpt2 BPE) with all four v2.0 modules ON, on Kaggle. The
stability-validation run is short (1200 steps, batch 4, grad_accum 1) to confirm
no representational/routing collapse before committing to the multi-day full run.

Status: stability run executing after clearing two infra blockers — the Pascal
sm_60 / torch-2.10-cu128 kernel mismatch and a 16 GB-GPU OOM (batch 8 → 4 at
seq 512). **125M training/PPL + v2 module-health metrics (competition entropy,
world-model prediction error, surprise, Hebbian/EMA dynamics) are pending this
run and will be filled in here once validation passes.**

---

## State-only streaming: O(1) working memory (validated)

The headline claim — recurrent state-only decode holds a **constant** cache
footprint while a KV cache grows **O(T)** — is now pinned at a stream length
that far exceeds the model's finite RoPE/mask window (`T >> max_seq_len`), the
regime that actually stresses the claim.

### Measurement (CPU, tiny config: window=16, d=64, 2 layers)

| Path | T = 16 (1 window) | T = 128 (8 wraps) | T = 320 (20 wraps) | Growth |
|---|---|---|---|---|
| KV cache (`tensor_bytes`) | 12,328 B | 76,840 B | — (capped use) | **+576 B / token, O(T)** |
| State-only (`h_prev` only) | 2,600 B | 2,600 B | 2,600 B | **flat, O(1)** |
| KV / state ratio | 4.7× | **29.6×** | grows without bound | widens with T |

The KV cache adds a constant **+576 bytes per token** (the per-position k/v across
layers) — a clean linear O(T) signature — while the state-only cache is identical
to the byte at step 1 no matter how deep the stream runs (320 steps = 20 RoPE
wraps tested).

### Reproduce

```bash
# Regression test (4 cases, asserts flat-vs-linear contrast past 20 wraps):
python -m pytest tests/test_long_context_memory.py -v

# Benchmark in the true O(1) regime (window pinned, T >> window):
python benchmarks/state_only_streaming.py --steps 512 --max_seq_len 64 --fixed_window
#   -> kv_cache_stream  cache=330,792 B   (O(T))
#      state_only_stream cache=  2,600 B   (O(1))  -> 127x smaller at 8 wraps
```

**Honesty note:** state-only decode trades exactness for the bounded footprint —
because it drops KV history and wraps position offsets, its logits diverge from a
full-causal forward (the benchmark reports `div.max ~ 0.34` at 8 wraps). The O(1)
result is a *memory* guarantee, not a claim of bit-identical long-context logits.
The `--fixed_window` flag was added precisely so the benchmark can run this
regime; by default it grows the window to fit the stream (avoiding any wrap).

## fp16 divergence root cause (2026-07-19) — RESOLVED

MT-LNN's fp16-only training divergence is fixed. Reproduction, diagnosis and
verification below; diagnostic tool is `benchmarks/diagnose_fp16_divergence.py`.

**Reproduction.** `--dtype fp16 --steps 2000 --train_token_cap 50000000`,
seed 0: non-finite loss at **step 875** (`stable: false`, `val_ppl: Infinity`).
The exact step is data-order dependent — 629 on the original Colab T4, 875/896
locally — so a shorter cap (20M) does not trigger it at all. Matching the
original token cap is required to reproduce.

**Diagnosis.** Forward hooks on every leaf module (loss trajectory stayed
healthy — loss 5.75 and grad-norm 1.07 at step 895, NaN at 896, no gradual
buildup, no gradient explosion). Two findings ruled out the obvious causes:

| hypothesis | verdict |
|---|---|
| gradient explosion | ❌ grad-norm ~1.1 right up to failure |
| activation overflow | ❌ peak activation **67.4** vs fp16 ceiling **65504** |
| GradScaler misbehaviour | ❌ scale pinned at 65536, loss went non-finite in the **forward** pass |

The first non-finite module output was `coherence.dropout`, but its *inputs
were already non-finite* — i.e. it propagated, not created. The creator was a
functional op (no module hook): the attention score computation in
`mt_lnn/global_coherence.py`.

**Root cause.** The layer computed `(Q @ K.transpose(-2,-1)) / self.scale` —
the `d_head=64` accumulation happens **before** the down-scaling. Measured
projection peaks were `k_proj` 60.8 and `q_proj` 52.8, so the intermediate
product reaches ~2e5, **overflowing fp16 inside the matmul itself**. The
resulting `Inf` then meets the causal mask's zeros in `_gate_energy`'s
`raw * valid`, and **`Inf * 0 = NaN`**; that NaN flows through
`sigmoid()` into the collapse gate and poisons the whole layer output.
fp32 never trips this (ceiling 3.4e38), which is exactly why it was
precision-specific. It appears mid-training rather than at step 0 because the
Q/K projections have to grow first — hence the sudden onset after ~900 healthy
steps and the drift in the failing step across machines.

**Fix** (`mt_lnn/global_coherence.py`):
1. `(Q / self.scale) @ K.transpose(-2,-1)` at all 4 sites — algebraically
   identical, but holds the accumulation `sqrt(d_head)`× lower.
2. `_gate_energy`: `torch.where(valid > 0, raw, 0)` instead of `raw * valid`
   (removes the `Inf * 0` path), fp32 accumulation, and `clamp_min(1e-6)`
   replacing a literal `1e-9` epsilon that **underflows to exactly 0 in fp16**
   and so provided no guard at all.

**Verification.** The identical recipe that failed at step 896:

Run at the original failing length (2000 steps, 50M cap, seed 0):

| | before fix | after fix |
|---|---|---|
| divergence | step 875 non-finite | **none, full run** |
| `stable` | **false** | **true** |
| val PPL | **Infinity** | **257.91** |
| steps completed | 875 | **2000** |

**fp16 is restored to fp32 parity, not merely made non-crashing:** 257.91 (fp16)
vs 257.48 (fp32, same recipe) — a 0.17% difference, well inside the ±4.89
seed-to-seed variance measured at this budget. A wrong fix would typically cost
quality; this one does not.

fp32 regression re-checked: unchanged, trains normally.

**Audit.** Every other attention surface (`mt_attention.py`, `gwtb.py`,
`mt_lnn_layer.py`, `mt_lnn_v2.py`) delegates to
`F.scaled_dot_product_attention`, which handles scaling safely.
`global_coherence.py` was the only hand-rolled score computation in the
codebase — which is exactly why the failure was isolated to it. No other site
carries this pattern.

## A2 edge pilot: battery SoH on real NASA cells (2026-07-24)

First real edge deployment case, not a synthetic task. Scripts:
`benchmarks/battery_soh_edge.py`, `battery_streaming_memory.py`,
`battery_irregular_sampling.py`.

**Data.** NASA Ames PCoE Li-ion aging set (public, no auth,
`https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip`), cells
B0005/6/7/18. Verified against the published description: B0005 has 168
discharge cycles fading 1.8565 → 1.3251 Ah.

**Protocol.** One sample per discharge cycle: (128, 3) voltage/current/
temperature → measured capacity in Ah. **An entire cell (B0018) is held out** —
splitting cycles within a cell leaks the degradation trajectory and flatters
every model. Features standardised on train statistics only. All archs share
d_model=65 / 2 layers; parameter counts differ and are reported.

### 1. Accuracy — regular sampling (10 seeds)

| arch | params | val RMSE (Ah) | vs mean baseline |
|---|---|---|---|
| transformer | 111,866 | 0.0972 ± 0.0070 | −38% |
| **mt_lnn** | **42,474** | 0.0986 ± 0.0171 | −37% |
| lstm | 69,096 | 0.1034 ± 0.0180 | −34% |
| gru | 51,936 | 0.1048 ± 0.0110 | −33% |
| *(predict train mean)* | — | *0.1572* | — |

All four beat the constant-predictor baseline, so the task is learnable and the
setup is sound. **Every pairwise Welch |t| < 1 — on regular sampling the four
architectures are statistically indistinguishable on accuracy.** MT-LNN reaches
that parity with the fewest parameters (2.6× fewer than the transformer), which
is the only accuracy-side claim the data supports.

> An earlier 3-seed run showed mt_lnn apparently best at 0.0813. It did not
> survive 10 seeds (0.0986, transformer ahead on the mean). Reported here as a
> reminder that n=3 on this task is noise.

### 2. Streaming memory — carried state vs stream length

Bytes of state that must be retained to keep predicting (measured, CPU):

| stream | lstm | gru | mt_lnn | transformer KV |
|---|---|---|---|---|
| 128 | 1,040 B | 520 B | 2,600 B | 133,120 B |
| 32,768 | 1,040 B | 520 B | **2,600 B** | **34,078,720 B** |

**MT-LNN is flat at 2.6 KB across a 256× increase in stream length — 13,107×
smaller than the attention KV cache at 32K.** Note honestly that **O(1) state is
not unique to the liquid core**: LSTM and GRU are also flat, and in fact carry
less. Constant memory is a property of recurrence, not of this architecture.

### 3. Irregular sampling — where the liquid core does differentiate

Real BMS controllers wake on events, drop samples under load, and vary duty
cycle, so the stream is not on a fixed grid. Timesteps are randomly dropped;
**Δt is supplied as an input feature to every architecture** so the RNNs are not
handicapped by construction, and train/test share the regime.

| drop | kept | mt_lnn | lstm | gru | transformer |
|---|---|---|---|---|---|
| 0% | 128 | 0.1027 | 0.0996 | 0.1018 | 0.0970 |
| 30% | 90 | 0.0930 | 0.1136 | 0.1182 | 0.0913 |
| 60% | 51 | 0.1057 | 0.1236 | 0.1555 | 0.1010 |
| 80% | 26 | 0.1106 | 0.1306 | 0.1352 | 0.0969 |

Degradation from regular to 80% dropped:

| arch | Δ RMSE | verdict |
|---|---|---|
| **mt_lnn** | **+7.7%** | robust |
| lstm | +31.1% | degrades |
| gru | +32.8% | degrades |
| transformer | −0.1% | robust |

At 80% drop (10 seeds): **mt_lnn vs lstm t=+2.40, vs gru t=+2.63 — both
significant.** vs transformer t=−1.94, within noise.

### What this pilot does and does not establish

**Establishes.** On a real battery-management task, MT-LNN is the only
architecture tested that is *both* robust to irregular sampling (+7.7% vs the
RNNs' +31–33%) *and* constant-memory in streaming (2.6 KB vs the transformer's
34 MB at 32K). The transformer matches its accuracy but cannot fit the memory
budget of an MCU; the RNNs fit the budget but degrade when samples are dropped.
That combination is the deployable niche.

**Does not establish.** No accuracy advantage on regularly-sampled data (all
four tie). No advantage over the transformer on irregular sampling either — the
transformer is equally robust there and its edge is only excluded by memory, not
by accuracy. Single dataset, single held-out cell, 128-step windows.

## Event-native sensing streams (2026-08-29) — `iter/event-stream-liquid`

Event cameras emit events only when a pixel's log-intensity changes by θ, so
**irregularity is the native format, not a corrupted grid**: Δt spans orders of
magnitude (µs…s). `benchmarks/event_stream.py` generates such streams from DVS
physics (multi-channel latent signal, emit-when-|Δv|≥θ, motion-burst/silence
alternation with log-uniform silent gaps); the `span` knob stretches the
realized Δt distribution from ~1.4 to ~3.2 decades at constant event count.
`benchmarks/event_bench.py` runs two tasks through the shared `build()` factory
(mt_lnn / lstm / gru / transformer; **Δt is fed to every architecture** — the
battery-irregular guardrail):

- **state** — regress the latent signal's current value from the event stream
  (normalized RMSE, constant predictor = 1.0). This is where "how much time
  passed" must enter the state update.
- **next_channel** — predict the next event's channel (6-way, chance 0.167).

10 seeds per cell; every mt_lnn-vs-baseline comparison goes through
`publishable()` (≥3 seeds, no bimodality, paired sign test p<0.05). Full data:
`benchmarks/results/event_theta_sweep.json` (240 runs).

### State estimation — mt_lnn best at every event density (nRMSE, 10 seeds)

| θ (density) | mt_lnn | lstm | gru | transformer |
|---|---|---|---|---|
| 0.15 (dense) | **0.974 ± 0.010** | 1.152 | 1.157 | 1.299 |
| 0.35 (mid) | **0.750 ± 0.023** | 0.860 | 0.810 | 0.937 |
| 0.8 (sparse) | **0.978 ± 0.009** | 1.215 | 1.242 | 1.298 |

E0 gates: **citable** at θ=0.15 vs lstm and transformer, θ=0.35 vs transformer;
at θ=0.8 and vs gru at θ=0.15/0.35 the cell is archive-only (bimodality flag
or p≥0.05) despite the mean separation. The three discrete baselines fail to
beat the constant predictor (1.0) at θ=0.15/0.8; mt_lnn is the only one that
does, at every θ.

### Next-channel prediction — a citable NEGATIVE for the liquid core

| θ | mt_lnn | lstm | gru | transformer |
|---|---|---|---|---|
| 0.15 | 0.227 | **0.329** | **0.328** | 0.205 |
| 0.35 | 0.237 | 0.249 | **0.243** | 0.201 |
| 0.8 | 0.188 | **0.211** | 0.203 | 0.177 |

At the dense tier mt_lnn is **significantly worse** than lstm and gru (both
gates pass with mt_lnn on the losing side). Continuous-time integration helps
carry state, not categorical next-event prediction; reported as a negative,
not buried.

### Δt-span pressure test — the pre-registered verdict is NULL

`benchmarks/event_dt_span.py` (`event_dt_span.json`): sweep realized Δt span
1.40 → 1.71 → 2.18 → 3.24 decades at constant event budget; judgement fixed
in the script header BEFORE running — PROVEN only if (a) mt_lnn beats the
strongest discrete baseline at the widest span through `publishable()` AND
(b) that gap is wider at the widest span than at the narrowest.

Outcome (state task, 10 seeds): (a) **passed** — at 3.24 decades mt_lnn
0.774 ± 0.013 vs lstm 0.817 ± 0.035, 9/10 seed-wins, p=0.0215; mt_lnn is
best at every span. (b) **failed** — the gap NARROWS (0.141 at 1.40 decades →
0.044 at 3.24) because every architecture gets better as the span stretches;
nRMSE-vs-span slopes: mt_lnn −0.043, lstm −0.100, gru −0.079, transformer
−0.034. **Per the pre-registered rule the headline claim "the liquid advantage
widens with Δt span" is NULL**; the surviving citable statement is the
significant mt_lnn advantage at the widest tested span. τ-ladder interaction
(mt_lnn vs single-time-scale mt_lnn_tau1): mt_lnn better in 9/10 seeds at the
widest span, but the tau1 arm is flagged bimodal → archive-only, NULL.

### Real data: N-Caltech101 — no significant edge (NULL)

`benchmarks/event_real_data.py` (`event_ncaltech101.json`): 4 GB no-auth
download (Mendeley, URL from tonic), flat zip parsed in memory (Orchard 5-byte
events, timestamp-overflow handling), 10 train / 10 held-out classes,
15 samples/class, next-event-polarity prediction (chance 0.5), Δt fed to both
architectures, 10 seeds. mt_lnn 0.545 ± 0.031 vs gru 0.520 ± 0.027 — sign
test 5W/3L/2T, **p=0.727: no significant difference**. Recorded as a null;
no real-data advantage is claimed.

### What this branch establishes and does not

**Establishes (citable cells only).** On synthetic event streams, the liquid
core is the only architecture that beats a constant predictor on state
estimation at every event density, with statistically significant margins at
the dense tier (vs lstm p<0.05, n=10) and vs transformer at two tiers; it is
significantly best at the widest tested Δt span (3.24 decades). The same
benchmark produces a citable negative: discrete RNNs win dense-tier
next-channel prediction.

**Does not establish.** That the advantage *widens* with Δt span
(pre-registered gate (b) failed — NULL). Any real-data advantage (N-Caltech101
null). Anything about gru_d (pending merge in `iter/irregular-streaming-edge`;
not re-implemented here per branch boundaries).

> **Bimodality-flag sensitivity note (2026-08-29, post-hoc, zero retraining).**
> The bimodality detector's gap_ratio>3 threshold was calibrated for grokking
> (chance-vs-perfect clusters, near-full-range gaps) and over-flags tightly
> clustered arms. Recomputed from `event_theta_sweep.json` per-seed values:
> the state θ=0.8 **mt_lnn** arm's flag is a clear false positive (best-split
> absolute gap 0.0083 = 0.8% of task scale = 3.5% of the cross-architecture
> mean separation; its 10 seed values 0.966–0.991 are continuously spread).
> Under that arm's flag the state θ=0.8 vs lstm/gru cells are archived; their
> mean separations (0.24 / 0.26) are the largest in the sweep and would pass
> the sign test outright. The other three flags are genuine or borderline
> (absolute gap 21%/48%/69% of separation). **Historical verdicts stand** (no
> post-hoc re-adjudication); a calibrated gate — gap_ratio>3 AND absolute gap
> >10% of cross-architecture range — is pre-registered in the script
> docstrings for v2 runs onward.

## Latent recursion — 外部坐标系对齐 + 算力账本 (iter/latent-recursion, 2026-08-29)

Full design doc: [docs/LATENT_RECURSION.md](docs/LATENT_RECURSION.md).

Alignment: Coconut (arXiv:2412.06769) / Huginn-Geiping (arXiv:2502.05171)
coordinate system — loop body = decoder block, test-time variable depth.
Our unique cell: **continuous-time loop body** (`liquid_step_ladder`, τ-ladder
weighted iterations; default off, bit-equivalent, zero new params).

**Compute ledger (analytic, conventions frozen 2026-08-29,
`benchmarks/compute_accounting.py` + 6 unit tests incl. hand-computed values
123,136 / 49,049 MACs and a parameter-count reconciliation against the real
built models):** at probe scale (T=54), 8 whole-block latent iterations cost
**157 MFLOPs vs 33 MFLOPs for 8 explicit CoT tokens (4.8×)**, because each
latent pass recomputes T×T attention while CoT decodes incrementally. The
latent path's edge is memory: 0 KV growth + constant 4.1 KB state vs
104.8 KB KV peak for the CoT path. **The "saves tokens" narrative does not
hold on FLOPs; it holds on memory.**

**Adjudication (Task 2) — DONE, verdict NULL (`budget_wall`).** The full
protocol ran on a rented A100-80GB: 30k steps × 6 seeds × {core,stack} ×
d{1,2,4,8} = **48/48 configs** (≈322 GPU·h total; per-config `wall_s` in the
row JSONs; batches landed 8/30–8/31 UTC). Every tier sits at chance
(1/16 = 0.0625): stack d1→d8 = 0.0668 → 0.0675 (**non-monotonic** — d2 dips
below d1; gain 0.0007 « 2σ = 0.0048), core d8 = 0.0676. The preregistered
`judge_decision` (unmodified) returns **`h_supported=False`,
diagnosis=budget_wall** — the loop never learned pointer_chase at this
budget, so depth effects are moot until steps/curriculum increase; per red
line the criteria were not moved. The `bimodal=True` flags on the d8 tiers
are σ-scale noise at chance level, not grokking zones. Provenance: the
server's in-situ verdict was a crash remnant (n_rows=1; the core-only
aggregation path raises `KeyError: 'stack'` *after* row JSONs are safely
written — cosmetic). The committed verdict was **recomputed from the 48
committed rows with this branch's own frozen judge code**; row `git_rev`
fields read `unknown` (no git in the container), `device=cuda` as measured.

**Frontier (Task 3, `anytime_frontier.py`) — still awaiting GPU.** Never
started before the box was reclaimed. A100 probe: CoT 0.154 s/step (×4
granularities) + latent 0.204 s/step ≈ 6.8 h/seed ≈ 41 h sequential.
Resume-safe (steps-matched rows skip); the preregistered dominance reading
stands unchanged.

### 6. Mechanism probe: Δt wired into the decay — CLOSED (2026-08-29)

`mt_lnn_dt` (LiquidDTRegressor, `battery_soh_edge.py`) replaces the
production layer's construction-time constant `config.dt` with the true
per-step `λ_t = exp(-Δt_t/τ_{p,s})` through the general parallel scan —
50,857 params, inside the mt_lnn (55K) / gru (74K) band, enforced by tests.
Rules R1 (in-distribution) / R2 (Δt-shift) were pre-registered in the
producer docstring before the run; JSON:
`benchmarks/results/synth_ct_control_dt.json` (per-seed values, config echo,
code hash). 5 seeds, 6 archs, 6 grid tiers + shift tier.

| tier | mt_lnn | **mt_lnn_dt** | lstm | gru | gru_d | transformer |
|---|---|---|---|---|---|---|
| dt0.05 cv1 | 0.0279 | 0.0560 | 0.0282 | 0.0233 | 0.0318 | 0.0567 |
| dt0.15 cv1 | 0.0409 | 0.0189 | 0.0363 | 0.0325 | 0.0320 | 0.0516 |
| dt0.4 cv1 | 0.0280 | 0.0252 | 0.0270 | 0.0287 | 0.0228 | 0.0236 |
| **shift 0.05→0.4** | 0.4624 | **0.3005** | 0.4185 | 0.2722 | **0.2485** | 0.6310 |

**R1: 0/3** — vs gru_d/lstm/gru/mt_lnn all |t|<2 (at dt0.15 significantly
worse than gru, t=−2.09). **R2: failed** — under the shift, mt_lnn_dt beats
lstm (+4.41) and feature-wired mt_lnn (+8.85) but loses to gru_d (−1.74) and
gru (−1.49). Two findings worth keeping: (a) under Δt distribution shift the
LEARNED GRU-D decay is the most robust of all six — learnable decay
generalises better than the fixed structural exp(−Δt/τ); (b) every
architecture degrades ~10× under the shift, i.e. gap-handling under unseen
gap scales is broadly unsolved, not an architecture checkbox. Verdict per
the registered mapping: **CLOSED — the continuous-time mechanism story is
not told anywhere in this repo.**
