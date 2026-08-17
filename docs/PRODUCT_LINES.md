# AwareLiquid Product Lines — M-Series vs O-Series

Two model lines, one codebase, deliberately different trade-offs. This split
exists to keep claims attributable: every capability statement below is
backed by a table in [BENCHMARKS.md](../BENCHMARKS.md).

## M-Series — hybrid (attention + liquid adapter)

**What it is.** A frozen pretrained transformer (attention intact) plus the
MT-v2s liquid adapter: multi-timescale selective-decay recurrence +
fast-weight associative memory, 8.4M adapter parameters (0.76% of a 1.1B
base), streaming state across decode steps.

**Serving today.** `M1` on awareliquid.ai — TinyLlama-1.1B +
`llama_mt_adapter_v2s_003000.pt` (24 MB), bilingual SFT.

**Honest capability card.**

| claim | status | evidence |
|---|---|---|
| General QA / reasoning of its base class | ✔ unchanged | capability suite ±1.2pt vs base |
| Cross-window recall through liquid state | ✔ unique | 0.56±0.09 vs 0.000 structural for attention/LoRA |
| Better perplexity than LoRA | ✘ do not claim | attribution: MT adds ≈0 beyond LoRA |
| Long-context LM gains from state | ✘ do not claim | two null results; state = episodic K→V memory |

**Positioning.** Cloud/GPU serving where full quality matters; the liquid
adapter adds a memory capability attention cannot express, at ~1% parameter
overhead.

## O-Series — pure recurrent (ARR, no attention)

**What it is.** The same pretrained MLPs/embeddings with EVERY
self-attention block replaced by an MT recurrent mixer (`mt_lnn/arr.py`),
then distilled from the hybrid teacher. **Zero attention, zero KV cache,
O(1) inference state** regardless of context length.

**Status: research preview.** Distillation trajectory (WikiText-2 test PPL,
teacher 11.8): 13k → 264 → 32.9 → **25.4** at ~18M cumulative distillation
tokens (2.15× teacher). Converging with tokens; not yet at parity. ARR
cross-window recall: negative at current budget (curriculum retry queued).

**Honest capability card.**

| claim | status |
|---|---|
| Constant memory, no KV cache growth | ✔ by construction |
| Runs the base's knowledge without attention | ✔ at 2.15× PPL cost (and falling) |
| Matches teacher quality | ✘ not yet — token budget, not architecture, is the current limiter |
| General-benchmark competitiveness | not a goal for this line |

**Positioning.** Edge / CPU / low-power / unbounded-stream scenarios where
KV-cache growth is disqualifying. A deliberate trade: some quality for
extreme memory behaviour.

## Why not one model with an `enable_attention` switch?

Attention weights and mixer weights are different training artifacts — a
hybrid model with attention disabled is a broken model, not an O-series
model. The split is at the **checkpoint** level: one repo, two checkpoints;
`serve/server_hf.py` rebuilds the correct graph from checkpoint metadata.

## Biological-prior policy (both lines)

Biology provides initialization, engineering provides the optimum: τ
timescale ladders initialize from biological priors and are
softplus-trainable. Ablation: freezing τ at biological values scores 0.285
cross-window recall vs 0.621 trained — the prior is a good start and must
not be a cage. Modules without positive evidence stay OFF in shipped configs and are
documented as negative results. The switch-matrix has now been run
(BENCHMARKS.md §O1 module switch-matrix): **all five optional modules
(predictive coding, competitive GWTB, world model, rhythm, Hebbian) are
PPL-neutral at the 48M scale** — the full stack costs 5.6% throughput for a
noise-level change. Lean/shipped O1 configs run the core trunk only.
