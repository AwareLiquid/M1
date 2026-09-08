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

## O-Series-hybrid — ARR + 回填注意力（研究中，尚无数字）

**What it is.** The same recipe, but only a *fraction* of the layers are
converted; the rest keep their pretrained attention, bit-identical. The knob
is `--keep_ratio` on `benchmarks/distill_arr.py`, driven over
{0, 1:8, 1:4, 1:2} by `benchmarks/arr_ratio_sweep.py`.

**Why it exists.** The frontier (Qwen3-Next, Kimi Linear, GLM-5.3-Flash) has
converged on keeping ~1/4 of the layers as full attention. Those ratios were
chosen at 10¹⁰–10¹¹ parameter scale; whether 1:4 is also the拐点 at our
distillation budget is an open question this repo has to answer with its own
numbers.

**Status: confirmed 2026-09-06** (4 比例 × 6 seeds, 21/24 臂有效,
`--warmup_b 200` 稳定化配方, A100-80GB bf16) —
see [docs/ARR_RATIO_PARETO.md](ARR_RATIO_PARETO.md) + the ledger in
`benchmarks/results/rebuilt/arr_ratio_pareto.{json,md}`.

| claim | status |
|---|---|
| 回填注意力改善 PPL | ✅ **1:4 PASS** — mean 67.37 vs ratio-0 142.89 = **+52.9%**（门限 ≥15%） |
| 哪个比例落在质量/内存帕累托前沿上 | ✅ **1:4**（6/22 注意力层；192.0 MB @32K fp16 vs 0 比例 0.65 MB O(1)）；1:8 +37.8%；1:2 判负（3/6 臂发散） |
| 单位携带内存下优于任一父本 | ✅ 曲线实测：O(1) 端（0.65 MB）与质量端（1:4，接近教师）都是曲线上真实端点 |
| **O(1) / 恒定内存** | ✘ **do not claim** — 见下方边界 |

**双峰性警示（写进引用处）**：1:4 好模式 18.9–20.5 / 坏模式 162.1–164.0；
对照同样双峰 74.5–207.6。配对口径 6/6 全胜；均值口径方差大，引用附 spread。
后续（G1 盲区）：1:4 上补跑 cross-window recall 作补充列（需重训留存 checkpoint）。

**Honest boundary (write it into every artifact).** A hybrid is **not O(1)**:
carried memory = constant recurrent state **+ O(T) KV of the surviving
attention layers**. The two are reported in separate columns and never
merged; only ratio 0 (the pure O-series above) has an empty KV column. KV is
billed by the same `kv_bytes` formula `benchmarks/kv_frontier.py` uses, at
both fp16 and 2-bit — our hybrid does not get a friendlier accountant than
the competition.

**Positioning (confirmed 2026-09-06).** 1:4 cleared the gate (+52.9% vs the
ratio-0 control, 6/6 paired seeds): this is the line for "unbounded stream,
but quality matters more than the absolute memory floor" — O(1) and quality
are now endpoints of a measured curve rather than a binary choice.

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
