# Kaggle Run — Phase 5 Backbone (TinyLlama-1.1B + MT adapter + LoRA)

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


Run date: **2026-05-28**
Hardware: **Kaggle free GPU — Tesla T4 (14.6 GB), single GPU**
Wall-clock: ~3 h (train) + ~5 min (PPL eval) + ~5 min (needle eval)
Reproduce: `docs/guides/KAGGLE_RUN.md`

## What's in this directory

| File | Content |
|---|---|
| `train.log` | Full stdout of `train_llama_mt_adapter.py` (1000 steps, loss curve, dataset loading) |
| `ppl_ablation.json` | Base vs adapter PPL on WikiText-2 valid, 38 400 tokens |
| `ppl_ablation.log` | Full stdout of `bench_llama_mt_ablation.py` |
| `needle.json` | NIAH-Single1 results across ctx ∈ {1024, 2048, 4096} × depth ∈ {0.1, 0.5, 0.9} |
| `needle.log` | Full stdout of `bench_llama_mt_needle.py` |

The adapter checkpoint `llama_mt_adapter_001000.pt` (~150 MB) is **not** in git (`.gitignore`'d).
Download it separately from the Kaggle Output panel and place it at:

    checkpoints/llama_mt_adapter/llama_mt_adapter_001000.pt

Then run `demo_llama_mt_adapter.py --adapter checkpoints/llama_mt_adapter/llama_mt_adapter_001000.pt` to use it locally.

## Headline numbers

| Metric | Value |
|---|---|
| PPL (base TinyLlama-1.1B) | **9.161** |
| PPL (+ MT adapter + LoRA, 1000 steps) | **6.553** |
| **PPL reduction** | **−28.5 %** |
| Trainable params | 2.30 M / 1.17 B (**0.196 %**) |
| Training loss range | 2.5 → ~1.9 (smooth descent, no NaN) |
| Decode-time penalty | +10 % vs base |
| Needle exact-match (all 9 cells, both variants) | 0.000 — base-bottlenecked |

See `BENCHMARKS.md` §"1.1B Scale: TinyLlama-1.1B" and `mt_lnn_arxiv.md` §6.6 for the full analysis.
