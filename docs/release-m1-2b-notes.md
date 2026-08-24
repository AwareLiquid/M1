# M1-2B — Release Notes 模板（发布时填数字）

> 用法：复制到 GitHub Release body / HF README。`【】` 内为待填项。
> 风格基准：m1-128m-v1 的 notes（What it is / Serve it / Measured / honest / MIT）。

---

## M1-2B — 1.9B hybrid (milestone)

**What it is:** 1.9B-parameter hybrid — window attention + liquid core (selective
decay, exp parameterization) in every layer. 【35】 layers, d_model 【2912】.
Trained from scratch on 【data_slim 486M tokens】, 【N】 steps.

**Status:** 【milestone checkpoint / converged】. This checkpoint is 【val PPL
【X】】 at step 【N】.

**Serve it:**

```bash
CKPT_PATH=ckpt_【XXXXXX】.pt TOKENIZER=gpt2 python -m uvicorn serve.server:app
```

> 1.9B fp32 ≈ 7.6 GB weights — CPU works but is slow; a GPU instance is
> recommended for interactive use.

**Measured:**

- 【val PPL 曲线关键点：如 107 → 74.75 → 【最终值】】
- O(1) carried state: constant-size recurrent state, no growing KV-cache
- 【若有外推/长序列数据，补上；没有就删掉这行】

**Boundaries (honest):**

- 【若为里程碑发布】This is not a converged production model — text quality is
  below established open LLMs of similar size. Published for reproducibility and
  roadmap transparency.
- 【若为收敛发布】Language-modeling quality is a research result, not a
  state-of-the-art claim; PPL still trails matched-size strong baselines.

License: MIT.
