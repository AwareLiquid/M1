---
id: H001
status: UNDER_TEST
thread: T001
created: 2026-08-28
refs: [benchmarks/latent_recursion.py, benchmarks/results/reasoning_depth.jsonl, _rescue_20260831/latent_recursion/latent_recursion_verdict.json]
---

# H001 — 整块循环(stack)深度使深度-准确率单调上升,核心循环(core)不升

## Statement

`stack_iterations`(循环体 = 注意力+LNN 整块)随深度增加,pointer-chase 类任务
准确率严格单调上升;`core_iterations`(循环体 = LNN 子层)不上升。
坐标系:Coconut(arXiv:2412.06769)/ Huginn-Geiping(arXiv:2502.05171)的
潜空间递归路线,差异格 = 连续时间步长(τ 阶梯)。

## Pre-registered judgment(写死于 `judge_decision`,PR #9 分支;不事后移动)

H 成立 ⇔ 同时满足:
1. mean(stack) 随深度(d1→d8)严格单调上升;
2. mean(stack,d8) − mean(stack,d1) ≥ 2·σ_d8(σ ddof=1);
3. E0 门槛:publishable(≥3 seeds + 非双峰 + 配对符号检验 p<0.05);
   可引用判决默认 6 seeds(3 seeds 配对检验最小 p=0.25,数学上不可引用)。
判负时四选一数据诊断: budget_wall / bimodal_grokking_zone /
flat_iterations_ignored / direction_but_underpowered。

## Evidence basis(立项依据,只引不测)

- pointer_chase d2 @ 30K 步 grok 到 acc 1.000(`reasoning_depth.jsonl`,verdict-30k,seed 0);
  d2 @ 12K 仅 0.249 —— grokking 相变存在,深度是足够步数下的未开发维度。
- 算力账本(Task 1,`compute_accounting.json`):stack N=8 = 157 MFLOPs vs
  8 CoT token = 33 MFLOPs(4.8×),潜空间优势主张在内存(0 KV + 4.1KB 恒定状态),
  不在 FLOPs。

## Verdict log(追加式)

- 2026-08-31: Phase B Task 2 探针(1 seed × 30K,difficulty=8)48/48 cell
  全部 mean_acc ∈ [0.062, 0.072] < 0.25 → 诊断 **budget_wall**,
  h_supported=false。E0 禁引用单 seed,非最终判决。
  ⚠️ learnability 缺陷立项后才发现:d4@20K 仅 0.373,d8 远在可学习域外,
  判决任务选择依据见 ADJ-001。verdict 文件 n_rows=1 为事故残留(KeyError
  已修,304ed35),完整 48 行重跑判决见 ADJ-002。
