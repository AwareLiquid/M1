---
id: ADJ-015
date: 2026-10-25
refs: [README.md, scripts/check_doc_budget.py, benchmarks/baselines.py, benchmarks/scaling_comparison.py, benchmarks/selective_copy.py]
---

# ADJ-015 — README 尺寸预算上调 (650 → 715)

**决策**：`scripts/check_doc_budget.py` 的 `BUDGETS["README.md"]` 由
650 行上调至 715 行。

**根因证据**：AwareLiquid/M1 镜像同步 commit `a9a1045a` 把 README 折叠成
单行乱码，且内容停在 modern-trunk 落地前（MT-LNN 88.93 vs modern 78.86，
落后 11.3%）。恢复完整 README 并新增 "Benchmark provenance" 章节后实测
**695 行**（`python scripts/check_doc_budget.py -v`），超旧预算 45 行；
PR #2 的 audit-results job 实测报 `FAIL README.md: 695 / 650`。

**理由**：provenance 章节逐条给出 `benchmarks/baselines.py` 中五个对比模型
的精确架构定义（SimpleCausalTransformer / ModernCausalTransformer /
SimpleCausalLNN / Mamba-130m / MT-LNN）、数据集与 tokenizer
（WikiText-103-raw-v1 + GPT-2）、每个 benchmark 的 metric 原始定义
（纯 lm_loss PPL、Selective Copy 三指标、跨窗口 recall、O(1) 账本）、
参数匹配口径（125M 为 matched width/depth，Selective Copy 为 matched
params）、E0 引用协议与复现命令——这是第三方复现与外部尽调的直接入口，
属真实内容增长而非流水账。新预算 715 行留 ~3% 余量。

**替代方案（被否）**：
1. 拆到 `docs/BENCHMARK_PROVENANCE.md`——需求明确要求 benchmark 细节落在
   README（单一入口），拆分割裂"结果表 → 怎么测的"链路；
2. 瘦身既有章节（如 File map 约 150 行）腾预算——删除与本次改动无关的
   参考内容，风险与审阅成本更高，File map 是模块索引锚点；
3. 只改 README 不动预算——audit-results 持续红，PR 无法合并。

**验证**：`python scripts/check_doc_budget.py -v` 应输出
`PASS  README.md: 695 / 715 行`；CI audit-results 转绿。

**DEVIATION**：none。
