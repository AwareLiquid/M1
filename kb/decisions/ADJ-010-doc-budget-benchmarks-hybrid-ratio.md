---
id: ADJ-010
date: 2026-09-08
refs: [scripts/check_doc_budget.py, docs/ARR_RATIO_PARETO.md, kb/hypotheses/H002-arr-backfill-pareto.md]
---

# ADJ-010 — BENCHMARKS.md 文档尺寸预算上调 (1850 → 1950)

**决策**：`scripts/check_doc_budget.py` 的 `BUDGETS["BENCHMARKS.md"]` 从
1850 行上调至 1950 行。

**理由**：H002 判定（SUPPORTED）落地后，BENCHMARKS 需承载
"O-series hybrid-ratio sweep" 确认节（1:4 PASS +52.9%、双峰性警示、
内存两列分列、6-seed 确认跑说明）——这是判定轨行（RESULTS.md L65）引用的
可验证锚点章节，随 ARR 生产者工具入 main（PR #47）一并落地，属真实内容
增长而非流水账膨胀。插入后实测 1892 行，超出旧预算 42 行；新预算 1950
留有 ~3% 余量。

**替代方案（被否）**：
1. 不引章节、改引用证据文件——判定行已有文件锚点，但 H002 判据表
   （门限 ≥15%、四比例数值）在 BENCHMARKS 是 paper-facing 的必要落点，
   归档到 docs/ 会割裂"RESULTS 每行 → BENCHMARKS 章节"的账本契约；
2. 瘦身 BENCHMARKS 既有章节——edge 批次2 缺失章节（§1–§5/§7/§8）回填
   是独立待办（check_claims WARN 清单），预计还会净增，先瘦后涨非
   合理次序；
3. 拒绝本 PR——复现路径断裂（生产者工具孤悬）比预算红线更严重。

**约束**：本预算仅覆盖已确认章节；后续 edge 缺失章节回填若再触顶，
须另行 ADJ 决策并附当时行数实测，不得在同 PR 内静默改预算。
