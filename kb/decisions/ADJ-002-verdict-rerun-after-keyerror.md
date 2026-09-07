---
id: ADJ-002
date: 2026-09-01
refs: [benchmarks/results/ADJUDICATION_LOG.md]
---

# ADJ-002 · 2026-09-01 · verdict n_rows=1 残留重跑

**调整**:在 PR #9 分支用完整 48 行行 JSON 重跑 `write_verdict`
(数据逐配置原子写未受损,判负标准未动)。

**根因证据**:rescue 目录 `latent_recursion_verdict.json` n_rows=1、
depths=[4]、gate tag=`stack_d4_vs_d4` —— core 单模式进程收尾
`judge_decision` 无条件访问 `tiers['stack']` → KeyError 崩出残留;
已由 commit 304ed35 修复(has_stack 保护 + no_stack_arms 降级)。

**验证**:重跑后 n_rows=48、depths=[1,2,4,8]、diagnosis 仍为 budget_wall
(内容不变,结构修正);回归测试 `test_latent_recursion_driver.py` 两例通过。

---

> 本文件为决策 ADR 工件（一决策一文件，正文不可变；勘误走新 ADJ 并在本文件尾追加指向）。
> 索引见 benchmarks/results/ADJUDICATION_LOG.md。
