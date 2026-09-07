# ADJUDICATION_LOG — 决策日志(人写,追加式)

> 每条记录一个关键决策/调整:调整了什么、根因证据、如何验证。
> 与 `kb/hypotheses/`(假设档案)、`docs/DECISION_TRACE_SPEC.md`(字段契约)配套。
> 纪律:必写"根因证据"(文件/数据/commit),必写"验证"(改后怎么确认是对的)。

---

> **纪律（2026-09-06 起）**：决策正文一律住在 `kb/decisions/`（一决策一文件，ADR 工件，
> 不可变、勘误开新 ADJ）；**本文件只做索引——每条决策 1 行，禁止在表下追加正文**。
> 活文档只放当前态，剧情归 git 历史与归档（docs/archive/）。

| 决策 | 日期 | 标题 |
|---|---|---|
| [ADJ-001](kb/decisions/ADJ-001-latent-recursion-learnability-gate.md) | 2026-09-01 | latent-recursion 判决任务脱出 budget_wall |
| [ADJ-002](kb/decisions/ADJ-002-verdict-rerun-after-keyerror.md) | 2026-09-01 | verdict n_rows=1 残留重跑 |
| [ADJ-003](kb/decisions/ADJ-003-adopt-decision-lineage-kb.md) | 2026-09-01 | 采纳决策血缘规范(kb/ + DECISION_TRACE_SPEC) |
| [ADJ-004](kb/decisions/ADJ-004-arr-phase-b-arm-discipline.md) | 2026-09-01 | arr Phase B 缺臂补齐判据钉死 |
| [ADJ-005](kb/decisions/ADJ-005-non-migration-audit-observation-list.md) | 2026-09-01 | 决策血缘迁移的"不迁移清单"审计 |
| [ADJ-006](kb/decisions/ADJ-006-three-track-gate-no-conclusion-merges.md) | 2026-09-04 | 追溯登记三笔"无结论合并"违规 + 门禁三轨拆分 |
| [ADJ-007](kb/decisions/ADJ-007-ct-probe-evidence-clearance.md) | 2026-09-04 | CT 机制探针判决证据清偿入库(单点丢档第 3 次) |
| [ADJ-008](kb/decisions/ADJ-008-edge-line-batch2-clearance.md) | 2026-09-06 | 边缘证据线批次2清偿:生产者代码 + 6 份判负/加固证据 + 判定轨回填 |
| [ADJ-009](kb/decisions/ADJ-009-merge-window-consolidation.md) | 2026-09-07 | 绿 PR 队列批次整合纪律(合并窗口制) |
