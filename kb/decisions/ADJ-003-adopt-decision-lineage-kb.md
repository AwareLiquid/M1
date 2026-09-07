---
id: ADJ-003
date: 2026-09-01
refs: [benchmarks/results/ADJUDICATION_LOG.md]
---

# ADJ-003 · 2026-09-01 · 采纳决策血缘规范(kb/ + DECISION_TRACE_SPEC)

**调整**:引入 kb/ 假设层(H/T 工件)、`docs/DECISION_TRACE_SPEC.md`
血缘字段、本日志;**不引入** aexp 运行时(signac/Claude Code hook/kb 全量模板)。

**根因证据**:交接丢档事故(REGISTRY 回收记录:"潜空间递归判决实际存在
但交接时以为丢失")暴露假设层缺位;pointer_chase d8 误立项暴露
"决策依据只存在于当时会话记忆"。外部参考:HEP(协议)、aexp(工程实现,
560 测试通过但 Beta + 单人 + signac 3.0 悬崖 + hook 仅 Claude Code 生效)、
PROV-AGENT(W3C PROV 语义)。

**验证**:新判决级 JSON 自 2026-09 起带 `lineage` + `protocol` 字段;
合并窗口门禁(PR #14 落地后)校验 H→JSON 链完整;首批 H001–H003 建档。

---

> 本文件为决策 ADR 工件（一决策一文件，正文不可变；勘误走新 ADJ 并在本文件尾追加指向）。
> 索引见 benchmarks/results/ADJUDICATION_LOG.md。
