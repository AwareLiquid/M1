---
id: H005
status: PROPOSED
thread: T003
created: 2026-09-06
refs: [docs/RESEARCH_PLAN.md, RESULTS.md]
---

# H005 — 带 DOI 可核查引用的知识 grounding API 存在真实付费需求

## Statement

面向小模型 agent 的知识 grounding 服务（PR #25：检索命中→带来源注入，
无命中→拒答）以"引用可核查（DOI）"为差异点，能获得真实付费用户，
而非只停留在试用。

坐标系：Exa / Tavily（搜索/RAG API，给链接）、Perplexity API（答案引擎）；
差异格 = 可核查引用 + 面向 agent 的拒答纪律（绝不编造）。

## Pre-registered judgment（上线时写死于运营口径，届时不得移动）

SUPPORTED ⇔ 30 天窗口内同时满足：
1. ≥3 个独立付费用户 或 累计收入 ≥¥500；
2. 引用可核查抽检：随机 100 条 grounding 输出的 DOI 100% 可解析。

INCONCLUSIVE（非判负）：付费为 0 但活跃试用 ≥20 → 定价/渠道问题，
转观察名单（6 个月后复审一次）。

REFUTED ⇔ 活跃试用 <20 且付费 0（差异点不成立的诚实结论）。

## Evidence basis（只引不测）

- PR #25：grounding 检索 / 来源注入 / 拒答纪律已实现并冒烟通过（hook）；
- docs/RESEARCH_PLAN.md §0：外部定价锚点（Exa/Tavily 按调用计费的成熟价格带）；
- 本仓 kb 生命周期纪律本身即"可核查来源"文化的 proven 底座。

## Verdict log（追加式）

- 2026-09-06：立项 PROPOSED。owner：AricRedemption；期限：上线后 30 天
  （事件锚 = API 公开可用日）。
