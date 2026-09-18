---
id: ADJ-014  # 编号勘误 2026-09-19：原稿占 ADJ-012，与已落地的机队定形件撞号，回溯改号（先例 H006）
date: 2026-09-09
refs: [kb/README.md, docs/guides/ITERATION_PRINCIPLES.md, HANDOFF.md, docs/PR_MERGE_POLICY.md]
---

# ADJ-014 — 教训层 kb/lessons/ 立层 + 执行层迭代止损 + H 立项外部扫描强制

**决策**：三项治理改动，同源立项（外部方法论对标，见根因）：

1. **kb/lessons/ 教训层**（`L###` 工件）：跨项目可复用的行为教训沉淀为
   "触发条件→对策→来源"结构条目。HANDOFF §5 现有 7 条迁移为 L001–L007，
   §5 原位改为指针。生命周期 ACTIVE → DORMANT/RETIRED（只命名不删除）；
   演进纪律（每合并窗口 ≤4 操作，ADD/EDIT/REMOVE/SUPERSEDE，准入判据
   "删掉会再犯吗"）写入 kb/README。§4.2 异常晋升（同一异常 ≥2 次）自此
   有了匹配载体：按 L 条目的 Trigger 节做匹配，不再依赖散落文本。
2. **执行层止损（ITERATION_PRINCIPLES §4.6）**：agent 调试会话采用双预算制——
   同一假设/同一故障连续修复尝 ≥3 次 → 停止，落 L 或 ADJ 工件后换方向
   或转 DORMANT；任何一次成功清零连续计数。语义对齐 AI-Scientist v2
   `max_debug_depth: 3`（连续失败深度，非总尝试数）。与实验层预注册止损
   （learnability gate 等）互补，不替代。
3. **H 立项外部扫描强制**（kb/README Evidence basis 节）：H 工件立项与
   判负后转向时，Evidence basis 必须含外部扫描（谁做过类似的事、失败在哪）。
   既往范本 = ADJ-003 的 aexp/HEP 逐项审计、ITERATION_PRINCIPLES 附录 B。
   另：ADJ 正文自此设 `DEVIATION` 字段（none 或写明协议偏离），
   对齐 ML pre-registration 惯例"偏离允许但必须显式文档化"。

**DEVIATION**：none（本决策即治理决策，无实验协议偏离）。

**根因证据**：
- 教训无结构化登记 → 异常晋升规则（ITERATION_PRINCIPLES §4.2）声明了
  "≥2 次强制评审"但无机制能数出"2 次"——教训散落在 HANDOFF §5 /
  ADJ 日志 / RESULTS RETRACTED 三处，无从按触发条件匹配；
- 执行层无止损 → 单会话内对同一方向死磕无硬规则，止损后动作依赖会话
  记忆而非工件（ADJ-003 诊断的"决策依据只存在当时会话记忆"）；
- 外部对标（2026-09-09 调研，均已读源码/官方文档验证）：ExpeL
  (AAAI 2024) 受控规则库演进（AGREE/REMOVE/EDIT/ADD 每轮 ≤4 操作、
  库满先 REMOVE 后 ADD）、AI-Scientist v2 BFTS `max_debug_depth: 3`、
  OpenEvolve `EvaluationResult{metrics, artifacts}` 错误侧信道、
  Claude Code 官方 2-strikes 清上下文规则、ML pre-registration workshop
  "协议偏离必须文档化"。选择：规则条目化取 ExpeL 形态但落 markdown
  （零运行时，对齐 ADJ-003"不引入运行时"基调）；止损阈值取 3（调试
  场景比对话纠错容错略宽，社区 2-3 之间）。

**验证**：
- `kb/lessons/` 7 个 L 工件 frontmatter/正文节齐全，Source 锚可解析；
- HANDOFF §5 改指针后行数不超 660 预算（`check_doc_budget.py` 绿）；
- ITERATION_PRINCIPLES §4.6/§4.7、kb/README L 纪律节文本可读、编号一致；
- 本文件为 ADJ 索引新增 1 行（索引 ≤60 行预算内）；
- 全量 pytest 不受影响（纯文档改动）。

**替代方案（被否）**：引入 JSON 错误注册表 + 自动 error_detector
（机器人变模特项目形态）——需要运行时与匹配逻辑，单人仓收益不抵维护
成本且 markdown 工件与 kb 三条铁律（不可变、单向引用、零依赖）同构；
教训条目"已知解决方案直接复用"模式被否——研究教训多为条件相关，
对策仍须走预注册。
