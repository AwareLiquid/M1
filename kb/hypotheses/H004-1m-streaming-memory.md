---
id: H004
status: PROPOSED
thread: T002
created: 2026-09-06
refs: [docs/RESEARCH_PLAN.md, benchmarks/results/kv_frontier_ledger.json, RESULTS.md, arXiv:2501.00663, arXiv:2504.06214]
---

# H004 — O(1) 常量状态 + 跨会话联想记忆在 1M 流式上下文下"记忆-成本"结构性占优

## Statement

在开源小基座（1.7B–8B 级，开题时锁定）上，以本仓 O(1) 常量状态层 +
memory_broker 联想记忆为记忆本体，在 1M 流式上下文设定下：
(a) 流式会话的推理状态内存恒定（不随上下文增长）；
(b) 跨会话联想召回可迁移（0.56 任务族的流式版）；
(c) 端到端成本（训练 + 推理）结构性低于同基座 KV 方案。

坐标系：Titans/MIRAS 测试时记忆（学习式记忆模块）、GDN-2/KDA 线性注意力
（常量状态换固定容量）、UltraLong/LongRoPE（KV 路线的上下文扩展）；
差异格 = 跨会话持久联想记忆 + 诚实的成本记账（仓内 kv_frontier_ledger 口径）。

## Pre-registered judgment（升 UNDER_TEST 时写死于评测脚本并回显 JSON，届时不得移动）

H 支持（SUPPORTED）⇔ 同时满足：
1. 有效上下文：RULER 最长档得分 ≥ 同基座 LongRoPE 扩展基线的 0.9×；
2. 记忆：流式 1M token 会话中状态内存恒定（≤ 同基座 KV@64K 占用的 0.5×），
   且跨会话联想召回迁移任务达到预注册门槛（具体阈值随任务定义在开题时写死，
   参照系 = RESULTS.md 跨会话行任务族）；
3. 成本：端到端账本 ≤ appetite（A+B 段合计，docs/RESEARCH_PLAN.md §2.1）且
   推理期成本 / 1M-token 会话 ≤ KV 方案的 1/4（记账口径 = kv_frontier_ledger
   + compute_accounting）；
4. publishable 门：≥3 seeds、非双峰、配对检验 p<0.05 的可引用细胞 ≥ 2 个。

判负四选一诊断：context_extension_failure / memory_layer_overhead /
recall_transfer_failure / cost_ledger_miss。

## Evidence basis（只引不测）

- RESULTS.md proven 区：跨会话联想记忆任务上 attention/LoRA 对照恰好为 0
  （本仓王牌细胞，"记忆本体"假设的直接依据）；
- RESULTS.md proven 区：O(1) 常量状态 + 单核延迟 + ONNX parity 过线（部署可行性）；
- benchmarks/results/kv_frontier_ledger.json：非淘汰档全存活的成本口径基建；
- 外部（docs/RESEARCH_PLAN.md §0）：UltraLong/LongRoPE/Cerebras 扩展配方成熟；
  Titans/MIRAS 与 GDN-2/KDA/RAM-Net 构成活跃对话面。

## Verdict log（追加式）

- 2026-09-06：立项 PROPOSED（docs/RESEARCH_PLAN.md Bet A）。owner：
  AricRedemption；期限：A 段 POC 于首次 GPU 会话结束后判读（事件锚）。
  升 UNDER_TEST 的前置：#32（memory_broker）合入 + A 段基座/配方锁定。
