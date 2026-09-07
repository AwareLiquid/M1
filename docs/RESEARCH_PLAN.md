# 研究路线图 · 两个 bet 与预算纪律（2026-09-06）

> 本文件是论文/工程两条线的**唯一事实源**：会话记忆中的计划不作数（ADJ-003 教训）。
> 状态纪律：§1 表格随状态变化更新（每合并窗口复核）；判定回填对应 H 工件 Verdict log。
> 预算按 §2 的 appetite **阶段审批**：价值判据 SUPPORTED 才解锁下一笔，判负/超限即停。
> 方法论对齐：kb/H 预注册（本仓）× spec-driven development（spec 为唯一事实源）
> × Shape Up pitch（Problem / Appetite / Solution / Rabbit holes / No-gos，
> bets-not-backlogs：只立 bet，不做无限 backlog）。

## 0. 调研快照（2026-09-06 联网调研，外部事实只引不测）

**长上下文前沿**

- DeepSeek V4 把 1M 做成门槛（双 MoE：V4-Pro / V4 Flash ≈ $0.14/M tokens）
  [arXiv:2606.19348]；Gemini 3 Pro 标称 10M；Claude Sonnet 4 / Qwen 均 1M 档
  [morphllm 对比表]。
- **标称 ≠ 有效**：业界焦点已从"长度"转向"有效性与成本"；部分 1M 模型用户实测
  ~90K 起幻觉 [benchlm；r/LocalLLaMA]。
- **经济性痛点**：同等工作量下长上下文成本 ≈ RAG 的 20–24× [morphllm]。

**记忆/线性注意力热点（论文坐标系）**

- Titans（Google，NeurIPS 2025，381+ 引用）+ MIRAS：测试时记忆
  [arXiv:2501.00663；Google Research blog]。
- Gated DeltaNet-2（NVIDIA 2026-05，解耦 erase/write）[arXiv:2606.26560]；
  Kimi KDA、Mamba-3；Qwen3-Next 采用 3:1 GDN:attention 混合 [Raschka]。
- RAM-Net（可寻址记忆 + 常量内存）[arXiv:2607.07953]；Log-Linear Attention（ICLR 2026）。

**小模型 1M 扩展配方（工程可行性依据）**

- NVIDIA UltraLong-8B：开源 1M/2M/4M 权重，两阶段 CPT + iT [arXiv:2504.06214，ACL 2026]——**可直接作 Bet A 基座，跳过自扩（2026-09-06 修订采纳，见 §4）**。
- **差异化校准（2026-09-06 修订）**：Qwen 系已给 1M、DeepSeek V4 把 1M 做成门槛、标称 ≠ 有效（~90K 起幻觉）→ 纯"长度"无卖点。Bet A 主张钉死在两件别人没有的事：**跨会话持久联想记忆**（S3 王牌细胞）+ **诚实的成本记账**（kv_frontier_ledger 口径）。
- **benchmark 缺口（2026-09-06 修订）**：测试时记忆热点线（Titans/MIRAS，381+ 引用）缺"跨会话持久记忆"公开评测 → H004 增设 benchmark 子贡献（现成任务族 + 预注册纪律，CPU 级成本）。
- LongRoPE：~1k 步微调上 2M [arXiv:2402.13753]；Cerebras：合成数据省 99% 训练 token。
- 注意：二次注意力使长上下文 CPT 天然昂贵 [arXiv:2603.17484]——正是线性/常量状态的切入口。

**KaaS/RAG 定价锚点**

- Exa $7/千次（免费 20k/月）；Tavily $0.008/次；deep search $12–15/千次 [ColdIQ]。

**档期**

- ICLR 2027：摘要 2026-09-18 / 全文 09-25 [iclr.cc]——来不及新实验，放弃。
- 目标：arXiv 预印本（2026-10/11）→ ICML 2027（约 2027-01 截稿，届时官网核实）。

**算力与 No-go 账**

- A100 单卡 ¥70/天（≈$10）。从零预训练对标旗舰（≥100B × 10T tokens ≈ 6×10²⁴ FLOPs）
  ≈ 55 万 A100 日 ≈ ¥4,000 万 → **结构性不可行，列为 No-go**。

**本仓资产锚定（只引 proven 区，详见 RESULTS.md）**

- 跨会话联想记忆：attention/LoRA 对照恰好为 0 的王牌细胞。
- O(1) 常量状态：状态恒定、单核 26µs/步、ONNX parity 过线；KV frontier ledger 全套。
- memory_broker 三后端 + 固化策略（#32 待合入）；KaaS grounding 原型（PR #25 开放）。

## 1. 优先级清单（状态表，随做随更）

| 优先级 | Bet / 事项 | 类型 | appetite（预算纪律） | 状态 | 载体 |
|---|---|---|---|---|---|
| **P0-A** | 1M 流式记忆小模型（POC → 论文 → API） | 论文+工程同体 | 两段制：A 段 ≤¥500 → gate → B 段 ≤¥1500 | **未开工**（等 #32 合入） | T002 / H004 |
| **P0-B** | KaaS grounding 付费验证 | 工程/商业 | ≤¥300（serving 租金） | **未开工**（等 #25 补模板节） | T003 / H005 |
| 观察名单 | C 连续时间/事件流（workshop）；D 评测方法论；F O(1) serving 放大；G 边缘传感垂直；H 端侧个人记忆 | — | — | 不立 bet，触发条件见 §2.3 | ADJ-005 惯例 |

## 2. 两个 bet 的 pitch

### 2.1 Bet A：1M 流式记忆小模型（T002 / H004）

- **问题**：旗舰 1M 上下文贵（20–24×）且有效上下文缩水；智能体长会话真正需要的是
  "永不忘的流式记忆"，不是"更长的窗口"。
- **方案（2026-09-06 修订）**：基座直接采用开源 `Llama-3.1-Nemotron-UltraLong-1M`
  （跳过自扩；备选 Qwen3-1.7B/4B + LongRoPE，若许可/适配不顺）→ 挂本仓 O(1) 常量
  状态层 + memory_broker 联想记忆。差异化主张：**跨会话持久联想记忆（S3 王牌细胞）
  + 诚实的成本记账**；附 **benchmark 子贡献**：跨会话持久记忆公开评测（含 CSE 式
  采样偏置修正，arXiv:2608.17293）。注意：贡献不是"扩到 1M"，是"1M 之上的持久
  记忆与成本结构"。
- **Appetite（预算纪律）**：阶段审批制。A 段 POC ≤¥500（**修订后预期 ≈¥200**，
  自扩工作已由 UltraLong 开源权重消除）：基座 1M 基线复现 + O(1) 记忆层最小接入
  → 按预注册 gate 判读；判 SUPPORTED 才解锁 B 段 ≤¥1500：评测矩阵 + benchmark
  打包 + arXiv 预印本。超限或判负即停；判 SUPPORTED 后按价值追加，下不封顶。
- **Rabbit holes（坑，2026-09-06 修订）**：UltraLong 许可/权重可用性与 tokenizer
  兼容（开题日核销，备选 Qwen+LongRoPE 路线保留）；记忆层接入的位置编码/上下文
  交互（最小侵入先行）；长评被采样分布偏置（CSE, arXiv:2608.17293——评测口径
  预注册，直接修进 benchmark 子贡献）。
- **No-gos**：不从零预训练对标旗舰；不做通用 1M 模型（只做流式记忆差异化）；
  不承诺通用能力超越 DeepSeek/Gemini。

### 2.2 Bet B：KaaS grounding 付费验证（T003 / H005）

- **问题**：小模型知识不足；现有 RAG/搜索 API 给链接不给可核查来源。
- **方案**：#25 的 awareness grounding（1,027 条带 DOI 来源知识）包装成 API：
  检索命中→带来源注入；无命中→拒答。差异化 = **DOI 可核查**。
- **Appetite**：≤¥300（serving 租金）；上线后 30 天窗口判读。
- **Rabbit holes**：获客渠道（种子用户从 agent 开发者社区来）；与 Exa/Tavily
  正面竞争（不打搜索，打"可核查知识"）。
- **No-gos**：不做通用搜索引擎；不补贴烧钱获客。

### 2.3 观察名单（触发即升级为 bet，ADJ-005 惯例）

- C 连续时间/事件流：触发 = Bet A 段 POC 判读后仍有算力富余。
- D 评测方法论：触发 = P0 论文投出后。
- F O(1) serving 放大 / G 边缘传感 / H 端侧记忆：触发 = 对应 bet 判 SUPPORTED。

## 3. 执行手册（agent 可执行）

- **租用与记账**：A100 ¥70/天；成本按仓内 compute_accounting 惯例落 JSON；
  每段结束对账（预算 → FLOPs → 天数）。
- **Bet A 段步骤（2026-09-06 修订）**：① 基座 = UltraLong-1M 取权重 + license
  核查（备选 Qwen3+LongRoPE）→ ② RULER/LongBench 基线复测（验证基座 1M 可复现）
  → ③ O(1) 记忆层最小接入（memory_broker 接口）→ ④ H004 判据写死于评测脚本并
  回显 JSON（升 UNDER_TEST）→ ⑤ 对账回填 §1 + H004 Verdict log。
- **评测口径**：RULER（有效上下文）、LongBench v2、流式会话回放（自建，预注册）、
  跨会话召回（0.56 任务族的迁移版）；全部 per-seed + config 回显（仓内纪律）。
- **Bet A B 段（gate 通过后）**：O(1) 记忆层接入（memory_broker 为接口）→ 评测矩阵
  → 成本对比账（KV vs O(1)，kv_frontier_ledger 口径）→ arXiv 预印本（ICML 2027 冲刺）。
- **Bet B 步骤**：#25 补模板节（结论状态/AI 披露）→ 合入 → API 上线 → 30 天付费
  窗口 → H005 判读。

## 4. 回填纪律与决策记录

- 本文 §1 表格随状态变化更新；每合并窗口复核一次（含观察名单触发条件）。
- 判定落地三件套不变：预注册标准 + 证据轨 JSON 入 `benchmarks/results/` +
  RESULTS.md 状态位回填（kb/README 纪律）。
- **决策记录（追加式）**
  - 2026-09-06（修订二）：调研三项落地方案采纳——① 基座改 UltraLong-1M 开源权重
    （跳过自扩，A 段预期成本 ¥500→≈¥200）；② 差异化校准（纯长度无卖点，钉死
    跨会话持久记忆 + 成本记账）；③ H004 增设 benchmark 子贡献（跨会话持久记忆
    公开评测 + CSE 偏置修正）。H004 判据框架不变（基座"开题时锁定"条款已覆盖）。
  - 2026-09-06：立项本文档 + T002/T003 + H004/H005（PROPOSED）。owner：
    AricRedemption。依据：RESULTS.md proven 资产 × §0 外部快照 × Shape Up
    bets-not-backlogs 纪律。放弃 ICLR 2027（9/18 截稿来不及），目标 arXiv 先行
    + ICML 2027。
