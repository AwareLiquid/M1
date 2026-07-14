# MT-LNN 迭代方法论：原则可进化的科学发现（PIEVO 映射）

> 2026-07-15 · 方法论来源：Pu, Lin, Chen — *Principle-Evolvable Scientific Discovery via Uncertainty Minimization*（arXiv:2602.06448，西湖大学，ICML 2026）。
> 核心思想：把迭代从"固定先验下搜索假设"转为"用异常证据进化底层原则"，用不确定性最小化选实验。
> 证据基础：DEEP_REVIEW_2026_07_14.md + RESULTS.md（2026-07-11 对账版）+ BENCHMARKS.md correction notes。

---

## 1. 诊断：本项目过去的迭代模式 = PIEVO 批判的"固定先验"模式

| PIEVO 批判的行为 | 本项目的对应历史 |
|---|---|
| 把初始先验当不可变约束 | P0 = "微管/Orch-OR/GWT 生物结构 → 智能优势"，14 个月未修订 |
| 在固定假设空间内低效穷举 | 70+ 模块全部是 P0 空间内的枚举（astrocyte/quantum/Φ×3/sleep/...） |
| 把矛盾证据当误差丢弃 | AVP 符号反转 → "等 125M 就会翻正"；broadcast_gate→0.0009 → 加竞争机制而非质疑瓶颈假设 |
| 无法利用异常扩展概念边界 | 异常都记录在案（好），但从未触发原则改写（坏） |

2026-06 起的 RESULTS.md 对账 = 无意识的后验更新。缺的是显式的原则表示 + 实验选择策略。

---

## 2. 原则台账（本文档的核心，随证据更新）

### P0 — 生物结构移植产生智能优势 · **状态：REFUTED**
证据：5 模块 switch-matrix 全 PPL 中性；broadcast_gate 训练后 0.0009；AVP FAILED（Φ̂ 符号反转）；
LNN 基线 ≈ MT-LNN（Selective Copy 长度扫描）；×42 撤回后剩 ×1.32。
**推论**：P0 空间内的新假设（新生物模块）先验信息量≈0，除非带预注册 kill criterion 否则不再受理。

### P1 — 跨窗口持久 + 选择性衰减的 fast-weight 状态是差异化能力 · **状态：SUPPORTED**
证据：跨窗召回 0.56 vs 0（对照组结构性归零）；移除 fast-weight → 0.008；v2s 选择性衰减 0.553→0.621；
v1 的 62.8M 无选择性 EMA 状态仅 0.002（说明"选择性"是必要条件）。
**待消不确定性**：合成 KV 对 → 真实对话/RAG 任务的迁移（见实验 E2）。

### P2 — 多时间尺度谱初始化是生物先验中唯一有效成分 · **状态：SUPPORTED（弱）**
证据：冻结 τ 得 0.285（46% 效果），可训 τ 0.621；但可学习 log_tau 反而 -7.5% 质量。
等价于 S4/LRU 的 timescale 谱初始化，无需生物叙事。

### P3 — O(1) 推理状态在同质量下是成本优势 · **状态：UNCERTAIN（前半真，后半未证）**
证据：0.381 MB 恒定 vs KV 384 MB@128k 为架构事实；但 O 系列 PPL = 2.15× 教师，"同质量"未达成。
**待消不确定性**：蒸馏 scaling 曲线是否指向可关闭的差距（见实验 E3）。

### P4 — 混合架构在预训练上优于同参数 Transformer · **状态：UNCERTAIN（高风险）**
证据：125M −31%，但单种子、2000 步欠训练、自建弱基线、无 Mamba/GPT-2 对照。
**待消不确定性**：公平基线 + 收敛训练（见实验 E1 — 最高优先级，因为它可能杀死头条声明）。

### P5 — 幻觉自我觉察机制降低幻觉率 · **状态：NEVER TESTED**
论文 Module E 仅为检测机制设计 + proxy 图；causal/self_monitor head 在 model.py 中 RESEARCH-NOT-WIRED。
不投入，除非 P1 的记忆 grounding 路径给出自然的测量场景。

---

## 3. 实验队列（按 信息增益/成本 排序）

| # | 杀死/验证 | 实验设计 | 成本 | 决策影响 |
|---|---|---|---|---|
| **E1** | P4 | 同 setup 加 Mamba-130M + GPT-2-124M 基线，2000 步 ×3 种子 | Kaggle 免费 ~2 天 | 杀死 → 停预训练线，全力 adapter/记忆线；存活 → 才值得投收敛训练 |
| **E2** | P1 迁移性 | 真实多轮对话记忆 harness（非合成 KV）：N 轮后问前文事实，对照 = 同 budget 的 RAG | ~1 周 | 存活 → M1 成为 Awareness 记忆产品的底层差异化；这是研究线↔产品线的接口实验 |
| **E3** | P3 | 用已有 3 轮蒸馏数据点拟合 loss-vs-token 曲线，外推到教师水平所需 token | ~0（纯分析） | 曲线不收敛 → O 系列止损；收敛 → 按外推值申请算力 |
| **E4** | 负资产清理 | `use_decay_wm=False` + `use_predictive_coding=False` 默认，重跑基准 | 1 天 | 已有消融背书（PC 趋负 +0.65 PPL；decay_wm 逐 token Python 循环） |
| E5 | 工程投资 | fused attention bias（解锁 T≥4096 训练 + 降显存） | 2-4 周 | **仅当 E1/E2 存活后执行** — 工程投入 gated on 原则验证 |

排序原理（PIEVO 的 uncertainty minimization）：确认性实验信息量≈0，优先跑**最可能证伪核心命题的最便宜实验**。E1 排第一正因为它最危险。

---

## 4. 制度（防止回退到固定先验模式）

1. **本文件是活的原则台账**：每次 benchmark/消融结束，先更新原则状态，再决定下一个实验。原则状态变化 = commit message 一等公民。
2. **异常晋升规则**：同一异常出现 ≥2 次 → 强制原则修订评审。禁止用"再加一个模块"响应异常。
3. **新模块准入 = 预注册 kill criterion**：写代码前先写下"什么实验结果会杀死它"+ 谁跑、何时跑。没有 kill criterion 的模块进 `experiments/`，不进 `mt_lnn/` 主包。
4. **原则空间收缩**：23 个孤儿模块 + P0 的已证伪模块迁出主包。空间越小，迭代样本复杂度越低，eval bug 藏身处越少。
5. **测量优先于声明**：任何对外数字必须能溯源到 RESULTS.md PROVEN 表。O1 公开 README 与 PROVEN 表的一致性纳入发布检查（当前不一致：×42/needle 表待撤，见 DEEP_REVIEW §三）。

---

## 5. 一句话版本

**从"证明微管先验正确"转向"沿着异常进化原则"：现在的证据把紧凑原则空间收缩为 {持久 fast-weight 记忆, 选择性衰减, 时间尺度谱, O(1) 足迹}——下一步不是加模块，而是用 E1 杀验头条、用 E2 打通产品接口。**

---

## 附录 A · "通用智能技术体系"框架对账（2026-07-15）

针对外部流传的 AGI 技术栈框架（时序架构 / 认知核心 / 分层记忆 / 系统级配套）与本项目的逐项对账。
三档口径：✅ 有且被证明有效 · 🟡 有代码但惰性/未验证 · ❌ 没有。**"有代码"≠"有能力"是本项目最贵的教训。**

| 组件 | 档位 | 证据/位置 |
|---|---|---|
| SSM（Mamba/RWKV） | 🟡 有等价物 | v2 selective decay + chunked fast-weight ≈ GLA 一脉；E1 正面对比进行中 |
| SNN 脉冲网络 | ❌（孤儿代码） | `stdp_ops.py` 无人 import。**决策：不投**（见下） |
| MoE 稀疏路由 | 🟡 轻量版未验证 | κ-gate sparse resonance + compute skipping + GWTB 竞争路由；吞吐收益从未测量 |
| 世界模型（next-state） | 🟡 一支有测量 | PredictiveStateHead LM 上中性；`hamiltonian_head` 在物理轨迹任务能量漂移 3–6×/rollout 2–24× 优于 MLP（**除 fast-weight 外唯一有数字的认知模块**，生态位限于连续状态域） |
| 预测编码 | ❌ 实测有害 | switch-matrix PPL 趋负 +0.65，E4 已默认关闭 |
| 主动推理/自由能 | ❌ 孤儿代码 | `active_inference.py` 从未接线 |
| **三层持久记忆** | ✅ **最强项，横跨 M1+Awareness** | 情景=fast_weight_store（跨窗召回 0.56 PROVEN）；语义=knowledge_memory KB + Awareness knowledge cards；程序=Awareness skills 表（30 天半衰期）；自适应遗忘=LRU+skill decay |
| 神经符号 | ❌ | 无符号引擎 |
| 具身闭环 | ❌ | 只有感知半边（multimodal/sensory），无行动-反馈环 |
| 自主目标规划 | ❌ 残迹 | imagination/deliberation 未成体系 |

**优化优先级（证据排序）**：1) 三层记忆＝P1+E2 主线，两仓库拼合即完整分层，全力投入；2) hamiltonian 物理世界模型＝有数字的生态位，视产品方向深化，不指望改善 LM；3) MoE-lite＝E1 后带 kill criterion 排期。

**SNN 明确不投的三个理由**：(a) 能耗收益被神经形态硬件锁死，GPU 上替代梯度训练是质量/吞吐双输，且无相关产品面；(b) 属 P0（生物结构移植）空间的最深赌注，该原则已 REFUTED，按异常晋升规则需极强先验证据，当前为零；(c) SNN 的实用部分（时间稀疏 + 多时间尺度）已由 τ 谱初始化（P2）+ selective decay + compute skipping 在稠密硬件上实现。`stdp_ops.py` 随孤儿模块归档。

**方法论提醒**：该框架的叙事结构与本项目 2024 年的 P0 叙事同构（大而全生物启发清单）。过滤规则不变：任何组件不带可测量声明 + kill criterion 不进主包。
