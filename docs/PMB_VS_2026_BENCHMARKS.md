# PMB vs 2026 新基准 — 对照与差异化定位（B-10 产出）

> 2026-09-11 · 循环扫描轮产出（B-10 CANDIDATE 判读）
> 扫描范围：arXiv 2608.xxxxx 三篇 + mem0 行业报告 + Stanford MemoryArena。
> 结论先行：**PMB 的三个核心差异化全部存活**，其中两个对方完全没有；
> 但生态位正被快速占领（5 篇/6 个月），开源与定位文档须尽快落地。

---

## 一、对照对象总览

| 对象 | 时间 | 内容 | 与 PMB 的重叠度 |
|---|---|---|---|
| AgentMemBench (arXiv:2608.00009) | 2026-06 | 5 种记忆管理策略（ICW/EKV/GEM/CBS/WAM）× 3 公开数据集，Recall@k/MRR/F1/足迹/延迟 | 中：同测"跨会话记忆"，但任务形态 = 检索质量评测（LoCoMo/MultiDoc2Dial/MSC），非程序生成事实绑定 |
| StateMemBench (arXiv:2608.19652) | 2026-08 | 演化状态跟踪：234 多会话场景，closed-pool 三态评分（current/superseded/fail） | **高：与 PMB T2 同题**——但 T3 遗忘曲线、机械持久化均无 |
| MCB (arXiv:2608.19564) | 2026-08 | 记忆澄清边界：persist/use/verify/ask 四分，LLM 决策层评测 | 低-中：PMB T5 拒答是其子问题（evidence 模式无决策层，结构不同） |
| MemoryArena (Stanford) | 近期 | Memory-Agent-Environment 循环评测 gym | 待细读（环境交互维度 PMB 未覆盖） |
| mem0《State of AI Agent Memory 2026》 | 2026-04 | 行业报告：LoCoMo/LongMemEval/BEAM + 21 框架集成 | 生态图——PMB 未被提及（= 尚无人占位） |
| **Memora (arXiv:2604.20006)** | 2026-04 | weeks-months 个性化对话记忆：remembering/reasoning/recommending 三任务 + FAMA 失效感知指标；4 LLM × 6 记忆 agent | 中：LLM 记忆体评测（非机械持久化）；reasoning ≈ 多会话聚合（t4 族）；FAMA ≈ T2 stale 思想的 LLM 层版本。**无遗忘曲线、无位级持久化** |
| **FwPKM (arXiv:2601.00671)** | 2026-01 | fast-weight product key memory：稀疏槽位 episodic 层，TTT 式槽位梯度更新 | 系统侧（非基准）：序列内 LM 困惑度 + NIAH 128K；**无跨会话评测、无公开代码**——"episodic 语义"定位与 PMB 持久化主轴相邻但未做会话语义 |

## 二、逐维度对照（PMB T1-T5 vs 全部新基准）

| 能力维度 | PMB | AgentMemBench | StateMemBench | MCB | 结论 |
|---|---|---|---|---|---|
| 跨会话事实保持（T1 全网格 N×K） | ✅ 3-seed 配对 + 防枚举截断 | 部分（Recall@k 检索式，无 N×K 网格） | 部分（superseded 三态，无网格） | ❌ | **PMB 独有网格化设计** |
| 流式更新/演化状态（T2 stale 三指标） | ✅ recall + stale_answer_rate + stale_context_rate | ❌ | ✅ **同题且更细（闭池三态）** | 部分（freshness 判定） | StateMemBench 同题——**引用并对照，不重复造** |
| **遗忘曲线全曲线（T3，gap∈{0,2,8,32}）** | ✅ 单点报告明确无效（capacity-honesty 条款） | ❌ | ❌ | ❌ | **PMB 独有，无任何新基准覆盖** |
| 跨会话 JOIN（t4，新） | ✅ | ❌ | ❌ | ❌ | **PMB 独有**（且 B-8/B-9 发现：JOIN 是编码器敏感的检索盲区） |
| 拒答/未知键（t5） | ✅ false_answer_rate（结构上限已实证=100%） | ❌ | 部分（fail 态） | ✅ 决策层四分（但评 LLM 决策非记忆本体） | PMB 有机械指标；决策层协议（t5b）待做 |
| **bit 级持久化机械强制**（snapshot/restore 逐会话落盘，进程内=0 分） | ✅ **harness 构造性保证** | ❌（无会话边界语义） | ❌ | ❌ | **PMB 独有，且是"持久记忆"四个字的字面底线** |
| 防枚举截断（anti-cheat） | ✅（4000 chars / 200 chars 双层） | ❌ | ✅ closed-pool 思路相近 | ❌ | PMB 更严格（数值化） |
| 被测对象 | 记忆系统本体（MemorySystem 接口） | 记忆管线 + LLM 生成 | 记忆系统 + LLM | LLM 决策层 | PMB 纯度最高（隔离生成质量干扰） |

## 三、判定（B-10 预注册判负标准检验）

> 预注册判负："若对方已覆盖遗忘曲线全曲线 + bit 级持久化 → 差异化不存在"

- 遗忘曲线全曲线：**AgentMemBench ❌ / StateMemBench ❌ / MCB ❌** → 未被覆盖 ✅
- bit 级持久化机械强制：**三者全部 ❌**（AgentMemBench 无会话边界语义；StateMemBench
  有多会话但快照/落盘语义未强制）→ 未被覆盖 ✅

**判定：判负未触发，差异化成立。** B-10 从 CANDIDATE 转正。

## 四、行动项（按杠杆排序）

1. **StateMemBench 引用与对照**（同为演化状态题）：在 PMB README 增"Related work"
   节引用它，并明确 PMB T2 的差异（防枚举截断 + 网格化 stale 指标 + 遗忘曲线）；
   评估 StateMem 的 state-first 思路是否可作 PMB 的一个参赛系统（引用即对比）。
2. **t5b 拒答协议**（B-9 浮现）：MCB 是 LLM 决策层评测，PMB 是记忆本体评测——
   两者正交；t5b 的设计应引用 MCB 的四分法（persist/use/verify/ask）作为决策
   分类学。
3. **开源节奏提前**：5 篇/6 个月的占领速度下，PMB 应在 ~2-4 周内完成
   (a) README 定位重写（三轴差异化 + Related work）(b) PyPI/GitHub 公开包
   (c) 引用 AgentMemBench/StateMemBench 的对照表。
4. **不做的事**：不追 LoCoMo/MultiDoc2Dial 式 LLM 生成对话数据（偏离 PMB 的
   程序生成 + 精确评分优势线）。

## 五、证据与来源

- arXiv:2608.00009 (AgentMemBench, 2026-06-16)
- arXiv:2608.19652 (StateMemBench, 2026-08-20)
- arXiv:2608.19564 (MCB, 2026-08-20)
- MemoryArena: https://digitaleconomy.stanford.edu/publication/memoryarena-benchmarking-agent-memory-in-interdependent-multi-session-agentic-tasks/
- mem0 report: https://mem0.ai/blog/state-of-ai-agent-memory-2026
- PMB 锚点：benchmarks/persistent_memory/README.md（三轴声明 + 防枚举截断 + 机械持久化）

## 六、扫描轮 3 增补（2026-09-16）：Memora/FwPKM 核对 + 行业定位强化

- **Memora 核对结论：三差异化仍全部存活**。其 FAMA（失效感知记忆准确率）
  是 T2 stale 思想的 LLM 决策层版本——PMB t2 的 stale_answer_rate 已是该
  现象的机械指标；Memora"记忆 agent 频繁复用失效记忆"的核心发现恰是 t2
  stale 语义的 LLM 层印证（可互引）。遗忘曲线全曲线与 bit 级持久化仍未覆盖。
- **FwPKM 归系统侧对照**（同 Falcon/MoNe/KDN 谱系）：episodic 槽位记忆的
  "序列内"定位再次印证主轴 (a) 的缺口表述——episodic ≠ 跨会话持久化。
- **行业定位强化**：mem0 2026 报告确认 cross-session identity resolution 与
  temporal reasoning 为行业最难开放问题——PMB t4 即前者的机械最小化
  （B-16 两跳基线已证明协议可解：hash 0.0→0.583）。
- **PMB 侧新增证据**（同日产出，强化独有维度）：T3 遗忘曲线三类系统形态
  分离（叠加=容量断崖 / 梯度=级进衰减 / 检索=免疫，B-14/B-17/B-18/B-19）；
  t5b + B-16 协议缺口配对论证。生态占领速度更新：原 5 篇/6 个月口径 +2
  （Memora/FwPKM），系统侧 9 月仍有新进入者（KDN, 2609.07816）。
- 证据源增补：arXiv:2604.20006 (Memora) · arXiv:2601.00671 (FwPKM) ·
  arXiv:2609.07816 (Kalman Delta Networks)
