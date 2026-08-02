# M2 Roadmap — 2B 推理引擎:小模型对标 70B 的路线图
# M2 Roadmap — A 2B Reasoning Engine That Competes With 70B on Chosen Axes

> 版本 v1.0 · 2026-07-28 · 战略路线图(实验日志见 `BRAIN_INSPIRED_ROADMAP.md`,基准数据见 `BENCHMARKS.md` / `RESULTS.md`)
>
> **核心命题 / Thesis**:知识外置(RAG),本体只做推理与记忆控制。
> 2B 本体 = 纯推理引擎 + 记忆控制器;在数学/代码推理、长流式记忆、持续学习、端侧延迟四个轴上对标 70B base 模型。
> **不承诺**"全面对标 70B"——知识容量随参数缩放是物理规律,任何对外材料不得写全面超越。

---

## 1. 现状诚实评估(2026-07 实测)/ Honest Baseline

M1 本质:**带生物模块封装的线性递归模型(SSM 家族)**,与 Mamba/RWKV/Griffin 同能力类别。

| 维度 Dimension | 实测 Measured | 定性 Verdict |
|---|---|---|
| PPL(语言建模)| 与同参数 transformer 统计打平(10-seed)| 及格线,非卖点 |
| 推理内存 Inference memory | O(1) 恒定,1M token 处 **8063×** 优势 | 真实差异化 ✓ |
| 跨窗口记忆 Cross-window recall | 0.56 vs transformer 0.00 | 真实差异化 ✓ |
| 不规则采样 Irregular sampling | 退化 +7.7% vs LSTM/GRU +31~33% | 真实差异化 ✓ |
| 训练稳定性 Training stability | ~4× 更稳(loss variance)| 工程优势 ✓ |
| Hebbian 模块 | **实测惰性(inert)**,PPL 贡献 < seed 噪声 | 待改造或砍除 |
| GWT / 预测编码 / 睡眠 | 部分有数据(见实验日志复测五~十二:EWC ✓、经验回放 ✓、生成式做梦重放 ✓ 且 PC 的梦更好)| 贡献部分量化,需补齐 ablation |

**一句话**:M1 现在赢在"效率与记忆形态",不赢在"思考能力"。M2 的全部工作就是补思考能力。

**In one line**: M1 wins on efficiency and memory form-factor today, not on reasoning. M2 exists to close the reasoning gap.

---

## 2. 为什么 2B 有机会在特定轴上打 70B / Why a 2B Can Win on Chosen Axes

三个有确凿外部证据的突破口:

1. **窄域推理可越级(蒸馏 + RL)** — DeepSeek-R1-Distill-Qwen-1.5B 在 AIME 上超过 GPT-4o。路径:强教师推理轨迹蒸馏 → SFT → GRPO(可验证奖励)。
2. **递归深度 = 用计算换参数** — HRM(27M 参数)靠循环递归推理在 ARC-AGI 上打赢大模型;latent recurrent-depth 工作(Geiping et al. 2025)证明同一核心循环 N 次可替代更多层数。**M1 的液体核心天生递归——这是我们架构与该路线的天然契合点,目前未被利用。**
3. **知识外置** — 70B 的大部分参数在"背书"。2B + 检索 + 学习型记忆控制器可卸掉知识负担,把参数全部留给推理回路。

对标声明的纪律:只在【数学/代码推理、长流式记忆(RULER/流式基准)、持续学习(任务链)、端侧延迟/内存】四轴上做 head-to-head;每个对比注明对方是 base 还是 instruct。

---

## 3. 生物模块改造表 / Bio-Module Refit Plan

原则:**功能上像大脑,而不只是命名上像大脑。** 每个模块必须有独立 ablation,贡献 < seed 噪声的模块砍除。

| 模块 | 现状 | 改造方案(神经科学对应 + 有效 ML 技术) | 验收标准 |
|---|---|---|---|
| **Hebbian** | 惰性 | 改成 **fast weights / test-time learning**(DeltaNet、Titans 路线):推理时按 surprise 更新快速权重矩阵,实现"边用边学"。改不动就删,不留装饰品 | 在 recall 任务上 ablation ≥ 2σ 增益,否则删除 |
| **预测编码 PC** | 模块名 + 部分实验信号(做梦重放中 PC 的梦更好,5/5 种子) | 变成**真实训练信号**:逐层预测下一时刻潜变量,预测误差做辅助 loss;**误差大 = surprise = 记忆写入门控**(海马编码触发机制),直接驱动 RAG 写入决策 | 辅助 loss 使主任务收敛更快或更稳(多种子);surprise 门控写入优于均匀写入 |
| **GWT 瓶颈** | 未完全量化 | 真正的**稀疏全局工作区**:k-winner 竞争广播,仅赢家进入全局状态;天然形成可解释"注意焦点" | ablation 证明贡献;广播稀疏度-性能曲线 |
| **睡眠固化** | 概念 + 生成式重放实验 ✓ | **离线重放蒸馏**:空闲期把 episodic buffer(RAG 库高价值条目)蒸馏进权重 + EWC 防遗忘。互补学习系统:**权重=皮层(语义),向量库=海马(情景)** | 任务链持续学习基准上,"睡过"的模型显著优于未睡 |
| **5 时间尺度 τ** | 已有 | 保留,升级为 HRM 式**分层递归**:慢尺度规划、快尺度执行;推理时可变步数循环 + 自适应停机("难题多想几轮") | 思考步数 vs 准确率曲线单调上升 |

改造完成后,"大脑式思考"= 四个可验证机制:**迭代递归推理、surprise 驱动记忆、稀疏全局广播、睡眠期固化**。每个可独立出论文图。

---

## 4. 三阶段计划 / Three Phases

### P0 — 把递归推理证出来(现在,本地 RTX 5060 8GB 可做)

目标:整个 thesis 的最小证据 —— **"多想 = 更准"曲线**。

- [ ] **P0-A** 液体核心加 `thinking_steps` 参数:同一核心循环 N 次做潜空间迭代;训练时随机采样深度(Geiping 式 depth randomization),推理时可变;N=1 严格等价现状(向后兼容)
- [ ] **P0-B** 合成推理基准 `benchmarks/reasoning_depth.py`:深度敏感任务(多步算术链 / 奇偶校验 / 多跳推理),transformer 对照组,多种子
- [ ] **P0-C** 训练 + 出图:思考步数 ∈ {1,2,4,8} vs 准确率曲线;若曲线单调上升 → thesis 成立,进 P1
- [ ] **P0-D** 生物模块清理:Hebbian 改 fast-weights 或删;GWT / PC 补 ablation(沿用实验日志的 5-seed 纪律)

**风险与止损**:若 8 步思考对准确率无增益(多种子),说明当前核心的递归不产生有效迭代计算 → 先修核心的状态更新算子(参考 TRM:递归时注入输入、状态残差连接),而不是加大规模。

### P1 — 蒸馏优先,不做预训练(数百美元云预算)

- 教师:开源强推理模型(Qwen3 系列等)生成推理轨迹;学生:350M~1B M1 架构
- SFT + logit 蒸馏(反向 KL),样本效率比预训练高一个数量级——穷人路线里唯一走得通的
- μP(maximal update parametrization)做缩放迁移:50M → 350M 先验证缩放曲线,**不盲跳 2B**
- 评测切换:弃 PPL,换 GSM8K / MATH-500 / ARC / RULER + 自有流式与持续学习基准

### P2 — 2B 混合架构(融资/算力到位后)

- **混合层配比**:纯 SSM 在精确检索上有已知短板(业界共识),液体核心为主 + 少量滑动窗口注意力层(Jamba/Zamba/Samba 证据),保近似 O(1) 同时补 recall
- 后训练:SFT → GRPO(可验证奖励:数学/代码)→ 长度控制
- 记忆控制器上线:surprise 门控读写 RAG,睡眠期蒸馏固化
- 端侧交付:ONNX / WebGPU 路径已验证(见 `benchmarks/export_o1_for_browser.py`)

---

## 4.5 P0 实验日志(诚实记录,含负结果)/ P0 Experiment Log

### 2026-07-28 · 第一轮:anytime 随机深度训练 → 两个负结果,均有明确解释

设置:2 层 MT-LNN(203K)vs ModernCausalTransformer(247K),3 seeds × 3000 步,
loss 仅在答案位;MT-LNN 训练时每步随机采样深度 1..8,评估深度 ∈ {1,2,4,8}。

| 任务 | 结果 | 诊断 |
|---|---|---|
| pointer_chase k=2/k=4(16 节点) | 双方 loss 均钉死在 ln(16)=2.77,acc=随机;**连 k=1 纯查表也学不会**(6000 步) | 不是深度问题、不是架构问题(transformer 同败)。每样本全新随机置换 → 无法记忆,必须学会上下文查表算法 → 处于 grokking 平台期。破法:更小图 / 更长训练 / 课程学习 |
| mod_chain k=8 | 双方部分学会(mt_lnn 0.26,transformer 0.29,随机 0.10);**mt_lnn 各深度完全同分** | **随机深度训练教会模型"无视迭代"**:深度不变解是最容易的最优化路径,反馈门保持 0。Geiping/HRM 均未用朴素随机深度——前者用截断反传+泊松采样,后者用逐迭代深监督 |

**决策**:P0 主声明改为 HRM 式 fixed-depth sweep——每个深度 d 训练一个全新模型
(参数量相同,权重绑定迭代),d 越大解得越好即命题成立。anytime(一套权重任意深度)
降级为 P0 之后的进阶目标,需要深监督/截断反传才有希望。
`benchmarks/reasoning_depth.py --mode fixed` 已实现。

### 2026-07-28 · 第二轮:fixed-depth sweep → 两个更硬的负结果

| 实验 | 结果 | 含义 |
|---|---|---|
| mod_chain k=8 fixed sweep(深度 1/2/4/8 各训一个全新模型,6000 步 × 3 seeds) | 0.289 / 0.297 / 0.293 / 0.293 —— **深度全平,差异在噪声内**;transformer 0.302 仍领先 | 即使固定深度训练+评估,只循环 LNN 子层也买不来能力 |
| pointer_chase 破平台探针(8 节点 k=2,12000 步) | **transformer 破平台:0.9932(基本解决,loss 0.07)**;MT-LNN depth1=0.2494、depth4=0.2498,loss 从 2000 步起钉死 2.0(已收敛,非训练不足) | 任务在此规模可学(transformer 证明);**MT-LNN 的混合栈没学会,且液体核心多迭代 4 次毫无帮助**。上下文关系查找的计算发生在注意力里;疑似 LNN 子层在阻碍注意力形成归纳头 |

**核心教训**:"只循环液体核心 ≠ 思考"。推理任务需要的组合查找由注意力承担,
把迭代范围限制在 LNN 子层是在错误的部件上加深度。

### 2026-07-28 · 第三轮:消融阶梯 → 两个嫌疑人都排除

同探针任务(8 节点 k=2,12000 步,transformer 对照 0.9932):

| 消融 | acc | 结论 |
|---|---|---|
| `--no_scan`(关液体递归,LNN→gated FFN) | 0.2889 | 液体递归**不是**主要阻碍(仅 +0.04) |
| `--n_layers 4`(注意力层翻倍) | 0.2488 | **不是**容量问题(零帮助) |

排除法指向 **MicrotubuleAttention 本身**。代码检查发现:GTP-cap 距离衰减偏置
`-γ_h·(i-j)` 对每头强制生效,4 头 γ init = [0.8, 0.2, 0.05, 0.0125](ALiBi 式
几何序列)——距离 25 处 3 个头的惩罚达 1.25~20 nats,只有 1 个头准全局。
归纳头需要两层注意力组合,可用远视头太少。另一嫌疑:GQA(n_kv_heads=2)。

### 2026-07-28 · 第四轮:GTP 衰减假说证实 + GQA 协同效应

同探针任务(8 节点 k=2,12000 步,seed 0):

| 变体 | loss@12k | acc | 结论 |
|---|---|---|---|
| 基线 MT-LNN | 2.0(钉死) | 0.249 | — |
| `--gamma_init 0.001`(全头全局) | 1.59(**仍在降**) | 0.360 | **GTP 衰减确认是阻断器** |
| `--full_mha`(仅关 GQA) | 2.02(钉死) | 0.248 | GQA 单独无罪 |
| γ + full MHA | **1.14(仍在降)** | **0.557** | **协同**:头能看远后 KV 多样性才起作用 |
| transformer 对照 | 0.07 | 0.993 | — |

**M2 架构原则 #1(P0 的第一个正面产出)**:生物启发的 GTP-cap 距离衰减把
4 头中 3 头的远视力干掉,而归纳头(上下文查表的基础回路)需要两层远视注意力
组合。修复不是删生物机制,而是 ALiBi 式**全局头配额**:每层至少 K 个头
γ≈0(真全局),其余头保留生物衰减。混合架构中"少量注意力层"必须是全局层。

### 2026-07-28 · 第五轮:结案 — γ+MHA 30k 步 **acc = 1.0000**

γ 全局化 + full MHA 的 MT-LNN 在 30k 步**完美解决**探针任务(1.0000),
反超 transformer 对照(0.9932)。**完全解释成立**:M1 的上下文关系推理
能力被 GTP 衰减 init(主因)+ GQA(协同)完全扼杀,修复后无残余劣势——
液体子层不拖后腿。(no_scan 对照收敛轨迹一致,确认液体核心无摩擦。)

**P0-C 结论**:
1. "只循环液体核心 = 思考"被证伪——组合查找的计算在注意力里
2. 真正的产出是**架构原则 #1(全局头配额)**,一行配置的修复,恢复推理满血
3. 思考深度命题需要在"注意力已修复"的模型上重新检验(P0-C′,待做):
   修好注意力后,更难的任务(更多跳数/更大图)上迭代深度是否开始起作用

### 2026-08-01 · 第六轮:复现危机 — 固定难度探针在 grokking 掷硬币区,单 seed 结论全体作废

| 数据点 | 配置(kv=2, 30k 步, 8 节点 k=2 固定难度) | acc |
|---|---|---|
| Kaggle seed 0 | g0(全衰减 init) | 1.0000 |
| **本地 seed 1 / 2(复现)** | 同上 | **0.1832 / 0.1652(≈随机)** |
| Kaggle seed 0 | g2(配额 2 全局头) | 0.179 |

**结论**:任务在 30k 步呈双峰(要么解出要么随机),单 seed 的"满分 vs 随机"
对比是 Bernoulli 抽样噪声。**第四、五轮的 γ/MHA 消融结论与 Kaggle 反例互相
不矛盾,但全部降级为"未裁决"**——它们测的很可能是"到达 grokking 的概率/时间"
而非能力有无。第 2 条(GTP 衰减=阻断器)相应降级为假设。

**新协议(裁决中)**:单环 + mix 课程任务(在 1/1 本地运行中于 28k 步可靠
grok 到 loss=0)+ 每配置 ≥3 seeds + per-k 评估。首批:Kaggle kernel
`m1-gqa-quota-replication-g0`(g0×3 seeds)+ 本地 g2。方法论教训入档:
**双峰任务上必须报告 grok 率(n seeds 中解出几个),禁止报告单 seed 准确率**。

## 5. 评测纪律 / Evaluation Discipline

- **不再以 PPL 为主指标**(打平已证,无增量信息)
- 主指标:GSM8K、MATH-500、ARC、RULER(长上下文)、自有流式记忆与任务链持续学习基准
- 一切声明 ≥ 5 seeds,报均值 ± 标准差;负结果照记(沿用 `BRAIN_INSPIRED_ROADMAP.md` 的诚实记录传统)
- 对外表述:只说四轴对标,注明对手模型的具体版本与 base/instruct 状态

## 6. 关联文档 / Related Docs

- `BRAIN_INSPIRED_ROADMAP.md` — 类脑机制实验日志(复测一~十二,EWC/重放/做梦数据)
- `BENCHMARKS.md` / `RESULTS.md` — 当前诚实基准
- `ABLATIONS.md` — 消融记录
- `docs/PRODUCT_LINES.md` — 产品线定位

---

### English Summary

**Thesis**: knowledge lives in RAG; the 2B model is a pure reasoning engine + learned memory controller. We target parity with 70B *base* models on four axes only: math/code reasoning, long-stream memory, continual learning, and edge latency/memory — never "overall parity."

**Why possible**: (1) narrow-domain reasoning can leapfrog via distillation + RL (R1-Distill-1.5B > GPT-4o on AIME); (2) recurrent depth trades compute for parameters (HRM 27M on ARC-AGI; latent recurrent-depth scaling) — and M1's liquid core is *natively recurrent*, an unused structural advantage; (3) offloading knowledge to retrieval frees parameters for reasoning circuitry.

**Bio-modules become verifiable mechanisms**: iterative latent reasoning (variable thinking steps), surprise-gated memory writes (predictive-coding error as the hippocampal write trigger), sparse global broadcast (k-winner GWT), and sleep-phase consolidation (offline replay distillation + EWC). Any module whose ablation gain is below seed noise gets cut.

**Phases**: P0 (now, local 8GB GPU) — prove the "think longer → more accurate" curve on depth-sensitive synthetic tasks; P1 (small cloud budget) — distillation-first at 350M–1B with μP scaling checks, no from-scratch pretraining; P2 (funded) — 2B hybrid (liquid core + sparse sliding-window attention), GRPO post-training, memory controller, edge delivery via ONNX/WebGPU.
