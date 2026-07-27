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
