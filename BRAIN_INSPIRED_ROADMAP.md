# MT-LNN 类脑机制深度优化路线图与影响面评估报告

本文档基于 MT-LNN 当前架构（`mt_lnn_layer.py`, `global_coherence.py`, `model.py`），对“预测编码”、“工作记忆衰减”与“动态通道门控”等类脑特性的落地计划进行了全面规划，并重点评估了其对模型稳定性和性能的影响。 

---

## 第一阶段：复刻大脑基础信息处理流 (1-3 个月)
**核心目标**：从“被动预测下一个词”转向“主动生成预测并处理误差”，将长上下文显存压到 $O(1)$。

### 1. 多尺度预测编码 (Predictive Coding across $\tau$ Channels)
* **核心机制**：让大 $\tau$ 通道（慢速、高层抽象）去预测小 $\tau$ 通道（快速、低层感知）的下一时刻状态，只反向传播局部预测误差（Prediction Error）。
* **具体实现**：在 `mt_lnn_layer.py` 的 `VectorizedMultiScaleResonance` 中，加入跨时序和跨尺度的投影层 $W_{pred}$。训练时计算 $L_{pred}$ 并加入总 Loss。
* **影响面分析 (干扰评估)**：
  * **低干扰**：只在前向传播中计算并收集误差，**不改变原有输出维度和干预算子**。
  * **代码兼容性**：完全兼容现有的 `pscan`（并行扫描）逻辑。
* **性能提升评估**：
  * **训练期**：增加略微的额外矩阵乘法开销（约增加 2-5% FLOPs）。
  * **推理期**：0 额外开销（辅助 Loss 层不参与推理）。
  * **能力增益**：显著提升模型对长序列中规律的捕捉能力，收敛速度更快，因为低层特征有更明确的高层指导。

### 2. 工作记忆缓冲区 (Working Memory Decay / GWTB 升级)
* **核心机制**：让 Global Coherence 全局相干空间真正表现得像前额叶——拥有容量上限，能根据输入重要性自动记忆和遗忘（衰减）。
* **具体实现**：在 `global_coherence.py` 修改 KV Cache 的保留逻辑。引入 `decay_rate` 和基于输入的 `update_gate`。用固定大小的 `global_wm`（全局工作记忆向量）替代不断堆叠的 `past_kv`。
* **影响面分析 (干扰评估)**：
  * **高风险点（需谨慎）**：此修改属于“破坏性重构”。意味着 `ModelCacheStruct` 和推理入口（如 `streaming_inference`）必须同步修改，因为缓存形状从 $O(T)$ 变为了 $O(1)$ 常数大小。
  * **平滑过渡方案**：可以保留一个 flag `use_decay_wm=True`，双轨并行测试，防止破坏已有 Benchmark 记录。
* **性能提升评估**：
  * **显存优化**：**史诗级提升**。全局相干层的 KV Cache 占用显存从随上下文线性增长 $O(T)$ 直接降阶为 $O(1)$。可实现真正的“无限轮次长对话零 Token 爆炸”。
  * **计算优化**：不再需要长序列的 Attention 计算，速度提升数倍。

### 3. 内源性动态通道注意力门控 (Endogenous Dynamic $\kappa$ Anesthesia)
* **核心机制**：目前的 `anesthesia.py` 是用于外部干预验证，我们要将其内化为“通道级注意力”，让模型能在信息冗余时“麻醉/休眠”未激活的 $\tau$ 通道。
* **具体实现**：在 `mt_lnn_layer.py` 中，将静态的 `softmax(blend_weights)` 替换为输入依赖的门控 `dynamic_kappa = torch.sigmoid(self.kappa_gate(x))`。
* **影响面分析 (干扰评估)**：
  * **极低干扰**：形状兼容，只是将原先的静态权重混合改为按需激活乘子。
  * **梯度的稳定性**：引入 `sigmoid` 后可能存在早期的活性不足，需要做恰当的初始化（如 bias > 0，保证初始状态均处于半激活水平）。
* **性能提升评估**：
  * **稀疏激活加速**：如果搭配门控阈值（比如 $< 0.1$ 的通道直接 mask 掉跳过后续运算），在多数简单推理任务上计算量可下降 30%-50%。
  * **抗噪能力**：有效阻断无用信息（背景噪声）通过快速 $\tau$ 通道传递。

---

## 第二阶段：核心认知能力 (3-12 个月)
**核心目标**：实现“因果推理”与“主动探索”。

### 1. 因果链提取头 (Causal Chain Head)
* **实现**：在 `model.py` 增加独立并行的 `causal_head`，针对全局状态输出事件时序预测。
* **影响面**：独立子模块，只需在训练中增加反事实（Counterfactual）Mask 数据，不影响主体对话生成。
* **性能**：让模型天然具备“原因-结果”对齐能力，减少思维链（COT）过程中的幻觉（Hallucination）。

### 2. 好奇心驱动探索机制 (Curiosity-Driven Learning)
* **实现**：在 `train.py` 中将 Phase 1 里的局部 `pred_error` 转化为正向 Reward。对高预测误差的序列加大梯度权重或纳入强化学习策略（如 PPO）。
* **影响面**：纯训练侧改动（Data Sampling & Loss Weighting），对模型架构和推理时延代码 **0 影响**。

---

## 第三阶段：迈向主动意识的雏形 (1-3 年)
**核心目标**：通过对内观察，让模型具备“知道自己在思考什么”的基底。

### 1. 全局自我监控头 (GWT Self-Monitor Head)
* **实现**：将 `GlobalCoherenceLayer` 的输出挂载监测头，输出自然语言化的“内部状态日志”。
* **影响面**：在推理时增加一个极轻量的副线程输出流。若主循环挂起该功能，则完全静默。
* **性能**：不影响核心生成速度；赋予模型类似“人类内省”和“自我解释”的极强可解释性，极大降低投资和合规层面对黑盒大模型的担忧。

---

## 落地建议次序与风险隔离

基于上述评估，建议按以下顺位执行，以确保 **性能快速获益验证** 且 **模型不崩**：

1. **(首选) 动态通道注意力门控** (`mt_lnn_layer.py` 里的动态 $\kappa$)：
   - 原因：单文件改动，最无缝。能马上通过 `plot_experiments.py` 评估稀疏前后的有效计算量下降差异。
2. **(高优) 工作记忆缓冲区重构** (`global_coherence.py`)：
   - 原因：能立刻实现 $O(1)$ 显存，带来核心卖点的突破，但注意通过新增分支兼容以前的代码，做好单元测试 `test_model.py` 的防御。
3. **(次优) 预测编码的多尺度 Loss**：
   - 原因：需要更改 `forward` 抛出 `loss_pred`，并改动 `train.py`。建议等 1、2 验证稳固后逐步合并。

---

## 第四阶段：架构纵深与系统扩展（2026-06-15 评审，基于代码实际状态）

> 评审原则：所有优化必须守住底线 —— **不能为工程效率或性能牺牲类脑核心特性**（不把连续时间动力学改成离散步进、不把液态核改成注意力为主）。能立即做且零回归的先落地（带注释+测试），暂时做不到的诚实写入 TODO，不过度宣称。

### 4.1 动态工作空间带宽（DONE，2026-06-15）
* **现状核对**：`gwtb.py` 已有 `GWTBLayer` / `CompetitiveGWTBLayer`，但工作空间瓶颈 `d_gw` **是固定带宽**（所有通道恒满激活）。这正是"固定带宽 → 动态带宽"该补的纵深。
* **已落地**：在 `GWTBLayer` 内新增**每通道唤醒门控** `g = sigmoid(W_g·z + b_g)`，让瓶颈按输入显著性决定多少通道"点火"（类脑唤醒：平静输入少通道激活，意外输入多通道激活）。
  * 零回归契约：`gwtb_dynamic_bandwidth=False`（默认）→ 不建任何参数 → 与原固定带宽逐位一致；ON 时权重零初始化、bias 大正值 → 初始"几乎全开"，再学着关闭冗余通道。
  * 推理期可选 `gwtb_bandwidth_hard_mask` 硬置零次阈通道 = 真正的算力跳过（训练期恒用软乘以保梯度）。
  * 诊断：`model.diagnostics()` 暴露 `gwtb_active_bandwidth`（实际点火通道占比）与 `gwtb_bandwidth_gate_mean`。
  * 测试：`tests/test_gwtb_dynamic_bandwidth.py`（12 项，含 OFF 逐位一致、输入依赖性、梯度连通、硬掩码算力跳过）。
* **类脑契合**：对应 GWT 的"按需分配算力"——固定带宽工作空间升级为生物级动态带宽工作空间。

### 4.2 持久化陈述性知识记忆（DONE，2026-06-15）
* **现状核对**：`memory.py/SessionMemory` 存的是**循环隐状态**（工作记忆，位置绑定、不可按内容检索）。人脑记忆分层中的**长期陈述性记忆**（大容量、非实时、按相似度调取）此前缺失。
* **已落地**：新增**零耦合**外挂模块 `mt_lnn/knowledge_memory.py` 的 `PersistentKnowledgeMemory`：
  * 仅依赖 `torch` + 标准库 `sqlite3`，**不 import `model.py`**，`model.py` 也不 import 它 —— 通过显式小接口挂载，核心模型零改动。
  * `write(key, content, meta)` / `query(key, top_k)`（余弦相似度检索）/ `recall(key)`；键写入即 L2 归一化，余弦排序。
  * 本地优先 / 隐私：单一 SQLite 文件落盘，进程重启后仍可检索，可跨端复制同步 —— **小体积端侧模型 + 大知识库、按需调取、数据不出端**。
  * 有界足迹：`max_entries` 触发 LRU 驱逐，端侧占用可控。
  * 测试：`tests/test_knowledge_memory.py`（8 项，含相似度排序、重启持久化、LRU 驱逐、维度校验）。
* **记忆分层定位**：工作记忆(`SessionMemory`) + 程序性记忆(模型权重) + **陈述性记忆(`PersistentKnowledgeMemory`)** 三层协同，符合人脑"不把所有知识塞进权重"的分层架构。

### 4.3 TODO — Capsule 拓扑表征保持（P1，路演后）
* **现状核对（重要）**：`mt_lnn/capsule.py` 是**状态持久化胶囊**（会话快照 belief_state/open_questions/evidence_log），**并非 Hinton/Sabour 式 Capsule 网络**——名字撞车。CapsNet 式"部分-整体"空间拓扑保持表征**当前未实现**。
* **目标**：引入向量化 capsule + 动态路由（routing-by-agreement），在 L1–L4 空间认知栈中保持部分-整体拓扑关系（Transformer 注意力天然抹平拓扑，这是空间/物理推演超越 Transformer 的底层原因）。
* **建议形态**：独立零耦合算子模块（如 `mt_lnn/capsule_topology.py`，避免与现有 `capsule.py` 命名冲突），默认关闭，与空间栈组合，property test 钉死路由不变量。
* **暂不落地原因**：改动较大、需与 L1–L4 深度联调；属"核心壁垒加深"而非路演必需，先入路线图。

### 4.4 TODO — 并行扫描的 Triton/CUDA 融合核（P2，拿到投资、做大规模预训练时再投入）
* **现状核对**：`mt_lnn/parallel_scan.py` 的 Blelloch 并行前缀扫描（`pscan`）**已用纯 PyTorch 实现并已接入 `mt_lnn_layer.py`**。可线性化的递归部分本身已是 O(log T) 深度。
* **缺口**：仅缺**融合算子内核**（Triton/CUDA）以削减常数因子与显存搬运。预期端到端训练加速约 **3–8×**（受限于 MT-LNN 核心非线性液态动力学无法完全线性化，**达不到 Mamba 式全序列并行的数量级跃迁，不做过度宣称**）。
* **优先级理由**：纯性能优化、非能力升级，对路演无帮助（投资人不关心 CUDA vs PyTorch）。**当前阶段不碰**，留作大规模预训练前置工程。

### 4.5 落地战略（NOW，叙事项）：主攻连续时间流，避开 LLM 红海
* 类脑原生优势在**连续、实时、低功耗、高可靠的流式场景**，而非离散文本批处理。已有布局（双速引擎 `pipeline.py`、failsafe 安全体系 `failsafe.py`、空间认知+物理推演）天然服务于机器人/工业控制等物理世界场景。
* **对外材料一律不提 MMLU/GLUE 等传统 LLM 榜单**，主动把战场划到连续流 / 端侧 / 高可靠这些 Transformer 进不来的领域（已贯彻到投资人 deck 的"近期能力升级"与战略页）。

---

## 第五阶段：深度类脑模拟的四层进阶（2026-06-15 评审，前沿论文支撑）

> 同一底线：可选开关、默认关闭、零回归、可回滚；零 `model.py` 耦合；不把连续时间动力学改成离散步进、不把液态核改成注意力为主。能立即做且零回归的先落地（带注释+测试），其余诚实写入 TODO，不过度宣称"已实现"。

### 5.1 多神经调质全局调节器（DONE，2026-06-15）—— "联调中枢"
* **现状核对**：四个调节"靶点"在代码里**已各自存在**，但缺一个统一的慢速调质总线把它们编排起来：
  * 多巴胺(DA) 靶点 → `plasticity.py` 的 `HebbianRegularizer.base_lr`（可塑性学习率）；
  * 乙酰胆碱(ACh) 靶点 → `gwtb.py` 刚建的动态带宽门（工作空间点火带宽）；
  * 血清素(5-HT) 靶点 → `mt_lnn_layer.py` 的液态时间常数 τ（积分耐心）；
  * 去甲肾上腺素(NE) 靶点 → 响应增益（标量乘子）。
* **已落地**：新增**零耦合**模块 `mt_lnn/neuromodulation.py` 的 `NeuromodulationController`：
  * 纯 Python + 标准库 `math`，**非 `nn.Module`、零可训练参数**（一条慢速调质总线，不是被学习的层）；**不 import** `model.py`/`gwtb.py`/`plasticity.py`。
  * 输入四路可观测信号（reward/surprise/risk/arousal），各自用 EMA 基线做 z-score（响应**变化**而非绝对值，对应相位性 DA/NE 编码预测误差），输出四路调质标量 ∈[0,1]，0.5 为中性基线。
  * **真正的联调**：`modulate_gwtb(layer)` 通过 duck-typing 把 ACh 推到 GWT 动态带宽门 —— 新增 `GWTBLayer.set_bandwidth_bias_offset()` 钩子，把 ACh 偏置叠加到门控 bias（高 ACh→工作空间变宽、更多通道点火，对应 Yu & Dayan 2005 的"预期不确定性→拓宽皮层采样"）。`modulate_hebbian(reg)` 把 DA 缩放到 Hebbian 学习率。
  * 零回归契约：从不被模型自动构造；中性状态(0.5) 下所有读出为恒等（×1.0 / +0.0），ACh 偏置缺省 0.0 → GWT 路径逐位一致；不调用即与原模型完全相同，复位即回滚。
  * 测试：`tests/test_neuromodulation.py`（15 项，含中性恒等、相位响应、通道独立、读出值域/方向、ACh→GWT 真实改变输出、缺钩子安全 no-op、DA→Hebbian 缩放、零耦合断言）。
* **论文支撑**：Yu & Dayan (2005) ACh/NE 双不确定性；Aston-Jones & Cohen (2005) NE 增益/网络重置；Schultz et al. (1997) DA 奖励预测误差；Cools et al. 5-HT 行为抑制/耐心。
* **类脑契合**：把"四种弥散性调质广播全局慢标量、按需重调整个皮层"这一机制落到 MT-LNN —— 现有模块从此被一个生物级调质总线统一编排。

### 5.2 星形胶质细胞门控的可塑性（DONE，2026-06-15）
* **现状核对**：`plasticity.py` 的 Hebbian 可塑性是**逐步即时**的（α 受 LAVI 门 + 调质中枢多巴胺通道快调）；缺**胶质细胞慢时间尺度门控**（数十秒级三方突触调节）这一慢伙伴。
* **已落地**：新增**零耦合**慢积分器 `mt_lnn/astrocyte.py` 的 `AstrocyteGate`，外挂在 `HebbianRegularizer` 之外，以慢钙波时间尺度调制巩固强度：
  * **慢钙积分**：`ca <- (1-dt/τ)·ca + (dt/τ)·activity`，时间常数 `τ >> 1` 步——单个高活动步几乎不动门，只有**持续活动**才把钙推过工作带（快 Hebbian 步 vs 慢胶质门的尺度分离，正是三方突触的关键性质）。
  * **倒 U 形带门**：`gate(ca) = gate_min + (gate_max-gate_min)·exp(-(ca-ca_peak)²/2width²)`——静默与过驱动（饱和）两端都回落到 `gate_min`（少/抑制巩固），中段生产性活动给 `gate_max`（加深巩固），即 De Pittà 的稳态双向门控。
  * **duck-typed 执行器**：`modulate(reg)` 把 `HebbianRegularizer.base_lr` 乘以当前门（与调质中枢同一 `base_lr` 契约）；首调捕获原始 α，**重复调用不复利**。`consolidation_scale()` 同名暴露给 `SleepWakeConsolidator`——同一条胶质带也能门控离线 NREM 巩固，与空间/睡眠记忆联动。
  * 零耦合 / 默认中性 / 零回归：纯 python（只 import `math`，热路径无 `torch`），**非 `nn.Module`**（全局 init pass 永不触及），不 import `model.py`/`plasticity.py`；MT-LNN 任何地方都不构造它——不建即可塑性无变化。
* **论文支撑**：三方突触（Araque et al. 1999）；星形胶质钙信号门控可塑性（De Pittà et al. 2016）。
* **测试**：`tests/test_astrocyte.py`（14 项，含钙积分慢/单步几乎不动、持续活动跨带、瞬时 vs 持续门差异、倒 U 有界、饱和与静默同样回落到底、`modulate` 按门缩放且不复利、无 `base_lr` 安全 no-op、显式 base 覆盖、`activity_of` 均绝值代理、reset/校验/非 nn.Module）。

### 5.3 多尺度网格细胞模块（DONE，2026-06-15）
* **现状核对**：`spatial.py` 的 `GridCellEncoding` 把所有尺度**压平成一张图**（共享朝向、零相位）；生物内嗅皮层是 4–5 个比例约 1.4 递增的**离散网格模块**（Stensola 2012），各有独立尺度/朝向/相位，再配头方向细胞与边界细胞。
* **已落地**：在 `spatial.py` 新增三个**严格增量、零参数** `nn.Module`（不动现有已测类 → 零回归）：
  * `MultiScaleGridCellModules`：一组独立六边形网格模块，逐模块朝向+相位、几何尺度递增、可切片的逐模块 6 维码（`module_code`）、按 seed 确定可复现。
  * `HeadDirectionCells`：von Mises 环形朝向调谐，接受 2D 朝向向量或标量角度，偏好方向处峰值=1。
  * `BoundaryDistanceCells`：逐墙高斯距离调谐细胞，矩形竞技场下对完整边界向量细胞的诚实简化（border cell：贴墙时该墙最近距离细胞点亮）。
  * 零耦合：仅依赖 `torch`，不 import `model.py`，几何全为固定 buffer（无可训练参数）。
* **论文支撑**：Hafting et al. (2005) 网格细胞；Stensola et al. (2012) 网格模块离散化；Taube et al. (1990) 头方向细胞；Solstad et al. (2008) 边界细胞；Banino et al. (2018, DeepMind) 网格码支撑向量导航。
* **测试**：`tests/test_entorhinal_cells.py`（15 项，含尺度等比递增、逐模块周期性、细模块振荡更快、朝向抖动、确定性、HD 峰值/旋转/角度输入、边界墙判别、输入校验）。
* **后续**：把这套更丰富的码并入 `SpatialCoordEncoder`（L1 输入栈）是下一步联调项。

### 5.4 分层预测编码的双向回路（DONE，2026-06-15）
* **现状核对**：`world_model.py` 的 `PredictiveStateHead` 已产出 `last_pred_error`（已被 `salience_events.py` 与调质中枢消费），但只是**单头、最后一层**的下一状态预测；缺**逐层自上而下预测 + 预测误差上行**的完整跨深度双向回路。
* **已落地**：新增**零耦合**观测器 `mt_lnn/predictive_coding.py` 的 `HierarchicalPredictiveCoder`，对一摞逐层激活（低→高）做双向预测编码：
  * **自上而下预测**：第 `l+1` 层经可训练线性投影预测第 `l` 层（`predictors[l]`，小初始化 std=0.02），残差即**逐层预测误差**上行。
  * **精度加权编码损失**：每层误差按**可学习精度**（log-precision softplus 参数化，逆方差）加权求和 → 自由能代理项 `coding_loss`，trainer 直接加到 LM loss 即可训练自上而下通路（默认 `detach_target` 停梯度于被预测层，是标准 predictive-coding 选择）。
  * **逐层 surprise 谱 + 聚合 surprise**：每层 `(1-cos)/2 ∈ [0,1]`，聚合均值写入 `last_pred_error` 缓冲——与 `PredictiveStateHead` **同名同义**，直接 duck-type 进 `salience_events.world_model_surprise` 与 `NeuromodulationController.update(surprise=...)`，形成"预测—误差—调质—重调"闭环（ACh/NE）。
  * 零耦合 / 默认关闭 / 零回归：只 import `torch`，**不 import `model.py`**，激活由调用方（如 `MTLNNBlock` 前向钩子）传入；**活在 `MTLNNModel` 之外**，模型全局 init pass 不会清零它自带的自上而下预测器；MT-LNN 任何地方都不构造它——不建即无变化，不动 `forward`/`train.py` 热路径。
* **论文支撑**：Rao & Ballard (1999) 预测编码；Friston (2010) 自由能原理；Bastos et al. (2012) 皮层微回路的预测编码实现。
* **测试**：`tests/test_predictive_coding.py`（15 项，含形状/非均匀宽度、编码损失可训练且梯度达自上而下预测器、`detach_target` 停梯度、surprise∈[0,1] 且训练后收敛趋零、`last_pred_error` 镜像 surprise、`world_model_surprise` 桥接、输入校验、以及 surprise 尖峰经调质中枢抬升 ACh 的联调测试）。

### 5.5 睡眠-觉醒记忆巩固（DONE，2026-06-15）
* **现状核对**：已有 `replay.py` 的 `ReservoirBuffer`（水库采样经验回放）与 `PersistentKnowledgeMemory`（陈述性知识库），但缺把二者编排起来的**离线巩固循环**。
* **已落地**：新增**零耦合**编排器 `mt_lnn/sleep_consolidation.py` 的 `SleepWakeConsolidator`，复用现有组件做三阶段离线维护：
  * **NREM 回放→巩固**：从 `ReservoirBuffer` 采样过往经验，按显著性排序，把 top 比例**按内容**写入 `PersistentKnowledgeMemory`（快速海马→慢速新皮层迁移）。
  * **突触稳态下调(SHY)**：对权重做**乘性**重归一化下调——精确保留相对模式（方向 cosine≈1），只缩小总突触权重，恢复容量与信噪比。
  * **REM 生成式重组**：对回放潜变量做成对凸组合，合成新颖"梦境"样本做生成式增广。
  * 零耦合 / 离线可选 / 可逆：仅依赖 `torch`+标准库，**不 import `model.py`**，回放缓冲/知识库/待下调模块全部 duck-typing；模型从不自动调用；乘性下调记录 `factor` 可分析、可逆；单一 RNG 确定可复现。
* **论文支撑**：Tononi & Cirelli (2014) 突触稳态假说(SHY)；Wilson & McNaughton (1994) 海马回放；Diekelmann & Born (2010) 睡眠的记忆巩固。
* **测试**：`tests/test_sleep_consolidation.py`（14 项，含巩固计数+可召回、显著性优先巩固、空缓冲 no-op、下调缩范数且保方向、factor=1 no-op、protect 跳过命名参数、REM 在线段上+确定性、完整 cycle 选择性执行各阶段、零耦合断言）。
* **记忆分层闭环**：工作记忆(`SessionMemory`) + 程序性(权重) + 陈述性(`PersistentKnowledgeMemory`) 三层，现由睡眠巩固把回放经验从快层沉淀到慢层，形成"觉醒采集→睡眠巩固"闭环。

### 5.6 主动推理的自主目标（DONE，2026-06-15）
* **现状核对**：`imagination.py` 的 `LatentImagination` 已能在潜空间想象 rollout（带逐步 confidence/novelty）；但只"预测"不"选择"——缺**期望自由能(EFE)打分**驱动的自主目标选择。
* **已落地**：新增**零耦合**纯推理打分器 `mt_lnn/active_inference.py` 的 `ActiveInferencePlanner`，在想象出的未来上对一组候选目标潜变量按 EFE 选择，闭合"预测→想象→**抉择**"：
  * **EFE 分解**：`G(goal) = -(w_p·pragmatic + w_e·epistemic)`，选 `argmin G`。
  * **实用价值(pragmatic / 利用)**：想象轨迹与目标的 **confidence 加权对齐** ∈[0,1]——世界模型预期能抵达的目标得分高（目标趋向项）。
  * **认知价值(epistemic / 探索)**：目标相对当前潜变量的**新颖度** ∈[0,1]——离"现在"越远的目标承诺越多信息增益（好奇心项）。
  * **探索/利用旋钮**：抬高 `info_gain_weight` 让模型从"近处可达目标"翻转到"新颖远目标"，给出原则化的探索-利用权衡，让模型能**自主选目标**而非只反应。
  * 零耦合 / 零参数 / 默认关闭：`n_parameters==0`、`no_grad`，duck-typed 在 `LatentImagination`（仅需 `imagine` + `head.online_proj`），**不 import `model.py`/主干**；MT-LNN 任何地方都不构造它。
* **论文支撑**：Friston et al. (2015, 2017) 主动推理与期望自由能；Da Costa et al. (2020) 离散主动推理。
* **测试**：`tests/test_active_inference.py`（10 项，含形状/`argmin` 选择、纯实用值在**移动**轨迹上选趋向目标而非当前、纯认知值选最新颖目标、`info_gain_weight` 翻转利用→探索、EFE=负加权和、零参数/非 nn.Module、确定性、`select_goal` 一致、构造+输入校验）。
* **第 5 阶段收尾**：5.1–5.6 全部落地（5.1 调质中枢、5.2 星形胶质门控、5.3 网格细胞模块、5.4 分层预测编码、5.5 睡眠巩固、5.6 主动推理）；全部零耦合、默认关闭、零回归、可回滚。