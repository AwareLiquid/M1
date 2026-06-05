# AwareLiquid Architecture (v2.0 — MT-LNN Next)

**Last updated:** 2026-06-06
**Status:** Active development — v2.0 module expansion in progress
**Companion docs:** [PRD.md](PRD.md) · [AWARENESS_NETWORK_PRD.md](AWARENESS_NETWORK_PRD.md) · [AWARELIQUID_SYSTEM_MVP.md](AWARELIQUID_SYSTEM_MVP.md)

---

## 1. Design Thesis

巨头在云端拼参数广度。我们在**推理体验的三个维度**做差异化:

| 维度 | Gemini 3.1 | AwareLiquid |
|---|---|---|
| 思考状态 | 每次 query 后蒸发 | 跨 session 持续演化 (capsule) |
| 计算分配 | 一刀切 thinking budget | 按语义熵 + Φ 信号动态路由 |
| 推理可信度 | 黑盒摘要 | 可回放、可归因的 Φ-trace |

知识广度不打，**用云端 API 当事实硬盘**，核心推理与状态留在端侧。

v2.0 在此基础上增加四个正交模块，每个都有独立开关，默认关闭，零回归风险。

---

## 2. 完整模块地图 (v2.0)

```
mt_lnn/                                       STATUS        TEST FILE
├── config.py              MTLNNConfig         ✅ 已实现
├── embedding.py           RoPE + TokenEmbed   ✅ 已实现
├── mt_attention.py        GQA 极性注意力       ✅ 已实现
├── mt_lnn_layer.py        13原丝 LTC + 侧向耦合 ✅ 已实现      test_model.py
│   ├── (内含) κ-gate      动态τ尺度门           ✅ 已实现
│   └── (内含) _hebb_signal Hebbian 信号采集     ✅ Phase D
├── rhythm.py              LAVI节律门控          ✅ 2026-06-06  test_rhythm.py
│   ├── LAVIEstimator      per-protofilament 节律检测  (13测试)
│   └── GlobalRhythmController 跨层节律校正
├── gwtb.py                GWT 压缩→SA→广播     ✅ 已实现
│   └── CompetitiveGWTBLayer 多源竞争广播       ✅ Phase A     test_gwt_competition.py
├── global_coherence.py    sparse top-k Orch-OR  ✅ 已实现
├── causality.py           因果一致性检测         ✅ Phase B     test_causality.py
├── world_model.py         预测隐态头             ✅ Phase C     test_world_model.py
├── plasticity.py          Hebbian 正则           ✅ Phase D     test_plasticity.py
├── deliberation.py        熵三级路由 + causal hook ✅ 已实现    test_deliberation.py
├── model.py               MTLNNModel (A-D集成)  ✅ 已实现      test_model.py
├── llama_adapter.py       HF 模型 MT 残差适配器  ✅ 已实现      test_llama_adapter.py
├── anesthesia.py          AVP 麻醉验证           ✅ 已实现
├── phi_hat.py             Φ̂ kNN 估算            ✅ 已实现
├── phi_iit.py             IIT 4.0 Φ (可选)      ✅ 已实现
├── memory.py              SQLite 持久 h_prev     ✅ 已实现      test_memory.py
├── capsule.py             信念状态 + 证据日志    ✅ 已实现      test_capsule_v2.py
├── streaming.py           state-only O(1) 推理   ✅ 已实现
├── recipes.py             Phase 5b 一键接入      ✅ 已实现
├── cloud_client.py        云端 Oracle 客户端     ✅ 已实现
├── observability.py       JSONL 指标写入         ✅ 已实现
├── reasoning_trace.py     推理时间线             ✅ 已实现
├── parallel_scan.py       pscan (Blelloch)       ✅ 已实现      test_parallel_scan.py
└── quantum_coupling.py    量子耦合 (可选)        ✅ 已实现
```

**Test coverage**: 195 tests, 195 pass (100%).

---

## 3. 层级架构 (five-layer view)

```mermaid
flowchart TB
    User([User Input]) --> L4

    subgraph L4["Layer 4 · Meta-Learning Plane (离线)"]
        ML["Capsule Clustering → Personal Prior<br/>memory.py · meta_learning.py"]
    end

    subgraph L3["Layer 3 · Verifiable Trace Plane"]
        PHI["Φ-IIT Monitor  phi_iit.py"]
        UI["Reasoning Timeline  trace_timeline.html"]
        PHI --> UI
    end

    subgraph L2["Layer 2 · Deliberation Router"]
        E1{"Token Entropy"}
        E2{"Semantic Entropy (N-sample)"}
        E3{"Fact Gap?"}
        E4{"Causal Consistency? ← Phase B"}
        E1 -->|low| OUT1[Local direct]
        E1 -->|mid| E2
        E2 -->|converge| OUT2[Local deep think]
        E2 -->|diverge| E3
        E4 -->|break| OUT2
        E3 -->|yes| CLOUD[Cloud Oracle]
        E3 -->|no| OUT3[Self-revise]
    end

    subgraph L1["Layer 1 · Stateful Reasoning Plane"]
        CAP["Capsule v2: belief_state + open_q + evidence_log"]
        STREAM["streaming_inference  O(1) state-only"]
        CAP <--> STREAM
    end

    subgraph L0["Layer 0 · Neural Backbone"]
        MT["MTLNNLayer: 13-filament LTC + κ-gate + LAVI rhythm"]
        CGWTB["CompetitiveGWTBLayer ← Phase A"]
        WM["PredictiveStateHead ← Phase C"]
        HEBB["HebbianRegularizer (train only) ← Phase D"]
        MT --> CGWTB --> WM
    end

    L4 -.feeds prior.-> L1
    User --> L1
    L1 --> L2
    L2 --> L3
    L0 -.serves.-> L1
    L0 -.serves.-> L2
    CLOUD -.quiet inject.-> CAP
    L3 --> Output([Answer + Φ-Trace])
```

---

## 4. Layer 0 — Neural Backbone 组件详解

### 4.1 已完成组件

#### 微管 LTC 层 (`mt_lnn_layer.py`)
- 13 protofilaments × 5 τ时间尺度，完全向量化并行
- LateralCoupling: 静态W_lat + 近邻耦合 + RMC内容感知
- MAPGate: per-protofilament 稳定性门控
- GTP水解周期性重置: 长上下文不丢失侧向混合

#### κ-gate 动态计算跳过 (`mt_lnn_layer.py:VectorizedMultiScaleResonance`)
- 基于**当前输入内容**的 τ 尺度权重
- 回答: "现在在处理什么" → 选择激活哪些时间尺度
- 稀疏模式 (sparse_resonance_kernel=True): top-k 尺度计算跳过

#### LAVI 节律门控 (`rhythm.py`) — 2026-06-06 完成
- 基于 **h_prev vs 输入相似度** 的节律检测
- 回答: "状态有多稳定" → 持续/瞬态模式切换
- 高 LAVI → 慢 τ 主导 (上下文维持)
- 低 LAVI → 快 τ 主导 (快速适应)
- GlobalRhythmController: 跨层节律聚合 + GWTB 前残差校正

```
κ-gate (内容信号) ⊕ LAVI (历史信号) = 完整稳定性-灵活性权衡
```

#### GWTB — 全局工作区瓶颈 (`gwtb.py`)
- 压缩 d_model→d_gw → 工作区SA → 广播回 d_model
- broadcast_gate 初始化为 0.01: 从 identity 出发渐进学习
- 实现 Baars/Dehaene GWT 的容量约束 (bottleneck = 意识瓶颈)

#### GlobalCoherenceLayer (`global_coherence.py`)
- Sparse top-k 因果注意力 (保留最高10%分数)
- Orch-OR 坍缩门: 基于注意力能量的二值化
- 可选衰减工作记忆 (use_decay_wm=True): O(1) 空间复杂度

---

### 4.2 Phase A — CompetitiveGWTBLayer 🔨 进行中

**生物原型**: GWT 的核心主张是"意识内容"通过**竞争**决定——多个专用处理模块(视觉、记忆、语言、推理)同时向全局工作区"投标"，只有赢家的表示被全局广播。

**现状缺口**: 当前 `GWTBLayer` 将单条 `x` 流直接压缩，没有多源竞争。

**实现方案**:

```
CompetitiveGWTBLayer
  ├── 继承 GWTBLayer (退化路径，module_bids=None 时完全兼容)
  ├── 新增: ScoreHead — 对每个 bid 打分 (B,T,d_model) → (B,T,1)
  ├── 竞争: softmax(scores) 加权融合 bids → z_combined
  └── 广播: 走原有 compress→SA→broadcast 流程

module_bids 来源 (通过 MTLNNModel.forward 可选注入):
  - lnn_out       : 微管 LTC 隐态 (已有)
  - attn_out      : 多头注意力输出 (已有)
  - coherence_out : 全局相干信号 (已有)
  未来可扩展: world_model_pred / causal_signal
```

**Config 开关**: `use_competitive_gwtb: bool = False`
**触碰现有测试**: 零（默认 False）
**新增测试**: `tests/test_gwt_competition.py` (8+ 测试)

---

### 4.3 Phase B — CausalConsistencyChecker 🔲 待实现

**生物原型**: 前额叶皮层对推理链进行持续的"预测误差"监测——当当前输出违背已建立的因果结构时，产生错误信号并触发重新评估。

**现状缺口**: `deliberation.py` 的路由只基于熵值，不检测生成内容是否与先前状态逻辑一致。

**注意**: 原方案建议接 Prolog/CaRing 推理引擎。**不采纳**。原因:
- Prolog 查询延迟 10-100ms，违反 I2 不变量 (<50ms/token)
- 触发频率 <1%，ROI 接近零
- 正确方案: 用已有的 h_prev 轨迹本身作为因果一致性信号

**实现方案**:

```python
# mt_lnn/causality.py
class CausalConsistencyChecker:
    """
    检测隐态轨迹中的因果断裂。
    
    原理: 正常因果链的 h_t 是 h_{t-1} 的平滑演化。
    断裂信号: 连续 window 步内 cosine_sim(h_t, h_{t-1}) < threshold
    
    接口: checker.update(h_prev) → consistency_score ∈ [0,1]
    集成: deliberation.py RouterDecision 增加 causal_consistency 字段
          consistency_score < 0.3 → 强制 SELF_CRITIQUE，无论熵值
    """
```

**deliberation.py 改动**: `RouteDecision` 新增可选字段，`decide()` 新增可选参数。完全向后兼容。

---

### 4.4 Phase C — PredictiveStateHead 🔲 待实现

**生物原型**: 预测编码理论 (Friston Free Energy Principle) — 大脑持续预测下一时刻的感知状态，以预测误差驱动学习。现有 `VectorizedMultiScaleResonance.use_predictive_coding` 只在**τ尺度之间**预测（慢τ预测快τ），缺少 token 级别的前向预测。

**实现方案**:

```python
# mt_lnn/world_model.py
class PredictiveStateHead(nn.Module):
    """
    给定 h_t，预测 h_{t+1} 的嵌入向量。
    
    损失: MSE(W_pred · h_t, embed(x_{t+1}).detach())
    权重: config.world_model_loss_weight = 0.01 (极小，不影响LM loss)
    
    推理时附加功能:
    - 预测误差 pred_error_t 作为 LAVI 的补充输入
      (模型自己对下一步的预测是否准确 → 更精准的节律感知)
    - pred_error 写入 last_pred_error buffer (监控用)
    """
```

**与 rhythm.py 的联动**: PredictiveStateHead 的误差信号可以增强 LAVIEstimator — 当预测误差突增，也标志着瞬态模式应该启动。

---

### 4.5 Phase D — HebbianRegularizer 🔲 待实现

**生物原型**: Hebb 法则 — "neurons that fire together, wire together"。激活模式相关的突触应当被巩固，减少灾难性遗忘。

**关于 STDP 的说明**: 原方案建议在 `mt_ltc_cell.py forward()` 里加 STDP。**不采纳**，原因:
- ProtofilamentLTC 的状态更新是连续时间衰减: `h_t = decay·h_{t-1} + (1-decay)·A_t`
- STDP 要求"突触前/后脉冲时序"，这在连续时间 LTC 里没有对应物
- 强行离散化会破坏 LTC 的核心动力学特性

**正确实现: Hebbian 损失项**（不改 forward，只改训练损失）:

```python
# mt_lnn/plasticity.py
class HebbianRegularizer(nn.Module):
    """
    Hebbian 正则项 (仅训练时启用):
    L_hebb = -α × mean(h_t ⊙ A_t)   # h=状态, A=输入投影, ⊙=逐元素乘
    
    当 h_t 与 A_t 同向时 (neurons fire together)，该项为负，
    加入总损失后鼓励这种共激活模式的权重被强化。
    
    LAVI 联动: α = base_lr × sigmoid(lavi_mean)
      - 持续模式 (高LAVI) → α 高 → 更强的记忆巩固
      - 瞬态模式 (低LAVI) → α 低 → 不在切换时过度巩固
    
    train.py 接入: total_loss += model.get_hebbian_loss()
    Config: use_hebbian=False (默认)，hebbian_lr=1e-4
    """
```

---

## 5. 数据流 (v2.0 增强版，单次 query)

```
User query
   ↓
[L1] prefill_state_only → 加载 capsule.belief_state (h_prev)
   ↓
[L0] per-token forward:
   MTLNNLayer (LTC + κ-gate + LAVI rhythm)
      ↓
   CompetitiveGWTBLayer (Phase A):
     bids = [lnn_out, attn_out, coherence_out]
     winner = softmax_competition(bids)
     broadcast(winner) → x
      ↓
   PredictiveStateHead (Phase C):
     pred_error = MSE(predict(h_t), h_{t+1})
     → LAVI 信号补充
   ↓
[L2] 每步 token → DeliberationRouter:
   ├─ CausalConsistencyChecker (Phase B):
   │    consistency = trajectory_smoothness(h_prev_window)
   │    if consistency < 0.3 → SELF_CRITIQUE
   ├─ 熵三级路由 (已有)
   └─ cloud inject (已有)
   ↓
[L3] 全程记录 (Φ, E, LAVI, causal_score, route_decision)
   ↓
[L1] capsule.save() ← belief + open_q + evidence
   ↓
Output: answer + reasoning timeline
```

---

## 6. 不变量 (Invariants)

| # | 不变量 | 守护机制 |
|---|---|---|
| I1 | Capsule ≤ 5KB | `capsule.py` 序列化大小断言 |
| I2 | 单 token 端侧延迟 < 50ms | L2 低熵路径不走 cloud；因果检测 <1ms (纯向量运算) |
| I3 | Φ 在生成全程可计算 | L3 与 L1 同步采样，不阻塞主 loop |
| I4 | 离线运行可用 | cloud router 失败必须 graceful degrade 到 self-critique |
| I5 | Evidence 可审计 | 每次 cloud inject 必须写 evidence_log |
| I6 | 新模块默认关闭 | 所有 Phase A-D 开关默认 False；零回归风险 |
| I7 | forward() 签名不变 | mt_lnn_layer.py forward / parallel_scan 不改签名 |

---

## 7. 耦合风险矩阵

```
                    config  mt_lnn_layer  gwtb    deliberation  model  train.py
─────────────────────────────────────────────────────────────────────────────────
Phase A GWT竞争      ✓(+2)              ✓扩展             ✓(+1)
Phase B 因果检测      ✓(+2)                      ✓(+1可选)  ✓(+1)
Phase C 预测头        ✓(+2)                                 ✓(+1)
Phase D Hebbian      ✓(+2)   ✓(读buffer)                         ✓(+1)
─────────────────────────────────────────────────────────────────────────────────
风险等级:             低      极低          低      极低        低     低
```

所有改动都通过**可选参数 + 默认 False**实现，不破坏任何现有测试路径。

---

## 8. 取舍记录 (v2.0 新增)

### 为什么不接 Prolog/CaRing 推理引擎?
Prolog 查询延迟 10-100ms。在 token 生成路径里，这违反 I2 不变量。且触发频率 <1%，维护一个独立推理引擎的 ROI 接近零。正确方案是用 h_prev 轨迹本身作为因果信号（Phase B）。

### 为什么不在 forward() 里加 STDP?
ProtofilamentLTC 是连续时间 ODE，没有离散脉冲事件。STDP 的数学前提（突触前/后脉冲时序）在这里不存在。强行加入意味着把连续动力学离散化——破坏 LTC 的核心特性。正确方案是 Hebbian 损失项（Phase D）。

### 为什么不重新组织目录结构 (core/gwt/causality/)?
现有 40+ 文件按功能组织在 `mt_lnn/` 下，有完整的 import 路径。重组只是移文件，没有架构价值，还会破坏现有所有 import。

### 为什么不做内在情绪/价值系统?
没有可验证的神经科学对应实现，没有可量化的工程验收标准。3年以上的事情不在当前 roadmap。

---

## 9. 实施时间线

| Phase | 模块 | 关键文件 | 工作量 |
|---|---|---|---|
| ✅ 已完成 | LAVI节律门控 | `rhythm.py` | 完成 |
| 🔨 A | CompetitiveGWTBLayer | `gwtb.py` (扩展) | 1-2 周 |
| 🔲 B | CausalConsistencyChecker | `causality.py` + `deliberation.py` | 2-3 周 |
| 🔲 C | PredictiveStateHead | `world_model.py` + `model.py` | 2-3 周 |
| 🔲 D | HebbianRegularizer | `plasticity.py` + `train.py` | 2-3 周 |
| 🔲 E | 完整 125M 预训练 + A-D 验证 | 全栈 | 6-12 月 |
