# MT-LNN v2.0 实施审查报告

**日期**: 2026-06-07  
**审查范围**: Phase A-D 实施的诚实评估 + 死代码 + 技术债务 + 修复路线

---

## 0. TL;DR

四个模块**都做得很浅**。它们在 `tests/` 里通过是因为测试只验证"形状/梯度/no-crash"，不验证"机制是否真的工作"。

`examples/v2_demo.py` 跑 10 步训练后的实证数据（这是诚实的证据）：

| 模块 | 期望表现 | 实际表现 | 诊断 |
|---|---|---|---|
| Phase A 竞争 | bid 权重分化，winner 浮现 | `(0.333, 0.333, 0.333)` 永远均匀 | 对称性破坏失败 |
| Phase B 一致性 | 主题切换时下降 | 0.998-1.000 永远满分 | 信号灵敏度死光 |
| Phase C 世界模型 | pred_error 随训练下降 | 0.99 → 0.97（几乎不动） | MSE 鼓励崩塌解 |
| Phase D Hebbian | 推动神经元共激活 | signal ≈ 0.02, lr=5e-5 → 梯度贡献 1e-6 | 信号几乎不存在 |

**统一诊断（来自外部研究核查）**:

> 所有四个模块**给了"信号"但没有给"必需的非对称性机制"**——
> noise injection / stop-gradient / EMA target / contrastive pair / orthogonality penalty。
> 缺少这些破对称性的力量，损失项虽然在数学上存在，但梯度方向是平的，模型学不到任何有用的分化。

---

## 1. 逐模块审查（基于研究检索）

### Phase A — CompetitiveGWTBLayer

**当前实现**：K 个 `BidProjector`（残差 MLP，初始 `delta=0`）→ 共享 ScoreHead（输出层零初始）→ softmax 加权和。

**为什么不工作（数学推导）**：
- 初始：所有 bid_k = x（identity），所有 score_k = 0，softmax 给出 1/K
- 反向传播：`d_loss/d_bid_k = (1/K) · d_loss/d_combined`，对所有 k 相同
- score 的梯度：`d_loss/d_score_k ∝ (d_loss/d_combined · bid_k) - Σⱼ wⱼ (d_loss/d_combined · bidⱼ)`
- 但所有 bid 相同 → 括号里两项相等 → **score 的梯度严格为零**
- 结果：bid 永远不分化，competition entropy 永远等于 `log(K)`

这是经典的 **routing collapse / pseudo-balance 失败**（DeepSeekMoE 论文明确点名的失败模式）。

**SoTA 修复方案**（来自 [DeepSeekMoE](https://arxiv.org/abs/2401.06066) 和 [Advancing Expert Specialization 2025](https://arxiv.org/html/2505.22323v5)）：

1. **共享 + 路由分裂**：固定一个 bid 总是参与广播（吸收公共信号），剩下的 bid 被迫在残差上专精
2. **正交性损失**：在 bid_projector 的输出投影矩阵之间施加 `||Wᵢᵀ Wⱼ||²` 惩罚
3. **Noisy top-k**：训练时在 score 上加高斯噪声 → 即使 bid 相同，winner 也会被随机区分 → 梯度流经不同路径 → 自然破对称
4. **Router z-loss**：`α · log²(softmax_denom)`，防止 logits 饱和

**最小可行修复**：方案 3（noisy top-k）是 2 行代码——只需要在训练时给 `scores` 加 `torch.randn_like(scores) * noise_scale`。

---

### Phase B — CausalConsistencyChecker

**当前实现**：cosine_similarity(current_h, window_mean_of_recent_h)。

**为什么不工作**：
- LLM 隐态有严重的**各向异性（anisotropy）**——baseline cosine sim 在 0.9+
- 动态范围被压扁，区分"正常 0.95"和"异常 0.85"几乎不可能
- 窗口均值本身就是平滑的 → 信号天然滞后且饱和
- demo 数据：所有 12 步的一致性都在 0.998-1.000，主题切换后**没有任何下降**

**SoTA 信号**（来自 [Farquhar/Kuhn Nature 2024](https://www.nature.com/articles/s41586-024-07421-0) 和 2025 后续）：

1. **Semantic entropy**：在 N 个采样输出上做 NLI-equivalence 聚类，测量*语义层*分歧（不是 token 层）
2. **Semantic Entropy Probes**：在**中间层**隐态上训练的线性探针（不是末层！末层被 LM head 压扁了）→ ~3M 参数即可逼近完整 SE
3. **Effective rank / spectral signals**：滑动窗口内隐态协方差的有效秩——模型"跑飞"进入幻觉时，秩会塌缩 → 远比 cosine 敏感

**结构性问题**：cosine-to-mean 这个方向本身就是死路。要修就要换信号源（不是调参数）。

---

### Phase C — PredictiveStateHead

**当前实现**：2-层 MLP，预测 h_{t+1}，MSE 损失，target 是 `x[:, 1:, :].detach()`。

**为什么不工作**：
- MSE 在多模态分布上是错误的——它鼓励**平均预测**
- 预测 detach 后的自己（同一个 representation space）→ **崩塌解**：`predictor(h) → mean(h)` 已经是 loss 局部最优
- demo 数据：world_model_loss 从 0.99 微动到 0.97——基本没学到东西

**SoTA 路径**（来自 [V-JEPA 2024](https://arxiv.org/abs/2404.08471) 和 [TWM-CPC 2025](https://arxiv.org/html/2503.04416v2)）：

1. **EMA target encoder**：用一个 EMA 副本（不是自己）作为预测目标 → 破对称性，防止崩塌
2. **InfoNCE / contrastive 损失**：用负样本（其他位置的 h）作为反例，预测目标必须 vs 反例**可区分**
3. **分层时间尺度**：[Rao 2024 Active Predictive Coding](https://direct.mit.edu/neco/article/36/1/1/118264) 在多个时间粗细度上做预测，单一 MLP 缺少这种抽象

**最小可行升级**：MSE → InfoNCE。代码量约 30 行，需要管理一个 EMA encoder。

---

### Phase D — HebbianRegularizer

**当前实现**：`L_hebb = -α · mean(out ⊙ x_in)`。

**为什么不工作**：
- 未训练网络上 `out` 和 `x_in` 是近独立的随机投影 → `mean(out ⊙ x_in) ≈ 0`
- demo 数据：hebb signal ≈ 0.02，α = 5e-5 → 总损失贡献约 1e-6（vanishing gradient）
- 单个标量信号无法定向加强"哪些权重"

**SoTA 路径**（来自 [Forward-Forward Hinton 2022](https://arxiv.org/abs/2212.13345) 和 [Self-Contrastive FF 2025](https://www.nature.com/articles/s41467-025-61037-0)）：

1. **Oja's rule**（不是原始 Hebb）：`ΔW = η · y · (x - y·W)` —— 减去 decay 项防止权重爆炸，**且**让信号非平凡
2. **中心化协方差**形式：`L = -α · mean((out - out.mean()) ⊙ (x - x.mean()))` —— 减去均值后是非零的，因为它测的是 fluctuation correlation 而不是 mean correlation
3. **Forward-Forward goodness**：每层用 `σ(Σy² - θ)`，**正负样本对比**才能产生有效梯度

**最小可行修复**：方案 2（中心化）—— 2 行代码改动，立即产生非零梯度。

---

## 2. 跨模块集成评估

之前实现的 `Phase C → LAVI` 链路：world_model.last_pred_error → block.lnn._last_wm_pred_error → LAVIEstimator(pred_error=...) → tanh(pred_error_scale) · pred_error 修正 LAVI。

**问题**：
- `pred_error_scale` 初始 0 → 修正始终 0 → 梯度只能通过 LAVI 的下游用途反向流回
- 当 pred_error 本身就在 0.97 这个量级（且不变）→ tanh(scale) · pred_error 即使 scale 学到一点点，乘积也是常数
- 实际效果：**这条链路在 demo 里完全没起作用**，LAVI mean 死锁在 0.5000

**根因**：和 Phase A-D 的问题同源——给了通道但没给非对称性。

---

## 3. 死代码清单

| 位置 | 状态 | 处理 |
|---|---|---|
| `world_model.py: last_error_by_position` buffer | 写入但永远不被读取 | **删除** |
| `causality.py: self._step` 计数器 | 增加但只用于 `__repr__` 调试 | **保留但精简**（调试用） |
| `rhythm.py: LAVIEstimator.pred_error_scale` | 学习率非零但实际效果死锁 | **保留 + 加 noise**（见 §5） |
| `plasticity.py: HebbianRegularizer.lavi_temperature` | 学习但 sigmoid(1.0 · 0.5) ≈ 0.62 几乎不动 | **保留**（无害） |

---

## 4. 技术债务

| # | 债务 | 严重程度 |
|---|---|---|
| TD1 | 所有 v2.0 测试只检验形状/梯度/不崩，不验证**机制有效性** | 🔴 高 |
| TD2 | `examples/v2_demo.py` 暴露的"训练 10 步无变化"应该是断言式测试 | 🟡 中 |
| TD3 | `CausalConsistencyChecker` 是纯 Python 类，无法 `state_dict` 序列化 | 🟢 低 |
| TD4 | `MTLNNLayer._last_wm_pred_error` 用 Python 属性（不在 nn.Module 体系内） | 🟢 低 |
| TD5 | 缺少跨模块集成测试（例如：use_world_model=True 时 LAVI 是否确实改变） | 🟡 中 |
| TD6 | 文档（PRD/ARCHITECTURE）声称模块"已完成"，但根据上述实证它们只是"接通了线路" | 🔴 高 |

---

## 5. 优先级排序的修复路线

### P0 — 立即修（本次提交）
1. **Phase A**: 加 noisy top-k（训练时给 scores 加高斯噪声）→ 破对称性
2. **Phase A**: 加正交性正则（bid_projectors 输出投影矩阵之间）→ 鼓励 bid 分化
3. **Phase D**: Hebbian 信号改为**中心化协方差**形式 `mean((out - out.mean()) ⊙ (x - x.mean()))` → 立即产生非零梯度
4. **删除死代码**: `last_error_by_position`
5. **加机制有效性测试**: 训练 100 步后断言 competition entropy 显著低于 log(K)

### P1 — 下次迭代（需更多工程量）
6. **Phase C**: MSE → InfoNCE + EMA target encoder（防止表示崩塌）
7. **Phase B**: cosine → effective-rank-over-window 信号（远比 cosine 敏感）
8. **跨模块测试**: 验证 use_world_model=True 时 LAVI 确实改变（不只是零冲击）

### P2 — 架构级重构（未来）
9. **Phase A 真正的 GWT 多源竞争**：从 MTLNNBlock 外部接收 (lnn_out, attn_out, coherence_out) 作为 bids，而不是当前的 K-MLP 投影同一个 x
10. **Phase C 分层世界模型**：多时间尺度预测（Rao 2024 风格）
11. **Phase B Semantic Entropy Probe**：训练一个 ~3M 参数的中间层探针

---

## 6. 与现有结构的契合度

| 维度 | 评估 |
|---|---|
| 接口兼容性 | ✅ 优秀。所有模块默认 off，零回归 |
| 配置驱动 | ✅ 优秀。MTLNNConfig 单一来源 |
| 测试覆盖 | 🟡 中等。覆盖形状/梯度但不覆盖机制有效性（见 TD1） |
| 与 rhythm.py 联动 | 🟡 中等。链路接通但实际未生效（见 §2） |
| 与 deliberation.py 联动 | ✅ 良好。Phase B 的钩子完全向后兼容 |
| 文档准确性 | 🔴 差。需要在 PRD 注明"机制有效性 vs 仅工程接通"的区别 |

---

## 7. 最坦诚的总结

我之前做的 Phase A-D **是工程脚手架，不是工作的机制**。它们：
- 通过了形状测试（结构正确）
- 通过了梯度测试（线路接通）  
- 默认关闭，零回归（接入安全）

但是它们**没有任何一个被证明能在训练中产生有用信号**。这不是 bug，是"研究原型 vs 生产实现"的差距。

下一步的正确做法是：先实施 P0 的修复（让 Phase A 和 Phase D 真的工作），再写**机制有效性测试**（这些测试会暴露当前所有"实现"的脆弱），最后才考虑 P1/P2 的架构升级。
