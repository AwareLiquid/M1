# 非对角输入相关转移（NDIT）设计 — A5 验收标准驱动的下一架构增量

> 日期：2026-08-15 · 依据：E2-A5 实测（对角 selective_decay 在 A5 上 0.126 in_dist，
> LSTM 对照 0.988）+ Merrill et al. ICML 2024 + DeltaProduct 文献

## 1. 为什么需要（E2-A5 的精确结论）

- A5 word problem 是 NC¹ 完全（Barrington 1989），是最小的非可解群词问题。
- Merrill et al. Cor 4.7：**对角或输入无关转移的对数精度 SSM 解不了 NC¹ 完全任务**
  （假设 TC⁰ ≠ NC¹）。
- E2 实测：`selective_decay`（对角、输入相关、负特征值）在 A5 上 in_dist_tok=0.126
  （chance=0.0167）——理论预测的精确复现。
- **结论**：parity 修复（E1/E5 系列）是参数化缺陷修复；NC¹ 需要结构变更：
  **非对角 + 输入相关的状态转移**。

## 2. 理论路线（文献给出的三个出口）

| 路线 | 机制 | 出处 |
|---|---|---|
| IDS4 | 输入相关的状态转移矩阵（非对角） | Merrill Thm 5.2 |
| DeltaProduct | Householder 反射矩阵的乘积 | Siu et al. 2024 |
| 线性注意力+门控 | 输入相关线性映射链 | Yang et al. |

**统一洞察**：parity 需要"转移读 token"（E5e 已解决）；A5 需要"转移矩阵本身
随 token 旋转状态空间"（非对角）。Householder 反射是唯一谱半径恒为 1 的
参数化（酉阵），天然解决训练稳定性。

## 3. 设计：Householder NDIT 层

### 3.1 转移算子

每 token t：
```
v_t = normalize(W_v x_t + b_v)          # 输入相关方向，D 维
Q_t = I - 2 v_t v_tᵀ                    # Householder 反射（酉，谱半径 1）
h_t = Q_t (λ ⊙ h_{t-1}) + (1-λ) ⊙ A_t   # λ 保留多尺度衰减（对角部分）
```

- **非对角**：Q_t 是稠密酉阵（rank-1 修正，实现 O(D²) 而非 O(D³)）。
- **输入相关**：v_t 直接由输入投影得到。
- **稳定**：Q 酉 → ‖h_t‖ 不爆炸；λ<1 提供压缩。
- **可表达**：Householder 积可生成任意酉阵的子群；A5 的排列矩阵表示
  （60 个 5×5 置换矩阵）是酉阵的特例——理论上可学习。

### 3.2 pscan 兼容性（关键工程问题）

对角转移用 `pscan_constant_A`（O(DT)）。Householder 是矩阵转移：
- 朴素 associative scan：O(D³T) 不可接受。
- **优化**：Householder 积的低秩展开——k 个 Householder 的积是
  rank-(k+1) 修正：`Q_1...Q_k = I + U_k Σ_k V_kᵀ`（WY 表示，Golub-Van Loan）。
  分块（chunk=16 或 32）内 WY 展开 + 块间 pscan → O(D²T) 可接受。
- 或者先做 naive 版本（D=8/16 的小 proto 维度下 D³T 也便宜），正确性优先。

### 3.3 参数与旗标

```
config:
  use_householder_transition: bool = False   # NDIT 开关
  householder_chunk: int = 16                # WY 展开块大小（pscan 版）
```
在 `VectorizedMultiScaleResonance` 内替换（或包裹）现有对角转移：
- 关闭 = 现有路径（零回归）
- 打开 = λ 衰减保留，Q_t 旋转替代纯对角缩放

### 3.4 验收标准（按此顺序，全过才算成功）

1. **parity 回归**：NDIT 不应破坏 parity（E1-d16/d32 协议 ≥ 现有 exp 模式成绩）。
2. **A5 in_dist**：> 0.5 in_dist_tok（LSTM 对照 0.988 是上限参照）——这是
   Merrill Cor 4.7 的直接检验，是"非对角是否真正获得 NC¹ 表达力"的判决。
3. **A5 长度外推**：extrap_tok 显著 > 0（LSTM 是 0.83——LSTM 用非线性递归
   获得非对角，NDIT 用显式酉结构，目标是 ≥ LSTM 且状态 O(1)）。
4. **零回归**：全部现有测试（967）保持通过；LM 规模 PPL 检查。

## 4. 风险与消融

| 风险 | 预案 |
|---|---|
| 学习到恒等（Q_t → I，非对角惰性） | 检查 Q_t 的 off-diagonal 能量；必要时加"旋转正则" |
| A5 的 60 元表示维度不足 | d_proto ≥ 5 才容得下置换表示；先 D=8/16 |
| WY 展开数值不稳定 | 块内朴素循环（正确性优先），优化后置 |
| 破坏 parity（E5e 成果） | 阶段 1 只做"NDIT 平行于对角"（双路径残差），逐步切换 |

## 4.5 Householder 路线判定（2026-08-15 实测，负结果）

**M2 里程碑已跑：Householder NDIT 在 A5 上失败**（in_dist_tok 0.130 vs
LSTM 0.988），且 **rank-1 和 rank-2 分数完全相同**（0.130）。学惰诊断排除了
"没学会"：off-diagonal 能量 0.96、v_t 强输入相关 → 旋转是活跃的。

**根因判断**：Householder 对合（Q²=I）不是 A5 的正确归纳偏置——
60 个元素的群乘语义无法由对合族编码；且更新公式 h_t = Q_t(λ⊙h_{t−1}) +
(1−decay)⊙A_t 的 g_t↔h_{t−1} 交互是**加法**的，前缀积需要**乘法**交互
（LSTM 的稠密 input-to-hidden 门控正是这种交互）。

**路线修正**：下一个增量是 **DeltaProduct 式稠密低秩修正**：
```
h_t = (I + δ_t · A(x_t)) h_{t-1} + (1 - λ_t) ⊙ B(x_t)
A(x_t) = 非对合、非对角、输入相关的稠密矩阵（低秩参数化 u(x)v(x)ᵀ）
δ_t = 输入相关缩放门
```
不是继续加 Householder 秩。A5 仍是验收标准，LSTM 0.988 是标尺。

## 5. 实施步骤

1. **M1 里程碑**：最小 NDIT 单元测试（Householder 酉性质、WY 展开与朴素积
   数值一致、谱半径=1）。
2. **M2**：A5 探针上 NDIT in_dist（理论判决实验，~30 分钟 GPU）。
3. **M3**：若 M2 通过 → 长度外推 + parity 回归 + 完整测试。
4. **M4**：LM 规模 PPL + 效率测量（O(1) 状态是否保持）。

## 6. 与 E5e 的关系

E5e 的 exp 参数化解决的是**对角转移内部**的 ±1 可达性（TC⁰ 层级的表达力）。
NDIT 解决的是**转移矩阵的结构**（NC¹ 层级的表达力）。两者正交且叠加：
`λ_t = 2·exp(-softplus(δ)/τ) - 1`（标量门控）× `Q_t`（Householder 旋转）。
