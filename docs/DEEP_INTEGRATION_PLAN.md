# 类脑机制深度整合设计（阶段 2）— 从"adapter 附加"到"架构核心"

> 日期：2026-08-16 · 前提：阶段 1（2B 基座充分训练）进行中
> 依据：本会话全部证据（P4 决策门、E5 系列、外推极限规律）

## 0. 为什么需要深度整合

当前类脑机制是"adapter 附加"（V2 adapter 挂在 frozen 基座外层），P4 决策门
证明这种形式的 LM 增量 ≈ 0。但 toy 层证明机制本身有效（parity 6/6、外推
0.999）。矛盾的解释：**机制有效，但附加方式无效**——类脑组件需要成为
架构核心（与注意力平等），而非事后挂载。

## 1. 三个深度整合方案（按证据强度排序）

### 1a. fast-weight 记忆核心化（S3 记忆主张的架构落地）

当前：fast-weight 是 adapter 的附加组件（frozen 基座外层）。
目标：**每层都带 fast-weight 记忆**（与 KV cache 平级的第二记忆通道）。

```
MTLNNBlock 新版结构：
  attention（KV cache 通道：工作记忆）
  + fast-weight（F 矩阵通道：持久记忆）← 核心化
  + 液态核心（selective decay：电路级计算）
```

证据链：
- fast-weight 跨会话 recall 0.56 vs 0.000（机制有效）
- P4 决策门：adapter 形式 ≈ 0（附加方式无效）→ 需要核心化
- 风险：核心化增加每层计算量（F 矩阵 O(D²)），需低秩控制

### 1b. selective_decay 长上下文核心化（S1 主张的规模化）

当前：selective_decay 在 toy 层验证（parity 6/6、外推 ×8 规律）。
目标：注入 2B 的每个液态核心 + **curriculum 训练长度**（外推规律）。

```
2B 训练协议升级：
  序列长度 curriculum：512 → 2K → 8K → 32K（按外推 ×8 规律渐进）
  目标：2B 模型在 32K-128K 上下文的精确计算能力
```

证据链：
- 外推极限 = 训练长度 × 8（五档 sweep 确认）
- O(1) 内存已实测（218MB@1M buffer）
- 结合两者 → "长上下文 + 精确计算 + 恒定内存"的完整能力

### 1c. 多步推理（deliberation）注入（推理能力路线）

当前：router/deliberation 已有雏形（entropy 3-way + causal-consistency floor）。
目标：2B 的推理能力 = 多步思考（类比 o1 的 reasoning）。

```
推理路径：
  单步前向（快速直觉）
  → deliberation 触发（高熵/低置信）
  → 多步液态迭代（stack_iterations / workspace）
  → 最终输出
```

证据：stack_iterations 已实现（P0-C′ 验证过），但未在 LM 质量上验证。

## 2. 实施顺序（依赖关系）

1. **1a fast-weight 核心化**（S3 最硬主张 → 架构落地）
2. **1b curriculum 长序列**（外推规律已确认 → 直接应用）
3. **1c deliberation**（依赖 1a/1b 的架构基础）

## 3. 验证标准（每个整合都有决策门）

| 整合 | 验证 | 通过标准 |
|---|---|---|
| 1a | 跨会话记忆基准 | recall 保持 0.56+ 且不伤 LM PPL |
| 1b | 32K 上下文的精确计算 | per-token ≥ 0.95 @ 32K |
| 1c | 推理基准（数学/代码） | 同参数下超越单步前向基线 |

## 4. 与"性能接近 Flash"战略的关系

- 1a/1b/1c 是**差异化能力**（Flash 没有的）：持久记忆、长上下文精确计算、多步推理
- 质量基础仍靠阶段 1 的充分训练（50K+ 步 + 数据扩充）
- 两者叠加 = "同参数质量追平 + 差异化能力领先"
