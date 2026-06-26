# External Plugins — 技术方案存档（MT-LNN / O1）

> 状态：**骨架已落地**（2026-06-27）。插件 A（Kuramoto）+ hook 基建 + 插件 C
> （收敛早停诊断）已实现并通过契约冒烟测试；插件 B（GWTB-Pull）为 stub。
> 验证：`python -m external_plugins.ablation.smoke_kuramoto` → PASS
> （observe 位级一致 / detach 复原 / 权重不变 / intervene 有限有界）。
> 真消融：`ablate_kuramoto` 已在 173M + tiny 两个 checkpoint 上跑出 PPL/R 结果
> （见 §5.5；含去各向异性对照 `center=True`，结论按 centered R 写）。
> 本目录下为外置插件方案。**O1 主干冻结，零修改。** 任何实现都不得改动
> `mt_lnn/` 主干源码、不得参与训练、不得改动原生模型权重。

---

## 0. 一句话结论

Un-0（Unconventional AI）经分析属于**同方向技术路线**（Kuramoto 耦合振荡器 + 吸引子计算），
**未超出**我们现有的 `mt_lnn/quantum_coupling.py`（耦合）+ `mt_lnn/attractor_ops.py`（吸引子）架构。
去魅要点：Un-0 宣称的"物理 1000× 能效"依赖**模拟振荡器硬件**，CPU 仿真拿不到该红利；
我们能吃到的只是**架构红利**（紧凑状态、流式 O(1) 记忆、极致省参）。

因此：**不改 MT-LNN/O1 主干**，仅以**推理期外置插件**形式新增三个可选能力，
只在外围做消融对照，正负结果均归档（阴性结果亦写入论文）。

---

## 1. 硬性约束（铁律，实现时不得违背）

1. **主干冻结**：不修改 `mt_lnn/` 下任何主干源码（`model.py` / `mt_lnn_layer.py` / `llama_adapter.py` …）。
2. **位置固定**：所有插件代码只许放在 `external_plugins/`。
3. **仅推理期**：插件只在 inference 阶段生效（`torch.no_grad()` 全程），**不参与训练、不产生梯度**。
4. **不侵入前向**：不得编辑主干 `forward()` 的模块图。插件只能通过
   **只读 forward hook / 模型 wrapper** 捕获中间激活或输出，在外部做后处理。
5. **零权重改动**：不改动、不微调、不重载原生模型权重。
6. **可彻底卸载**：移除 hook / wrapper 后，模型行为与原生完全一致（位级一致可作为验收项）。

> **关于"不侵入前向"的精确含义**（避免实现时跑偏）：
> 相位耦合等操作客观上需要读到 hidden states 才能算。这里"不侵入"指
> **不改主干源码、不入训练计算图**——通过 `register_forward_hook` 只读取激活，
> 把 Kuramoto 弛豫 / 牵引 / 早停作为**旁路计算**叠加在输出侧，原生 `forward` 定义一字不动。
> 这与"修改网络结构"是两件事，验收时以"摘掉 hook 即复原"为准。

---

## 2. 三个外置插件

### 插件 A — Kuramoto 相位耦合（KuramotoCoupling）
- **目的**：把相位同步度当作**信息绑定信号**，对应微管相干假说（Orch-OR）。仅做消融对比。
- **数学**：
  - 状态：每通道一个相位 `θ_i ∈ [0, 2π)`（从激活相位/投影得到）
  - 动力学：`dθ_i/dt = ω_i + Σ_j K_ij · sin(θ_j − θ_i)`
  - 读数：order parameter `R = |mean(e^{iθ})| ∈ [0,1]` 作为相干/绑定度
- **CPU 优化要点**：
  1. **线性收缩版优先**：`sin(Δθ) ≈ Δθ` → 退化为图拉普拉斯扩散（小稠密 matmul），
     落进 `attractor_ops` 可闭式分析的线性区；非线性 sin 仅在消融证明有增益时加回。
  2. **约束 `spectral_radius(K) < 1`**，保证少步收敛（settling ≤ 2–4 步）。
  3. **固定小步数 forward-Euler**，**不用**自适应 ODE solver。
  4. K 走**带状/稀疏**（复用 `quantum_coupling.py` 环形拓扑接口）→ O(N·k) 而非 O(N²)。
  5. 相位有界 → 低精度稳定，仅 wrap-around 累加器留 fp32，其余可 int8。
- **复用**：`mt_lnn/attractor_ops.py`（spectral_radius / settling_time / lyapunov）只读调用。
- **明确不做**：CPU 路径**不用** VQC（`quantum_coupling.py` 自述比经典慢 ~10×，留作 GPU 研究分支）。

### 插件 B — GWT 顶层牵引（GWTB-Pull）
- **目的**：用**小神经元集群**作为 driver 牵引主网络状态，替代部分注意力机制，
  让自上而下调制更贴合类脑环路。
- **机制**：复用 O1 已有的 competitive GWT-B bid 通路（`use_competitive_gwtb`），
  在**推理期**以小集群对 workspace 激活施加只读偏置（耦合式注入），**不重训 bid 权重**。
- **CPU 优化要点**：集群规模小、偏置为低秩；与插件 A 共用相位/耦合原语。

### 插件 C — 收敛早停（Settling-based Early Exit）
- **目的**：动态提前终止推理，省 CPU wall-clock。
- **机制**：接 `attractor_ops.settling_time`，当状态进入容差带即停止迭代/层推进；
  与 κ-gating 的动态跳算同源。**不用**自适应 solver。

---

## 3. 目录规划（实现时）

```
external_plugins/
├── README.md                 # 本文件（方案存档）
├── __init__.py               # 导出 PluginRunner / KuramotoCoupling / SettlingEarlyExit
├── hooks.py                  # [已实现] 可摘除 forward-hook 基建，observe/intervene 双模
├── kuramoto_coupling.py      # [已实现] 插件 A：均场 Kuramoto 相位耦合 + order parameter R
├── early_exit.py             # [已实现] 插件 C：收敛早停诊断（复用 attractor_ops）
├── gwtb_pull.py              # [stub]    插件 B：GWTB 牵引（接口待对齐 mt_lnn/gwtb.py）
└── ablation/
    └── smoke_kuramoto.py     # [已实现] 铁律契约冒烟测试（PASS）
```

> 主干 `mt_lnn/` 不新增 import、不新增 config flag。插件通过 wrapper 在外部装配。

---

## 4. 实验方案

- **对照口径**：沿用 `experiments/_o1_full_stack_smoke.py`（grad 非死 / loss 下降 / 联合收敛）
  作为"未破坏原生"的基线参照；插件实验额外加一条 **CPU 延迟基准**。
- **主指标**：
  - `settling 步数 vs PPL` 曲线（核心权衡）
  - order parameter `R`（相干/绑定信号是否真实存在）
  - CPU 单流延迟（×K 步数代价是否可控）
- **数据/任务**：现成 WikiText-2 PPL + needle bench（复用 `benchmarks/`）。

---

## 5. 验收与归并策略

- **效果好** → 仍以**外置插件**形态保留（除非另行决策才考虑并入主干）；写入正向实验。
- **效果不佳** → 留存为**阴性实验**，写入论文（符合"正负皆归档"打法）。
- **任何情况** → 摘掉 hook/wrapper 后主干行为复原（验收硬指标）。

---

## 5.5 真消融结果（2026-06-27，held-out 16×128）

> 命令：`python -m external_plugins.ablation.ablate_kuramoto`。全程 `no_grad`、
> observe 位级一致已验证、权重零改动。两个 checkpoint 横向对照。

| checkpoint | 层数 | baseline PPL | R_raw | R_centered | R_random(P=13) |
|---|---|---|---|---|---|
| m2_out_v7 (173M, step5000) | 12 | 195.79 | 0.584 | **0.318** | 0.247 |
| tiny (step20000)           | 6  | 31156   | 0.535 | **0.350** | 0.247 |

**关键发现 1 — order parameter R 一半是各向异性假象。**
裸 `R≈0.58` 看着很像"强相干绑定"，但**减掉跨通道共有主方向**（de-anisotropy
control, `center=True`）后 R 掉到 `0.32`。也就是说原始高 R 约**一半来自 hidden
state 的共享主方向**，并非真正的通道间相位绑定。不过去各向异性后的 `0.32` 仍**高于
P=13 均匀随机相位基线 `0.247`**（+0.07），所以存在**弱但真实**的跨通道结构——
远没有裸读数暗示的那么强。结论必须按 centered 值写，不能用 0.58。

**关键发现 2 — intervene 效果与模型训练度强相关。**
- 训得好的 173M 上，相位牵引**单调小幅抬高 PPL**（+0.02% ~ +0.13%，随 α 单调）→
  对一个已收敛的好模型，外加相位偏置是**温和阴性**。
- 欠训的 tiny（PPL=3.1e4，近随机）上，相位牵引"看似有益"（-0.08% ~ -0.88%）→
  但 baseline 太差，任何平滑扰动都能蹭低 PPL，**不构成强证据**。

**归档判定**：当前形态下 Kuramoto 相位耦合为**诚实阴性/弱信号**，按"正负皆归档"
写入论文阴性实验。R 的真实结构成分（centered>random）值得作为"O1 内部确有跨通道
相位组织"的观测证据保留，但绝不能用被各向异性夸大的裸 R 叙事。

---

## 6. 背景溯源

- 触发：Un-0 科普稿（机器之心，Unconventional AI 发布的物理计算生成模型）。
- 判断：同方向、无架构超越；我们的微管/Orch-OR 框架是**更有生物根基**的版本，
  可反向锐化 O1 叙事——"有真实 LM 用途的、可 CPU 流式跑的振荡器小模型"。
- 适配结论：振荡器/相位耦合**适合 O1（CPU 小模型线），不适合 M1（GPU 零开销 adapter 线）**。
