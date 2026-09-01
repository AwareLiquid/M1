# LATENT RECURSION — 连续时间递归深度对齐外部潜空间推理坐标系

> 分支 `iter/latent-recursion` (2026-08-29)。把 M2 既有的递归深度旋钮
> (`core_iterations` / `stack_iterations`) 挂到外部已验证的潜空间推理坐标系上,
> 并占住 **"连续时间循环体"** 这个无人做过的组合。状态: 协议全部预注册、
> 驱动全部验证、账本出数; **判决级实验数据待 GPU** (见 §5)。
> 测试/接手入口: [LATENT_RECURSION_HANDOVER.md](LATENT_RECURSION_HANDOVER.md)
> (现状风险清单 + 本地门 + Phase B GPU runbook)。

## 1. 外部坐标系 (背景, 勿重复调研)

| | Coconut (arXiv:2412.06769) | Huginn/Geiping (arXiv:2502.05171) | 本仓库 |
|---|---|---|---|
| **循环体** | 末层 decoder 块, 隐状态回哺替代 CoT token | 整个 decoder, 权重绑定重复 | `stack_iterations` = 整块 (注意力+LNN); `core_iterations` = 仅 LNN 子层 |
| **可变深度** | 固定潜空间步数课程 | 测试时任意深度 (训练深度随机化) | 同 Huginn: `set_stack_iterations` 运行时可变, 训练深度 mix |
| **连续时间步长** | ✗ (离散等步) | ✗ (离散等步) | ✓ **`liquid_step_ladder`** — 迭代按 τ 阶梯加权, 一次迭代 ≠ 均匀一步 |

第三列是我们唯一的独特格: Coconut/Huginn 的循环体都是**离散等步**的
decoder 块; 液态核心的共振 bank 天然有 P×S 个 τ 尺度, 使"第 k 次迭代"可以
对不同时间常数的通道是不同比例的一步 (快 τ 多走、慢 τ 少走)。

已知教训定位 (HANDOFF §2.5): "只循环液体核心 ≠ 思考, 组合查找的计算在
注意力里" —— 因此本线的裁决对象是 **stack 循环** (整块), core 只作阴性对照。

## 2. Task 1 — 算力对齐账本 (已出数, 口径冻结)

`benchmarks/compute_accounting.py`, 解析式计数 (零 profiler), 口径写死于
docstring (MACs×2=FLOPs; 稠密层逐权重元素; 因果注意力平均 key 数; 排除
norm/softmax/偏置; fp32 访存; 手算值单测 6 项 + 与真实构建模型参数量对账)。

probe 口径 (d=104, L=2, T=54, vocab=26) 关键行:

| 路径 | N=1 | N=8 | 访存特征 |
|---|---|---|---|
| mtlnn stack (整块循环) | 24.7 MFLOPs | **157.3** | KV=0, 恒定状态 4.1 KB |
| mtlnn core (仅 LNN 子层) | 24.7 | 99.6 | 同上 |
| tfm latent (Huginn 坐标, 解析) | 27.8 | 222.7 | KV=0 |
| tfm CoT token | 28.4 | **32.8** | KV 峰值 104.8 KB |

**首个诚实结论: "省 token" 叙事在 FLOPs 口径不成立** —— 整块潜空间循环
每 pass 重算 T×T 注意力, N=8 时比生成 8 个 CoT token (增量解码) 贵
**4.8×**。潜空间的优势格是**内存**: 无 KV 增长 + O(1) 状态, CoT 8 token
需 105 KB KV。这个不对称是 Task 3 前沿的先验。

## 3. Task 2 — 裁决实验 (协议预注册, 数据待 GPU)

**假设 H**: stack 循环使深度-准确率曲线单调上升, core 不上升。
**判负标准 (写死于 `latent_recursion.judge_decision`, 不事后移动)**:
H 成立 ⇔ stack 各档均值严格单调 ∧ mean(d8)−mean(d1) ≥ 2σ(ddof=1) ∧
E0 门槛 (`publishable`: ≥3 seeds + 非双峰 + 配对 p<0.05)。判负时诊断四选一:
`budget_wall` / `bimodal_grokking_zone` / `flat_iterations_ignored` /
`direction_but_underpowered`。

协议: pointer_chase 单环 (捷径已封) + mix 课程 (difficulty=8, n=16),
深度 {1,2,4,8} × 两循环模式 × **6 seeds** (3 seeds 的配对符号检验最小
p=0.25, 数学上不可引用 —— 单测锁定此协议事实)。

驱动 `benchmarks/latent_recursion.py`: resume-safe (每配置原子 JSON,
steps 匹配才跳过), 每配置时间戳日志, `--probe` 实测 ETA。Stage-1 验证跑
(MPS 1000 步 × 8 配置 × seed 0) 已走通全链路, 全档 chance + verdict 正确
诊断 budget_wall —— 驱动正确性证据, 不可引用。

**Phase B 判决已回填 (2026-08-31)**: A100-80GB 全协议 48/48 配置
(30k 步 × 6 seeds, ≈322 GPU·h), 全档 chance (stack d1→d8 = 0.0668→0.0675,
非单调; core d8 = 0.0676), 预注册判决 **h_supported=False,
diagnosis=budget_wall** —— 循环体在该预算下从未学会任务, 深度效应免谈
(处置见 R2: 加 steps 重跑; 判据未移动)。verdict 由 48 行原始数据以本分支
冻结 judge 代码重算 (服务器就地 verdict 为崩溃残留, 不可引用)。
详见 BENCHMARKS.md "Latent recursion" 节。

## 4. Task 3 — Anytime 前沿 (协议预注册, 数据待 GPU)

两臂同任务 (pointer_chase mix): **潜空间** = 一份权重 poisson 深度 mix 训练,
深度 {1,2,4,8} 评估 (anytime 语义); **CoT** = 确定性中间符号 (口径写死:
粒度 g 的链在 g,2g,...,k 跳处显式落节点, 链尾即答案, 贪心自回归评估,
错一步链全错)。横轴 = Task 1 账本 FLOPs (E_k 期望), 纵轴 = mean_k acc。

**预注册读数规则**: 潜空间点被压制 ⇔ ∃CoT 点 flops≤ 且 seed 均值 acc≥
(平局从严); 全部被压制 → 如实记负结果。鉴于 §2 的 4.8× 先验, 预期被压制,
此结果同样是价值 (它把"潜空间推理"钉在内存格而非算力格)。

## 5. Task 4 — 液态步长 τ 阶梯 (开关就绪, 默认 off 位等价)

`liquid_step_ladder` (config) + `ladder_scale_bias` (纯函数): 潜空间第 k 次
迭代 (stack pass 与 core 迭代统一计数) 的共振 bank 尺度混合 logits 加
`−τ_s/(κ+k)` —— 快 τ 受罚轻 (多走)、慢 τ 受罚重 (少走), k→∞ 回归 stock。
第 0 次迭代无偏置 → **ON+N=1 与 OFF 逐位一致**; 零新参数; 纯 kwargs 线程
不改任何既有调用方; 6 项契约测试 + 邻近回归 94 项全绿。

按任务书降级条款: Task 2 判决出来前**不强行出数**, 开关就绪 + 设计入库。

## 6. 状态与复现

| 项 | 状态 |
|---|---|
| 账本 (Task 1) | ✅ 出数, 口径冻结, `benchmarks/results/compute_accounting.json` |
| Task 2/3 驱动 | ✅ 验证 (Stage-1 MPS 1000 步全链路 + resume/日志实测) |
| Task 2/3 判决数据 | ⏳ **待 GPU**: 30k×6seeds, MPS 实测 ≈400h (Task2) + 120h (Task3) 不可行; Kaggle T4 估 1–2 周配额 |
| Task 4 旋钮 | ✅ 就绪, 默认 off, 未出数 |

```bash
py -m pytest tests/test_compute_accounting.py tests/test_latent_recursion_driver.py \
     tests/test_anytime_frontier.py tests/test_liquid_step_ladder.py -v   # 全部协议单测
py benchmarks/compute_accounting.py                                       # 账本复现
py benchmarks/latent_recursion.py --probe --device mps                    # Task 2 实测 ETA
py benchmarks/latent_recursion.py --steps 30000 --seeds 0 1 2 3 4 5       # 判决跑 (GPU)
py benchmarks/anytime_frontier.py                                         # 前沿跑 (GPU)
```
