# LIQUID_ATOM — 液态原子的定义、迭代阶梯与族谱（B-26②）

> 2026-09-19 · 用户拍板"参考原子工程化思想开始迭代"后转正落地。
> 立法件：[ADJ-013](../kb/decisions/ADJ-013-liquid-atom-constitution-and-interface-freeze.md)。
> 对标系：Transformer token-mixer 生态三条件（接口冻结 / 单行可换 / 硬件快速路径，
> 调研谱系见 notes/searches/topics/frontier-lnn-fit.md 与 backlog B-26）。
> 本文件是"户口"不是"判决"：各变体的实验结论只给指针，数字在
> BENCHMARKS/RESULTS/结果 JSON 里（kb 铁律 1 口径）。

## 一、原子（一行）

```
decay_{p,s} = exp(-dt / τ_{p,s})
h_new = h_prev · decay + σ(W_in·x + b_in) · (1 - decay)      # 跨 S 个尺度 softmax blend
```

- 代码：`mt_lnn/mt_lnn_layer.py :: VectorizedMultiScaleResonance`（P×S 全向量化，
  一步 einsum + 递推，Python 层无逐 P 循环）。
- **τ 是可学习参数**（`log_tau`，softplus 参数化，P×S 各自独立）——时间是
  被训练量，不是超参。这一句就是与 Mamba 族（线性时不变 + 输入选择性 Δ）
  的宪法级分界：本原子是**非线性、输入条件化的时间常数族**。
- 与 Transformer 原子的对偶：attention 混合"谁看谁"（空间/序列轴），
  本原子混合"多快衰减的记忆"（时间轴）。二者可在同一 block 内共存
  （`MTLNNBlock` = attention 子层 + LNN 子层），但**原子的定义不引用
  attention**——这是 ADJ-013 冻结的方向声明。

## 二、接口（ADJ-013 冻结）

```
MTLNNLayer.forward(
    x: (B, T, d_model), h_prev: (B,P,S,D)|None, position_offset,
    use_scan, pad_mask, ladder_iter) -> (out: (B,T,d_model), h_last: (B,P,S,D))
```

- 进出都是残差流口径 (B,T,d_model)；(B,P,S,D) 是 O(1) 流式状态
  （probe 口径恒定 4.1 KB，见 LATENT_RECURSION §2 账本——这是原子相对
  KV-cache 的内存格优势）。
- 运行时深度旋钮：`MTLNNModel.set_core_iterations(n)` / `set_stack_iterations(n)`
  （model.py:1073/1095），训练深度 mix、测试时任意深度（Huginn 坐标系，
  独特格声明在 LATENT_RECURSION §1）。
- **演进纪律（实践中早已执行，此处升格为宪法）**：任何原子改动 = 同一行
  方程上的变体开关；默认 off 时代码路径**逐位不变**；契约单测当合并门
  （先例：liquid_step_ladder 6 项契约 + 邻近回归 94 项；chunkwise 等价性
  单测）。

## 三、φ_τ 迭代阶梯（原子内部滚雪球的档位表）

| 档 | 换掉的那一行 | 仓库现状 | 绑定判据/判决 |
|---|---|---|---|
| 0 固定步长 | dt=常数，τ 每通道学 | ✅ stock 默认路径 | — |
| 1 输入条件化 λ_t | λ=f(x_t)（LTC 离散化本尊）| ✅ `selective_decay`（tanh/exp/snap/ste 四式）；前史 `signed_decay` | signed：parity 不可能性修复动机（Sarrof Thm2）；selective：文本 PPL 线判负（B-1，CLOSED）；exp 式外推读数见代码注释 E5e |
| 2 多尺度共振 | 单 τ → P×S bank + blend | ✅ 默认路径（`resonance_freqs` 初始化） | — |
| 3 迭代×τ 耦合 | blend logits 加 −τ_s/(κ+k) | ✅ 开关就绪 `liquid_step_ladder`，未出数 | **H006 DORMANT**：复活绑 B-2″ BAND_HIT 或审计通过的更省通道 |
| 4 自适应步长/solver | dt_k 可学/误差驱动 | ❌ 未开工 | XLA 静态 shape 冲突需设计；rung-4 前置 = rung-3 出数 |
| 5 时程学习 | 迭代次数本身可学 | ❌ 空想档 | 不预注册（防发散）；最强差异化档，触发再议 |

对照 Mamba 谱系：S4→S6→Mamba-3 迭代的正是档 1-3 的线性近亲（Δ 一族）；
本阶梯档 3 起进入非线性区，无社区对等物（B-26 扫描结论）。
H001（core-vs-stack）= 混合发生在 mixer 内还是积分器内的对照实验，
是这张表的语义地基。

## 四、变体族谱（同一原子的家族史）

| 变体 | config 开关 | 一句话 | 状态 |
|---|---|---|---|
| stock | （默认） | exp(−dt/τ) 对角递推 | 主干，所有 off 态基准 |
| signed_decay | `signed_decay` | λ=decay·tanh(sign)，负特征值扩到 (−decay,decay) | 入墙初始化教训（tanh 饱和）；参数保留 |
| selective_decay | `selective_decay(_mode)` | λ_t 输入相关；exp 式可达 ±1 | 分支判决在 BENCHMARKS 对应节 |
| chunkwise_scan | `use_chunkwise_scan` | 同数学、SSD 式分块求和序（仅 constant-A 路） | 等价性单测为门；selective 路仍 pscan |
| liquid_step_ladder | `liquid_step_ladder/kappa` | 迭代序号×τ 的 blend 偏置，零新参数 | 就绪未出数（H006） |
| fast_weight_core | `fast_weight_core` | 块内第二记忆通道（低秩外积，零门控出） | 独立 RNG 隔离先例（init-luck 教训） |
| rhythm/hebbian 挂饰 | `use_rhythm`/`use_hebbian*` | LAVI 门控 blend / Hebb 信号采集 | 旁路件，不入原子定义 |

（命名清洗待议项：`VectorizedMultiScaleResonance` 是实现名；对外叙事名
由 ADJ-013 定为 **τ-resonance atom**。生物隐喻（微管/原纤维/MAP/GTP）
保留为参数语义层，不再充当架构叙事主语。）

## 五、硬件食谱（条件③的判定表，不承诺移植）

判定准则：原子前向能否**一行投影到该硬件的指令主食**上。

| 投影形态 | 主食 | 已实证/推断 |
|---|---|---|
| pscan（Blelloch 平行扫描，log T 深度） | 宽 SIMD/多核、eager GPU | T4/MPS 全实验史；CPU 批量线 |
| chunkwise（块内 matmul + 块间 carry） | GEMM 巨兽：A100、TPU XLA | TPU 逐位镜像实测 0.25 s/step（COMPUTE_TIERS §十） |
| 顺序递推 O(1) 状态 | 边缘/流式（带宽受限小核） | LNN 族原生叙事格；本仓库 streaming/prefill 路 |
| 昇腾达芬奇（Cube 3D×Vector）/ RISC-V(RVV)/ 寒武纪 MLU 等 | GEMM 或 VLIW/向量 | **理论同型可达**（档 0-3 数学都是对角递推+einsum），但工具链成本（CANN/自定义 ISA）远超机队收益——**不在队，不追**（ADJ-013 拒绝项） |

结论：硬件快速路径这一条件我们**已有最小实证**（XLA/TPU + pscan + chunkwise），
B-26③"专用核函数"保持绑定 rung-3 出数之后再评估——原子赢面若不在域内，
写 kernel 是给输掉的架构修路（MambaOut 教训）。

## 六、复现锚点

```bash
py -m pytest tests/test_liquid_step_ladder.py tests/test_pscan_chunkwise.py -v
python -c "from mt_lnn import MTLNNModel, MTLNNConfig; print(MTLNNConfig().tau_min, MTLNNConfig().tau_max)"
```
