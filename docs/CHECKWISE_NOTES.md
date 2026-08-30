# CHECKWISE_NOTES — chunkwise scan 语义对齐审计

> 命名沿用任务书（CHECKWISE_NOTES = chunkwise scan notes）。
> 分支：`iter/chunkwise-scan`（2026-08-29）。纯分析文档，不含实现承诺。
> **实验侧配套**：两轮对照实验的数据、决策与后续测试清单见
> `CHECKWISE_EXPERIMENT_LOG.md`（探针实测：生产区间 CPU ~2x，16K 回落，
> 内存平手——本文原"长序列显存主战场"预判已被证伪并修正，见 §4）。
> 本文回答两个问题：
> (a) `pscan_chunkwise` 与 Mamba-2 SSD / FLA GDN 内核的语义对应关系——
>     未来能否直接换用现成 Triton 内核；
> (b) 哪些算子形态在 chunkwise 框架下可行、而在关联扫描（associative
>     scan / Blelloch）框架下不可行——为 `mt_v2_delta` 加速留设计接口。

---

## 0. 背景与教训

生态事实：Mamba-2 (SSD)、GLA、DeltaNet、Kimi KDA 的**训练**内核全部是
chunkwise 分解（块内并行 matmul + 块间 carry）；TNT (ICLR 2026) 对 TTT 族
报 17× 训练提速；MiniMax 官方自述 M2 因线性注意力基础设施不成熟而回退
全注意力。教训：**递归算子的训练形态要对齐生态内核，不做自研算子孤岛**。
本仓库此前唯一的主流形态构件是 Blelloch pscan（log-depth pow2 扫描）；
`pscan_chunkwise`（本次合入）把训练路径补齐到 SSD 同族。推理路径不变：
O(1) 状态递推与训练扫描形态无关。

## 1. 本实现的 chunk 分解（事实基准）

递归：`h_t = A_t · h_{t-1} + X_t`，A 形状 `(..., T)`（每步标量乘子，对
D 通道广播），X 形状 `(..., T, D)`。chunk_size = C（默认 64）。

```
块内:  H = L @ X_chunk + g ⊙ carry          # 一次 masked matmul
       L[i,j] = Π_{k=j+1..i} A_k  (j ≤ i)   # 块内衰减矩阵（含对角=1）
       g[i]   = Π_{k=start..i} A_k          # 块首到步 i 的总衰减
块间:  carry ← H[..., -1, :]                # 每 C 步一次 O(1) 小算子
```

数值构造（`mt_lnn/parallel_scan.py::_chunk_decay_matrix`）：
- 幅值走 **log-space segsum**：`L_mag[i,j] = exp(cumlog_i − cumlog_j)`，
  cumlog 是 log|A| 的 inclusive cumsum。不用 cumprod 商——fp32 下
  0.05^64 ≈ 1e-83 会下溢，商产生 0/0 = NaN；差值+exp 只会把真值
  <1e-38 的权重 flush 到 0，与 sequential 参考的下溢行为一致。
- 符号走**精确外积分解**：σ ∈ {±1} 时 `Πσ = scum_i · scum_j`（scum 为
  符号 cumprod），所以 signed/selective decay 的负乘子（λ∈(−1,1)）是
  一等公民，不需要 log 符号域。
- 等价性 = 浮点重结合误差，rtol/atol=1e-3 由
  `tests/test_pscan_chunkwise.py` 钉死（合入门槛）。

## 2. (a) 与 Mamba-2 SSD / FLA GDN 内核的语义对应

### 2.1 对应表

| 本实现 | Mamba-2 SSD / FLA `chunk_gated_delta_rule` | 备注 |
|---|---|---|
| `h_t = A_t h_{t-1} + X_t`，A 标量共享 | `S_t = a_t ⊙ S_{t-1} + k_t v_tᵀ`，a_t per-channel 对角 | 我们是 SSD 的特例：把 k⊗v 预先折叠进 X、衰减取各通道共享标量。标量→per-channel 不改变任何矩阵结构，只是 segsum 变逐通道 |
| `L = segsum(cumlog) ⊙ 符号外积 ⊙ tril` | `(L_tril ⊙ decay-mask)`，同为 log a 的 segsum | 同构。FLA 的 gate 在正数域（exp/cumsum），我们多出符号因子 |
| `carry = H[..., -1, :]`，逐 chunk 串行 | chunk 末状态 core state，`initial_state/final_state` 进出参 | 相同；FLA 内核原生支持状态注入，外层 carry 循环可直接换成内核调用 |
| 布局 `(..., T, D)` 任意前导维 | `(B, T, H, K/V)` | 把 `(B,P,S)` 前导维 flatten 成 batch 维即可，无需转置语义 |

### 2.2 能否直接换现成 Triton 内核：**能，判据如下**

1. **结构上可换**：`_scan_chunk` 的对外契约（进：`A_chunk, X_chunk,
   carry`；出：`H_chunk, next_carry`）与 FLA 的
   `chunk_*` 内核逐参对应。替换点只有一个函数，等价性测试
   （`test_pscan_chunkwise.py`）原样保留作为换内核的验收——
   内核替换不动语义层，这正是"对齐生态"的收益。
2. **signed λ 是唯一的语义缝隙**：FLA/SSD 的衰减 gate 全部在正数域，
   没有负 gate 内核。换用时两条路：(i) 把 signed λ 拆成
   `sign(λ)·|λ|`，符号因子留在 PyTorch 侧做 outer-product、幅值调内核
   （本实现的分解方式，零语义损失）；(ii) 等 FLA 支持负 gate（无 ETA）。
   默认（正 decay）路径无此问题，可直接整块替换。
3. **精度契约不同**：Triton 内核块内用 fp32 累加、块间可 bf16；我们是
   全程 fp32 且 rtol=1e-3。换内核后等价性容差需放宽到混合精度口径
   （预计 rtol≈1e-2 量级），合入门槛的判定式不变、常量另立。
4. **不引入新依赖是本分支红线**：换内核是后续独立分支的事（届时
   Triton/FLA 随生态基线一起评估），本文只固化对应关系与判据。

## 3. (b) chunkwise 可行 / 关联扫描不可行的算子形态

分界线一句话：**状态转移的 compose 成本决定框架**。关联扫描要求把每步
转移打成一个可结合的二元运算；compose 成本为 O(D³) 的算子（D=状态宽）
在 scan 里每步都付一遍；chunkwise 把块内 C 步折叠成 **C×C 的与 D 无关
的小矩阵运算**（三角求解/逆），每块只付一次。

| 算子形态 | 关联扫描 (pscan) | chunkwise | 折叠的核心运算 |
|---|---|---|---|
| 标量/对角衰减 + 加性输入（本实现） | O(T log T)，可行 | 块内 `L @ X`，可行 | segsum + masked matmul |
| **Delta 规则**（DeltaNet/GDN）`h_t=(I−β_t k_t k_tᵀ)h_{t-1}+β_t k_t v_tᵀ` | compose = D×D 矩阵乘，O(D³)/步，**不可行** | **可行**：块内 WY 表示 `S=(I−tril(β KKᵀ,−1))⁻¹`（C×C，与 D 无关）+ 2 次 matmul | C×C 单位下三角逆/前代 |
| **Householder / DeltaProduct NDIT**（`docs/NONDIAGONAL_TRANSITION.md`，现为 sequential loop） | D×D 单式矩阵 compose，O(D³)/步，**不可行**（现状即证据：`mt_lnn_layer.py` 走逐 token Python 循环） | **可行**：块内 C 个低秩修正的 UT/WY 折叠，同上 | C×C 三角求解 |
| 逐 token 完整 D×D 转移（无低秩/对角结构） | 不可行 | 也只有 O(C·D³)，**不可行** | —（chunkwise 不是万能） |
| 任意非线性 readout（tanh(h) 等） | 不可行 | 不可行 | — |

### 3.1 为 `mt_v2_delta` 留的设计接口（只分析，不实现）

- **仓库现状锚点**：`mt_lnn_v2.FastWeightMemoryV2` 已是 chunk-parallel
  （`fast_weight_chunk=64`），其 docstring 明说 `write_rule="delta"` 因
  "nonlinear in (k,v,F), the closed-form chunk scan does not apply" 而
  退化为 **sequential**——这正是上表 Delta 规则行的实例化：卡点不是
  理论而是缺 WY 折叠。
- **接口位**：`pscan_chunkwise` 的可替换点是
  `_chunk_decay_matrix`（返回 `(..., C, C)`）。Delta/UT 形态把该函数的
  返回值换成 `(I − tril(β KKᵀ, −1))⁻¹ ⊙ decay`，`_scan_chunk` 的
  `L @ X + g·carry` 骨架与块间 carry 协议**原样保留**。也就是说：
  chunkwise 框架落地后，delta 加速是"换一个 C×C 矩阵构造器"的局部改动，
  不再需要动扫描协议本身。
- **前置条件**：delta 块内二次型要求 K/V 显式存在（不能预折叠进 X），
  即 `pscan_chunkwise(A, X)` 的签名届时需扩为携带 K/V 的变体（类似 FLA
  的 `chunk_delta_rule(q,k,v,β)` 口径）；这是 `mt_v2_delta` 分支的
  设计决定，本分支不做。

## 4. 数值诚实边界

- **等价性口径**：与 `pscan_sequential`/`pscan` 的差异全部来自求和顺序
  重结合 + log/exp 各 1 ulp（每矩阵项 ~1e-5 rel）。rtol/atol=1e-3 的
  测试常量留了 100× 结构性余量：任何 chunk 边界 off-by-one、carry 错
  传、mask 错误都是 O(1) 误差，必然击穿。
- **下溢是特性不是 bug**：真值 <fp32 min 的权重（如 0.05^64）在
  sequential 侧同样 flush 到 0，两侧一致；segsum 差值形式保证永不出现
  0/0。
- **chunk_size 敏感性（探针实测回填，CPU/MPS 口径）**：特化版 16K
  fwd_bwd——CPU：C32/C64/C128 = 125.3/91.2/93.6ms（最优 64）；MPS =
  61.7/77.4/72.6ms（最优 32，更小工作集）。通用逐块版的 C↑ 反而更快
  （调度瓶颈），特化版恢复"C 与算术量正相关"的正常曲线。默认 64 保留
  （CPU 口径，MPS 可配）。**内存预判修正（2026-08-29）**：本文原预判
  "chunkwise 的 L 与 T 无关、长序列档是显存差距主战场"——探针实测
  **证伪**：训练形态峰值内存平手（rss 比 0.84~1.05），因为反向需保存
  H_loc/dL 等全尺寸中间量，与 pscan 的递归中间量相抵。完整数据见
  `CHECKWISE_EXPERIMENT_LOG.md` §5。
- **红线重申**：等价性测试不绿则 `use_chunkwise_scan` 的 on 通路不合入
  （只能默认 off 合入）；默认路径本分支零行为变化。

## 5. 实测状态与 Phase B 待回填

**探针已回填（CPU/MPS，2026-08-29，详见 `CHECKWISE_EXPERIMENT_LOG.md`）**：
- [x] 等价性：执行验证 PASS（max_rel_diff 2.5e-7~6.1e-7，门槛 1e-3）
- [x] 加速比（constant-A 特化版，fwd_bwd）：T=512 CPU ~2.0x / MPS
      1.1~1.7x；1K 1.87x；4K 1.19x；16K 0.88x（CPU）/ 0.98x（MPS，
      C=32 时 1.23x）；MPS 纯前向 16K 1.68x
- [x] chunk_size 敏感性：CPU 最优 64，MPS 最优 32（§4）
- [x] 内存：平手不省（预判修正，§4）

**Phase B 官方基准（2026-08-29 本地 CPU 档已跑，`benchmarks/results/pscan_bench*.json`）**：
- [x] const（生产默认）路径 chunk/pscan 墙钟比：1K 0.563x / 4K 0.720x /
      16K 0.844x / 64K 0.398x——全长度为正；64K 反常走高存疑（内存压力
      嫌疑，待 GPU 复核，不外推），详见 EXPERIMENT_LOG §5.4
- [x] general 路径慢 3.5~52.9x——P0-1 开关限定（selective 不触达）的正式背书
- [ ] 有 CUDA 机器后跑 GPU 档（tensor core 溢价假设）
- [ ] （若达标）是否推进"换用现成 Triton 内核"的独立分支立项
