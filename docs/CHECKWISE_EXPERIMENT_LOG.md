# CHECKWISE_EXPERIMENT_LOG — chunkwise 扫描分支实验记录与决策日志

分支 `iter/chunkwise-scan` · 记录日期 2026-08-29 · 语义审计见 `CHECKWISE_NOTES.md`（本文是它的实验侧配套）。

**数据性质声明**：本文全部性能数字来自两轮 /tmp 探针（后入仓为
`benchmarks/pscan_probe_{cell,2}.py`，原始数据 `benchmarks/results/pscan_probe_round{1,2}.json`），
设备为本机 **CPU（8 核）与 MPS**——**无 CUDA**。官方基准（`pscan_bench.py`，
Phase B）未跑。所有"GPU 上如何"的表述均为假设。

---

## 1. 现状快照（截至本记录）

- **分支 12 commits**（见 §8 清单），worktree 干净，未合并。
- **数学等价：已执行验证 PASS**。chunkwise（通用版与 constant-A 特化版）vs
  pscan 参照，全部形状/设备 max_rel_diff ∈ 2.5e-7 ~ 6.1e-7（门槛 1e-3）。
- **性能（生产默认路径 constant-A，训练形态 fwd+bwd）**：
  - T=512（train.py 真实默认）：CPU **~2.0x**，MPS **1.1~1.7x** ✅
  - T=1K：CPU 1.87x ✅；MPS 0.85x
  - T=4K：CPU 1.19x ✅；MPS 0.78x
  - T=16K：CPU 0.88x ❌；MPS 0.98x（C=32 时 **1.23x**）
  - MPS 纯前向 16K：**1.68x**（prefill 红利）
- **内存：平手不省**（训练形态 rss 比 0.84~1.05；CPU 纯前向最高 1.23）。
- **开关默认 off**，主仓零行为变化；推理/serve 路径不涉及。
- **Phase B 未执行**（等待合并指令）。
- 收窄后的主张："**生产训练长度下扫描算子 ~2x 提速、数学不变、内存持平**"，
  不是原目标的"更长更赚且省显存"。

## 2. 问题清单（按优先级，含处置状态）

| # | 问题 | 影响 | 状态 |
|---|---|---|---|
| P0-1 | `use_chunkwise_scan=True` + `selective_decay=True` 会路由到**慢 2~16x 的通用逐块版**（v1 层 `mt_lnn_layer.py` 的 lam_t 分支） | 若合入并开开关跑 selective 训练，严重回退 | **待决策**：建议开关只作用 constant-A 分支，selective 分支无视开关（合入前的小 commit + 1 个回归测试，见 §6-T1/T2） |
| P1-1 | 通用（selective）路径无特化：λ_t 逐步不同，吃不到"块不变 L"；仍是逐块 Python 循环（第一轮实测慢 2~16x） | selective 训练无法受益于 chunkwise | 记录在案；批量化方向见 §3-路线 C |
| P1-2 | 长序列档（16K+）CPU 回落至 parity 之下（0.88x） | 长上下文训练不赚 | 机制已定位（§5.3，缓存层级效应，非算法缺陷）；GPU 行为待测 |
| P2-1 | 内存不省，与分支原始主张偏差 | 叙事需修正（本文 §1 已收窄） | 已修正 |
| P2-2 | 官方 `pscan_bench.py` 矩阵缺 constant-A 对照组（探针已补，bench 没有） | Phase B 落盘数字将不含生产默认路径 | 建议 Phase B 前给 bench 加一组（几行改动） |
| P2-3 | GPU/CUDA 主张完全未验证（本机无卡） | Tensor Core 溢价假设悬置 | 待有卡机器（§6-T7/T8） |
| P2-4 | `train.py --compile` 仅 cuda 生效，本机 CPU 冒烟测不到 compile 路径 | Phase B 预期管理 | 已知，记录 |

## 3. 迭代方向与目标

**目标（收窄后）**：在不改变模型数学的前提下，让训练里每 step 跑数百次的
记忆链扫描在生产序列长度（T=512~4K）下更便宜；同时保持递归算子与
Mamba-2 SSD / FLA 生态同形状，为内核替换与 NDIT/delta 加速铺路。

- **路线 A（合入前，本分支）**：修 P0-1 开关限定 → Phase B 官方验证
  （pytest 全量 + bench + 训练冒烟）→ 按结果决定默认值走向。
- **路线 B（有 CUDA 机器后）**：`pscan_bench.py` GPU 全量档；验证 tensor
  core 溢价与 16K+ 趋势（§6-T7/T8）。这是"长序列优势"假设的真正主场。
- **路线 C（后续分支）**：通用路径批量化（块内全并行 bmm + 块间 carry 用
  pscan 扫 NC 步——特化版已示范该结构）；delta 规则 WY 折叠（接口位
  `CHECKWISE_NOTES.md` §3.1）；Triton/FLA 内核替换评估（判据 §2）。

## 4. 关键决策记录（决策 → 理由 → 结果）

- **D1 最小探针先行，代替直接进 Phase B**。理由：Phase B 全量验证贵，
  算子级对照几分钟可证伪核心主张。**结果**：两轮探针抓到 2 个静态审查
  （含验收轮）漏掉的形状 bug，避免带病进 Phase B；性能预期被校准。
- **D2 判定规则实验前写死**（等价 <1e-3；16K fwd_bwd 翻正且随 T 递增；
  内存比 <1）。理由：防事后合理化。**结果**：等价过；翻正在生产区间过、
  16K 未过；内存未过 → 主张收窄而非废弃，规则本身无一遍事后放宽。
- **D3 数值构造：log-space segsum + 符号外积 + tril-先于-exp**。否决
  cumprod 商（fp32 下溢 0/0=NaN）；否决 FLA 正域 gate（负乘子不是一等公民）。
  代价：每块 log/exp 超越函数（特化版只付一次，代价已消）。
- **D4 验收轮修 segsum 上三角 exp 上溢**（正 segsum → inf → inf×0=NaN 毒化
  整块）。触发器是静态审计，回归测试用两步 |A|=1e-30 钉死（138 nats > 88.7）。
- **D5 constant-A 特化替换通用展开**（L/g 块不变 → 构一次 L + 单次全块 bmm +
  carry 链复用 pscan_constant_A）。**结果：0.06x → 0.88~2.0x**，证明第一轮
  的瓶颈诊断（调度开销 + 重复构 L）正确。曾考虑的替代——继续调通用版的
  chunk_size——被第一轮 C 敏感性数据否决（C↑ 只缓解不治本）。
- **D6 通用（selective）路径保持逐块**：无块不变性可利用，批量化需要
  分层 carry 重构（路线 C），本分支不做。
- **D7 探针入仓留档**（修订最初"/tmp 用完即弃"）：数字可复现是实验记录
  的一部分。配套 D2 的规则写死在 `pscan_probe2.py` docstring。
- **D8 chunk_size 默认 64 保留**：CPU 最优（91ms vs C32 125ms / C128 94ms
  @16K）；MPS 最优是 32（62ms vs 77ms），但单一默认取 CPU 口径，MPS 用户
  可配。记录在案，不改默认。
- **D9 默认 off + 等价性作合入门槛**（红线）：性能主张不成立也不伤害
  任何现有路径；等价不过则 on 通路不得启用。

## 5. 实验数据（两轮探针，median of 3，形状 (2,4,2,T,64) fp32）

### 5.1 第一轮（通用路径，逐 token 乘子）— 负面结果

| 档 | T | pscan | chunkwise | 加速比 |
|---|---|---|---|---|
| CPU fwd_bwd | 1K/4K/16K | 12.2/24.6/86.1 ms | 31.9/124.4/1178 ms | **0.33/0.13/0.06x** |
| MPS fwd_bwd | 1K/4K/16K | 15.0/40.9/145.1 ms | 24.1/153.8/1277.6 ms | 0.84/0.56/0.16x |

等价 PASS（2.5e-7~6.1e-7）。C 敏感性（CPU 16K）：C32≈2138~2308ms、
C64≈1178~1288ms、C128≈806~975ms——**FLOPs 随 C 线性涨、墙钟反降** → 诊断：
瓶颈是 T/C 次 Python 迭代的调度开销与每块重复构 L，非算术。

### 5.2 第二轮（constant-A 特化版，生产默认路径）— 生产区间翻正

| T | CPU pscan_const | CPU 特化 | **CPU 加速** | MPS pscan_const | MPS 特化 | **MPS 加速** |
|---|---|---|---|---|---|---|
| **512** | 9.4~9.8ms | 4.7~5.0ms | **1.9~2.0x** | 10.5~14.5ms | 8.4~9.6ms | **1.1~1.7x** |
| 1K | 16.7ms | 8.9ms | **1.87x** | 11.5ms | 13.6ms | 0.85x |
| 4K | 49.4ms | 41.6ms | **1.19x** | 23.8ms | 30.3ms | 0.78x |
| 16K | 80.7ms | 91.2ms | 0.88x | 75.8ms | 77.4ms | 0.98x（C=32：**1.23x**） |
| 16K fwd | 16.9ms | 29.2ms | 0.58x | 23.2ms | 13.8ms | **1.68x** |

等价 PASS（3.5e-7~5.7e-7）。C 敏感性（16K fwd_bwd）：CPU C32/C64/C128
= 125.3/91.2/93.6ms；MPS = 61.7/77.4/72.6ms。内存 rss 比：CPU
0.84/1.05/1.05（1K/4K/16K），MPS 0.96~1.04。
注：T=512 为生产区间补测格（临时执行，未入 round2.json，数字见本表）。

### 5.3 16K 变慢的机制（实测推断，交叉验证）

先纠正直觉：pscan 总计算量也随 T 线性涨（log 是依赖深度不是总工作量），
chunkwise 每元素算术量（~64 MAC）始终比 pscan（~15 逐元素）**多**，赢靠
单位计算更便宜。机制分两档：

- **T ≤ 4K：工作集装进片上缓存** → 比的是计算效率 → 矩阵乘（SIMD/分块/
  零逐元素开销）溢价 ~5x，碾压 2.4x 算术膨胀 → **快 2x**。
- **T = 16K：工作集爆缓存**（X 单独 67MB，fwd_bwd 流过数百 MB）→ 两实现
  都退化为内存带宽竞赛 → 收敛到同一堵墙（80.7 vs 91.2ms，差 13%）；
  特化版略亏因多落 H_loc/dL 等全尺寸中间量。

交叉证据：同一 16K 在 **MPS 纯前向快 1.68x**（带宽高一个量级 + GPU 矩阵
单元，溢价存活）；MPS 最优 C=32（更小工作集）。→ 16K 回落是**缓存层级
效应，非算法缺陷**；CUDA（Tensor Core + HBM）预期溢价维持更久——待测。

## 6. 后续测试清单

**合入前（本分支，待指令，~30min）**
- T1 修 P0-1：开关限定只作用 constant-A 分支（v1 层 lam_t 分支无视开关）。
- T2 回归测试：`switch=True + selective_decay=True` 输出 == 旧 pscan 路径。
- T3（可选）`pscan_bench.py` 补 constant-A 对照组（P2-2）。

**Phase B（合并指令后，主 worktree）**
- T4 全量 `pytest tests/`（含 test_pscan_chunkwise 6 用例、
  test_chunkwise_scan_switch 2 用例、既有 test_parallel_scan 7 用例）。
- T5 `pscan_bench.py --smoke` → 全量，JSON 落盘（含 chunk 敏感性）。
- T6 `train.py --dummy --compile --steps 10 --batch 2`（注意 CPU 下 compile
  分支不生效，P2-4）。
- T7 回填 `CHECKWISE_NOTES.md` §5 官方数字 + 复核本文探针结论。

**有 CUDA 机器后**
- T8 `pscan_bench.py` GPU 全量档（tensor core 溢价假设的检验）。
- T9 GPU 下 16K/64K 档趋势——"长序列优势"主张的主场。

**长期（后续分支）**
- T10 通用路径批量化（分层 carry；特化版已示范结构）。
- T11 delta 规则 WY 折叠（接口位：`CHECKWISE_NOTES.md` §3.1）。
- T12 Triton/FLA 内核替换评估（判据：`CHECKWISE_NOTES.md` §2.2）。
- T13 训练端到端 A/B（开关 on/off 的 tokens/s 与 loss 曲线）——等价性
  只保证数学相同，不保证端到端吞吐提升按算子比例兑现。

## 7. 成本账

两轮探针 + 记录合计 ~1h（探针执行 ~35min，其余为实现与记录）。产出：
2 个真 bug 修复（carry 广播 `12acf56`、g 升维错位 `1cf9920`——均为
"静态审查不可见、冒烟 5 秒捕获"的形状类 bug）、1 个合入前必改问题
（P0-1）、性能主张校准（从"更长更赚省显存"收窄为"生产区间 2x"）。
若跳过探针直接 Phase B：carry 广播 bug 会让等价测试全红，至少返工一轮。

## 8. 分支 commit 清单（截至本记录，12 个）

| commit | 内容 |
|---|---|
| 50f3c40 | chunkwise 核心（通用逐块版） |
| d19048d | 等价性测试 6 用例（合入门槛） |
| 2035fd7 | use_chunkwise_scan 开关 + v1/v2 接线 |
| 6606edd | pscan_bench.py 官方基准脚本 |
| 70bb945 | CHECKWISE_NOTES.md 语义审计 |
| 43a2343 | 验收修复：segsum 上三角溢出 + v2 工厂透传 |
| 12acf56 | 探针冒烟修复：carry 广播升维 |
| 4dee98a | constant-A 特化（本轮性能翻正的实现） |
| 1cf9920 | 探针冒烟修复：特化版 g 升维错位 |
| 3317aec | 探针工具 + 两轮原始 JSON 入仓 |
| （本文档 commit） | 实验记录与决策日志 |
| （NOTES 回填 commit） | CHECKWISE_NOTES 实测回填 |
