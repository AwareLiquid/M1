# ARR 回填比例帕累托 — 质量 vs 携带内存（0:1 → 3:1）

> **状态：Phase A（骨架）。** 扫描脚本、层子集转换、测试、判读标准均已落地；
> **所有实验数字标 `TBD(Phase B)`** —— 一个数字都还没跑过，本页不得被读成
> 结果页。产物：`benchmarks/arr_ratio_sweep.py`（只写不执行，Phase A 规定）、
> `tests/test_arr_hybrid_ratio.py`、`benchmarks/results/arr_ratio_*.{json,md}`
> （Phase B 生成）。
> **测试交接手册：[docs/ARR_RATIO_HANDOVER.md](ARR_RATIO_HANDOVER.md)**
> （现状/已知风险/分步测试/关键值/改动理由，测试人员从那进）。

## 1. 为什么扫这个比例

O 系列（纯循环 ARR，全部注意力换成 MT 循环 mixer）是个全有或全无的赌注：
它换来 O(1) 携带状态，代价是 **2.15× 教师 PPL（25.4 vs 11.8）**。而 2025–2026
的前沿几乎同时收敛到"回填一小部分全注意力层"这个折中：

| 模型 | 混合方式 | 公开数字 | 来源 |
|---|---|---|---|
| **Qwen3-Next**（2025-09） | 75% Gated DeltaNet（线性）+ 25% 门控全注意力，**3:1 周期交错**；512 专家激活 10+1（1:50） | 80B 总参 / ~3B 激活；训练成本约为 Qwen3-32B 的 1/10（GPU 资源 9.3%）；32K+ 长上下文推理吞吐约 10× | 官方博客 / Qwen3-Next-80B-A3B 模型卡 |
| **Kimi Linear**（2025-10） | **3:1 KDA（线性，Gated DeltaNet 的 channel-wise 门控改进）: 全 MLA**；全注意力层用 NoPE，位置感知交给 KDA | 48B 总参 / 3B 激活，1M 上下文；KV cache 最多降 75%；1M 上下文解码吞吐最高 6×（TPOT 6.3× vs MLA）；RULER@128k 84.3（3.98× 加速），MMLU-Pro@4k 51.0；1.4T token 公平对比下优于全注意力 | arXiv:2510.26692 |
| **GLM-5.3-Flash**（2026-08-26） | **线性 + 稀疏混合**：3 层线性注意力 + 1 层 DSA 周期交错，45 层中仅 11 层用稀疏注意力；IndexPool 把 indexer 的 4 个 key 向量加权池化为 1 个 | 320B 总参 / 18B 激活；相比 GLM-5.3 注意力计算量 ↓3.01×、KV cache ↓4.44× | z.ai 官方博客 |
| **MiniCPM4**（2025-06，**本次补入**） | **InfLLM v2 可训练稀疏注意力**（见 §6） | 见 §6 | arXiv:2506.07900 |

注意一个容易混淆的点：**GLM-5（744B A40B）本身是 DSA 稀疏注意力 + MoE，
不是线性+稀疏混合**；走线性+稀疏混合的是 GLM-5.3-Flash。这里按实际机制分列。

**我们的空白**：这些比例都是别人在千亿规模、万亿 token 上选出来的。本仓库
的蒸馏预算（TinyLlama-1.1B，两阶段 MOHAWK-lite，千步量级）完全不在同一
量级，**3:1 是否是我们这条管线的拐点，必须自己扫**。

## 2. 诚实边界（红线 1，写死在所有产物里）

> **回填 k 比例注意力层之后，模型不再是 O(1)。**
> 携带内存 = **恒定循环状态**（不随 T 增长）+ **k 比例层的 O(T) KV**。
> 两列分列，永不合并；只有 ratio 0（纯 ARR）的 KV 列为 0。

这条边界以四种方式强制：

1. `benchmarks/arr_ratio_sweep.py` 的每个 JSON / Markdown 都带
   `recurrent_state_bytes` 与 `kv_bytes{16,2}` 两个独立字段，`_BOUNDARY_NOTE`
   被写进每张表的表头；
2. 帕累托表有独立的"恒定循环状态 MB"列与"KV MB @fp16 / @2bit"列；
3. `mt_lnn/arr.py` 的 `convert_to_arr` 只在**全转**时关闭 `use_cache`——
   部分转换会把它留成 True，因为剩下的注意力层真的需要 KV cache；
4. 本页与 PRODUCT_LINES / BENCHMARKS 的新节都复述这句话。

KV 列按 `benchmarks/kv_frontier.py` 的**同一个 `kv_bytes` 函数**计费
（bit-packed + KIVI 非对称量化 scale 元数据，零点是 0 字节）。我们对自己的
hybrid 用的是和对手一样的 2-bit 口径，不享受双重标准。

## 3. 方法（一条命令可复现）

* **层子集转换**：`convert_to_arr(model, layer_indices=...)`。`None` = 全转
  （历史默认，位等价，见 `test_full_conversion_is_bit_equivalent_to_the_historical_default`）；
  集合/列表 = 只转这些层，**其余层的预训练注意力逐位不变**（同一模块对象、
  同一权重、同一 KV 路径，见 `test_layer_subset_conversion_leaves_the_rest_bit_identical`）。
* **保留哪些层**：`select_attention_layers(n, k)` 把深度均分成 n·k 个桶、各取
  中点——保留层沿全深度铺开，而不是堆在一端（堆在一端会退化成"注意力编码器
  + 循环解码器"，那是另一个模型）。实际达成比例 `len(keep)/n_layers` 会在表里
  如实报告（取整到整层，只是近似等于 k）。
* **蒸馏口径**：沿用既有 O 系列管线（`benchmarks/distill_arr.py` 两阶段：
  A 层间 teacher-forced 隐层对齐，B logit KD+CE）。扫描通过子进程按
  (ratio, seed) 调用它，**只动 `--keep_ratio` 一个旋钮**——控制行与混合行走的
  是同一份代码，配方不会漂。
* **配置**：比例 {0（现状对照）, 1:8, 1:4, 1:2} × seeds {0, 1}，同教师
  （默认 TinyLlama-1.1B-Chat）、同 seq_len / 步数 / 优化器。
* **内存账**：循环状态用解析式 `P·S·d_proto + H·D·(D+1)`（每层元素数，fp16
  2 字节），并由测试对着真实 forward 产生的状态张量逐项核对；KV 用
  `kv_bytes`，上下文阶梯 512 → 1,048,576。

## 4. 判读标准（在看到任何数字之前写死）

| 项 | 定义 | 门限 |
|---|---|---|
| **主口径：PPL 改善** | `(ppl_0 − ppl_k) / ppl_0`，对 seed 均值。分母是 ratio-0 纯 ARR 对照 | 1:4 需 **≥ 15%** |
| 副口径：向 teacher 收敛率 | `(ppl_0 − ppl_k) / (ppl_0 − ppl_teacher)` | 仅报告，不作门限 |
| **1:4 判定** | 达标 = "回填有效"；未达标 = 记 **"回填无效"的负结果**，不重新协商门限 | 见上 |
| **进 RESULTS.md 的资格** | 最优比例补齐 **≥3 seeds**（本轮 2 seeds 是探索性） | 当前 `promotable = false` |

"PPL 改善"有两种合理读法（相对对照的降幅 vs 对 teacher 差距的闭合率），
**两种都算、都进表**，主口径预先指定为前者，避免事后挑选有利定义。

**门限之外的补充观测（不改主门限）**：ARR 既往已知弱项是跨窗 recall
（RESULTS.md 记为负），而回填注意力理论上最先修复的就是 recall——只看 PPL
可能把"recall 大改善但 PPL 平平"误判为负结果。建议 Phase B 在最优比例上
补跑 `benchmarks/cross_window_recall.py` 作为补充列；1:4 的 15% PPL 门
保持不变。

## 5. 帕累托表（`TBD(Phase B)` — 占位结构）

质量 + 携带内存（上下文 `T=32768` 为参考点，全阶梯见 §5b/§5c）：

| ratio | 注意力层 | 循环层 | val PPL | PPL 改善 vs 0 | 向 teacher 收敛 | 恒定循环状态 MB | KV MB @fp16 | KV MB @2bit | 合计 @fp16 | 合计 @2bit | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0（对照） | 0/22 | 22 | 142.89 | — | — | 0.653 | **0** | **0** | 0.653 | 0.653 | control — 唯一 O(1) 行 |
| 1:8 | 3/22 | 19 | 88.92 | +37.8% | +41.2% | 0.564 | 96.0 | 12.75 | 96.56 | 13.32 | 探索性 |
| **1:4** | 6/22 | 16 | **67.37** | **+52.9%** | +57.6% | 0.475 | 192.0 | 25.50 | 192.48 | 25.98 | **PASS（门限 15%）** |
| 1:2 | 11/22 | 11 | 263.52 | −84.4% | −92.0% | 0.326 | 352.0 | 46.76 | 352.33 | 47.08 | 探索性（3/6 臂发散剔除） |

数字来自 2026-09-06 确认跑（6 seeds × 4 比例，21/24 臂有效，warmup_b 200 +
e265875 修复；8/31 的 9/12 臂作废）。KV 列参考上下文 T=32768，口径
`kv_frontier.kv_bytes`。**双峰性**：1:4 好模式 18.9–20.5 / 坏模式 162.1–164.0
（对照同样双峰 74.5–207.6）；均值口径方差大，配对口径 6/6 全胜。1:2 判负
（预算内不可训）。

**5b. 携带 MB vs 上下文（fp16 KV 列）：** 已出数（2026-09-06 确认跑），见
`benchmarks/results/rebuilt/arr_ratio_pareto.md` 的上下文表——0 比例全 T 平线
0.653 MB；1:4 随 T 线性，交点 T≈512 附近（短上下文 O(T) 项尚小于恒定状态）。
**5c. 携带 MB vs 上下文（2-bit KV 列）：** 同上，1:4 @1M = 816.5 MB vs 0 比例
0.653 MB。

两张上下文表的形状是预先知道的，也是本实验的核心读数：

* ratio 0 一行**在所有 T 上都是平的**（KV 列为 0）——这是唯一一条 O(1) 曲线；
* ratio > 0 的行**随 T 线性上升**，斜率 ∝ 保留的注意力层数；
* 两条曲线在短上下文可能几乎重合（T 小时 O(T) 项还小于恒定状态），
  交点之后才分道扬镳。交点位置 `TBD(Phase B)`。

**不画的结论**：本表不给"哪个比例最好"的产品结论——那是 §4 的门限 +
产品线定位（PRODUCT_LINES.md）共同决定的，且必须先过 seed 数门槛。

## 6. 竞品对照（只列公开数字与机制，不做未跑过的性能对比主张）

| 模型 | 机制 | 注意力层的角色 | 公开数字 | 与我们的可比性 |
|---|---|---|---|---|
| Qwen3-Next | Gated DeltaNet + 门控全注意力 3:1 | 25% 层保留全注意力（**周期交错**） | 见 §1 | **不可比**：80B 规模、万亿 token 预训练；我们是 1.1B 蒸馏档。只借"比例"这个超参，不借它的数字 |
| Kimi Linear | KDA : 全 MLA **3:1** | 25% 层为全 MLA，且用 NoPE | KV cache ↓75%；1M 解码 6×；RULER@128k 84.3 | **不可比**（48B/5.7T token）。它的 3:1 是消融出来的最优（1.4T 公平对比下 3:1 训练/验证损失最低）——这正是我们想在自己预算下复现/证伪的假设 |
| GLM-5.3-Flash | 线性 + 稀疏混合（3 线性 : 1 DSA） | 45 层中 11 层稀疏注意力；IndexPool 4:1 压 indexer | 注意力计算量 ↓3.01×、KV ↓4.44×（vs GLM-5.3） | **不可比**（320B）。**机制差异**：它保留的是*稀疏*注意力，KV 只按 top-k 索引访问；我们保留的是*稠密*全注意力，KV 全量驻留 |
| **MiniCPM4**（OpenBMB / 清华，2025-06） | **InfLLM v2 可训练稀疏注意力**：KV 分块，细粒度语义核给每块打分（块内多核取 max），GQA 组内共享 top-k 块选择；注意力层**不引入额外参数**，短序列回退稠密注意力；TopK 选择比 NSA 省 60% 选择开销 | **全部层仍是注意力**——稀疏化的是"每 token 访问哪些块"，不是层数 | 论文口径 **81% 注意力稀疏度**下长序列建模能力可比稠密；128K 上下文每 token 约关注 6K token（≈95%）；MiniCPM4-8B 用 Qwen3-8B **22% 的预训练 token**（8T vs 36T）达到可比效果；Jetson AGX Orin 上解码相对 Qwen3-8B 最高 **7×**；长上下文检索任务保持满分（官方口径） | **机制对照，不可折算成性能对比**：按公开机制推导，InfLLM v2 的块选择基于**完整 KV**，因此它压缩的是**计算/访存**，KV 仍需全量驻留（携带内存仍 O(T)）；我们的回填压缩的是**层数的 KV**。两者不在同一根轴上，**不得互相折算，也不得据此宣称优劣** |

MiniCPM4 这一行的价值在于它是一个**反例坐标**：同样面向端侧、同样想省长
上下文开销，它选择"保留全部注意力、把访问稀疏化"，我们选择"把大部分层换成
循环、只回填少量注意力"。**两种做法的携带内存随 T 的行为不同**，这正是 §5
的两列账要分开写的原因。我们没有跑过 InfLLM v2，因此本行只有公开数字与
机制描述，没有任何本仓库的性能主张。

## 7. 远程全量 runbook（Phase B，合并后执行）

本地只做冒烟；全量按仓库惯例上 Kaggle / AutoDL（T4 起，A100 更好）。

```bash
# 0) 环境（与 docs/CLOUD_P4_RUNBOOK.md 同一套前置）
git clone https://github.com/everest-an/M1.git && cd M1
git checkout iter/o-series-hybrid-ratio   # main 上没有本实验代码（PR #2 评审中）
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 1) 先看计划，不烧钱
python benchmarks/arr_ratio_sweep.py --dry_run

# 2) 本地/远端先过 CPU 冒烟（几分钟，验证转换+KD+账目全通）
python benchmarks/arr_ratio_sweep.py --smoke --out_dir benchmarks/results

# 3) 全量：4 比例 x 2 seeds = 8 次蒸馏，同教师同预算
python benchmarks/arr_ratio_sweep.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --steps_a 1000 --steps_b 2000 --seq_len 512 \
    --out_dir benchmarks/results --work_dir benchmarks/arr_ratio_out
#   -> benchmarks/results/arr_ratio_{0,1_8,1_4,1_2}.json
#      benchmarks/results/arr_ratio_pareto.{json,md}

# 4) 选出的最优比例补第 3 seed（探索性 2-seed 数字不得进 RESULTS.md）
python benchmarks/arr_ratio_sweep.py --ratios 1:4 --seeds 0,1,2 ...

# 5) 门（Phase B 验收）
python -m pytest tests/test_arr_hybrid_ratio.py -q
```

单次成本（**折算预估，非实测**——锚点为 `docs/CLOUD_P4_RUNBOOK.md` 的 A100
数字：1.5B × 5000 步 ≈ 1.5–2 h/臂、小学生 5000 步 21 min；Phase B 落地后以
实际墙钟替换）：

| 档 | 单次（3000 步 + 3 次全测试集 PPL 评估） | 8 次合计 |
|---|---|---|
| AutoDL A100（推荐） | ~20–40 min | ~4–6 h |
| Kaggle P100 / T4 | ~2.5–5 h | 20–40 GPU·h（拆 3–4 个会话，`--ratios` 分片可并行） |
若 transformers 版本无法消费循环层的空 cache 槽，加 `--no_cache` 重跑并在
Phase B 日志里记录——**质量账不受影响，只有增量解码能力受影响**。

## 8. 红线自检

- [x] 内存账两列分列，写死在脚本、JSON、Markdown、本文档（红线 1）
- [x] 探索性 2-seed 的 `promotable` 预置为 false，需 ≥3 seeds（红线 2）
- [x] 无新增依赖；`use_cache` 由 `convert_to_arr` 按是否全转决定（红线 3）
- [x] Phase B：填表 + 1:4 门限判定 + MiniCPM4 行保持"机制对照、不做性能主张"
  —— 2026-09-06 确认跑：1:4 PASS（+52.9% > 15%），6 seeds promotable；
  warmup 修复（e265875）+ 双峰性如实标注；1:2 判负。§5/§5b/§5c 已回填。
