# O-series 回填比例实验 — 测试交接手册（Phase A → Phase B）

> **读者**：接手 Phase B 测试/执行的人。
> **写于**：Phase A 完成、任何实验数字产生之前（2026-08-29）。
> **性质**：操作手册，不是结果页——实验数字落在
> [docs/ARR_RATIO_PARETO.md](ARR_RATIO_PARETO.md)（Phase B 回填）。

## 0. 一句话现状

分支 `iter/o-series-hybrid-ratio`（worktree `../M1-o-series-ratio`）已把
"注意力回填比例 {0, 1:8, 1:4, 1:2} 的质量-内存帕累托"这个实验从**不可问**
变成**一张待填的表**：层子集转换、扫描脚本、预注册判读门、两列内存账、测试
全部就位，但**一行代码都没有执行过**（Phase A 协议禁止，只过了 py_compile）。
Phase B = 本手册 §2 的测试门 + 远程算力。

## 0.5 分支谱系与迭代时间线（新接手者先读）

```
main(c564a5d) ──38 commits── iter/irregular-streaming-edge(e87919e) ──5+ commits── iter/o-series-hybrid-ratio
```

- **本线不是从 main 切的**：任务协议要求从主 worktree 当时位置（edge 线顶端
  e87919e）建分支，因为扫描脚本的 KV 记账直接复用 edge 线落地的
  `benchmarks/kv_frontier.py`。
- **PR #2**：https://github.com/everest-an/M1/pull/2 —— head = 本分支，
  base = edge 线，diff 恰好本线的 commit，状态 OPEN **未合并**。评审通过后
  先合 PR #2 进 edge 线；edge 线何时进 main 是另一条 PR 的事。
- **O-series 质量迭代史**（ratio-0 对照的 25.4 从哪来）：13k → 264 → 32.9
  → 25.4（~18M 累计蒸馏 token，teacher 11.8，见 `docs/PRODUCT_LINES.md`）。
  注意：25.4 是既往轮参考值，本实验会在**同预算下重跑 ratio-0 作真对照**，
  新数以本轮为准。

## 1. 现状与已知问题（测试前必读）

### 1.1 已交付（commit 清单）

| commit | 内容 |
|---|---|
| `0c5baa6` | `mt_lnn/arr.py`：层子集转换语义 + 比例规划器 + 解析式状态账 |
| `8cbd267` | `benchmarks/arr_ratio_sweep.py`（511 行，只写不执行）+ `distill_arr.py` 默认关闭的透传参数 |
| `218cc1f` | `tests/test_arr_hybrid_ratio.py`（5 用例，CPU <2min 设计预算） |
| `4f71e24` | 文档骨架：ARR_RATIO_PARETO / PRODUCT_LINES / BENCHMARKS 新节 |
| `2eff92b` | 测试交接手册 + runbook 时间预估（折算值，标注见 §2.2） |
| （最新） | 手册补录：分支谱系（§0.5）、关键决策记录（§1.4）、设计盲区（§1.5）、文档地图（§1.6）、PR 状态（§5） |

共改 8 个文件；`serve/server.py`、依赖文件、其他分支职责零触碰。

### 1.2 核心事实与已知风险（诚实清单）

**核心事实：代码从未执行。** Phase A 协议只允许写代码 + py_compile（已全过），
所以以下风险是真实的、预期由 Phase B 第一轮测试暴露的：

| # | 风险 | 症状 | 处置 |
|---|---|---|---|
| R1 | 固定版本 transformers 对循环层空 cache 槽的兼容性（5.x 惰性写共享 Cache） | 混合学生 forward 报 IndexError/TypeError | 加 `--no_cache` 重跑（质量账不受影响，只影响增量解码）；**无论成败记入 Phase B 日志** |
| R2 | 测试直接使用流式内部属性（`_stream_h` / `_stream_fw` / `stream_enabled`） | `test_analytic_state_bytes_*` AttributeError | 属性名/用法漂移，回 worktree 修，重新走合并请求 |
| R3 | 主 worktree 有其他线 WIP：修改中的 `mt_lnn/session_state.py` + untracked `test_event_stream.py` / `test_parametric_memory.py` | 全量 pytest 把 WIP 测试一并收集，失败归因混乱 | 先单独跑本分支测试（§2.1 步 2）；全量失败先看**挂在哪个文件**再归因 |
| R4 | 远程单次耗时尚无实测（`distill_arr.py` 无墙钟日志进版本库） | 排程偏差 | §2.2 给的是折算预估带，落地后用实测替换 |

### 1.3 红线（违反任一即返工）

1. 回填比例 >0 的任何配置**不得宣称 O(1)**——内存 = 恒定循环状态 +
   保留层 O(T) KV，两列分列写死在所有产物里；
2. 2-seed 探索性数字**不得进 RESULTS.md**（脚本已预置 `promotable=false`，
   最优比例补齐 ≥3 seeds 才有资格）；
3. 不改 `serve/server.py`；不碰其他分支职责（块内配方 / chunkwise 扫描实现 /
   全配置 KV 账表——本分支只报自己四个配置的解析式 KV）。

### 1.4 关键决策记录（为什么是现在这个样子）

| # | 决策 | 备选与不选的原因 |
|---|---|---|
| D1 | 蒸馏入口 = `benchmarks/distill_arr.py` | 任务锚点写的是 `train_distill.py`，但那是从头训小学生的 P2 线；O-series 的 25.4 序数出自 distill_arr.py 的原地转换管线，沿用它才能"同教师同预算、配方不漂" |
| D2 | 保留层中点铺开（均匀桶取中点） | 堆在一端会退化成"注意力编码器 + 循环解码器"。22 层留 6 层时落点约每 3–4 层一个，与 Qwen3-Next / Kimi 的周期交错近似等价，混杂有限 |
| D3 | 判读门预注册：主口径 `(ppl_0−ppl_k)/ppl_0`，1:4 ≥15% | "向 teacher 收敛率"作副列同报但不作门——两种定义都进表，防事后挑有利的 |
| D4 | KV 记账复用 `kv_frontier.kv_bytes`（fp16 / 2bit，2bit 含 KIVI scale） | 给对手用什么会计，给自己就用什么；否则帕累托表自欺 |
| D5 | `use_cache` 仅全转时关闭 | 修 bug：hybrid 保留的注意力层真的需要 KV cache |
| D6 | 按 (ratio, seed) 子进程跑，只动 `--keep_ratio` 一个旋钮 | 转换原地不可逆天然要进程隔离；单旋钮保证控制行与混合行同配方 |
| D7 | PR base = `iter/irregular-streaming-edge`（而非 main） | base = main 的 diff 会携带基线 38 个 commit 的噪音；代价是基线分支也要推远程（已推） |
| D8 | Phase A 零执行，语法门 = py_compile | worktree 无 .venv/data 是设计使然；一切执行验证押后到 Phase B（本手册 §2） |

### 1.5 尚未覆盖的盲区（R1–R4 之外，接手人应知）

R1–R4（§1.2）是执行类风险；下面是**实验设计层面**的已知盲区与建议动作：

| # | 盲区 | 影响 | 建议 |
|---|---|---|---|
| G1 | 判读门只看 PPL，而 ARR 既往已知弱项是**跨窗 recall**（RESULTS.md 记为负）——回填注意力理论上最先修复的恰恰是 recall | 可能出现"PPL 不过门、recall 却大幅改善"，被 15% 门误判为负结果 | 在最优比例上补跑 `benchmarks/cross_window_recall.py` 作**补充观测列**（不改主门限） |
| G2 | 2-bit KV 是解析记账口径（KIVI 式），仓库没有 2-bit 量化的 serving 实现 | 内存表是"账"不是实测 | 定位就是与 kv-frontier 同口径对比；要实测另立任务 |
| G3 | 蒸馏与评估上下文 = 512 | PPL 改善未必迁移到长上下文 | 若 1:4 过门，追加 sliding-window 长上下文 PPL（复用既有 eval 口径） |
| G4 | hybrid 的流式/增量解码路径未接 serve（红线禁改 `serve/server.py`） | `--no_cache` 只保质量账，不保增量解码能力 | Phase B 记录 cache 行为即可；serving 接线另立任务 |
| G5 | 竞品数字（MiniCPM4 等）未逐条二次核源 | 对照表弱化 | 引用处已标来源（arXiv:2506.07900 等），不做性能主张，风险可接受 |

### 1.6 文档地图（想知道什么去哪读）

| 问题 | 文档 |
|---|---|
| 怎么测、怎么接手、风险与盲区 | 本手册 |
| 实验设计 / 判读门 / 竞品对照 / 远程 runbook | `docs/ARR_RATIO_PARETO.md` |
| O-series 与 hybrid 的产品线定位 | `docs/PRODUCT_LINES.md` |
| 基准口径与账本约定 | `BENCHMARKS.md`（hybrid-ratio 新节）+ `docs/KV_FRONTIER.md` |
| 既往结论、已撤回主张（防重复踩坑） | `RESULTS.md`（重点 "What we do NOT claim"） |
| 云 GPU 环境惯例与 A100 耗时锚点 | `docs/CLOUD_P4_RUNBOOK.md` |

## 2. 怎么测试

### 2.0 服务器路径（fresh clone——远程测试同事从这里开始）

远程 `main` 上**没有**本实验的代码（本线在 PR #2 的分支上，评审后才合入
edge 线、再随 edge 线进 main），所以服务器上必须显式切分支：

```bash
git clone https://github.com/everest-an/M1.git && cd M1
git checkout iter/o-series-hybrid-ratio     # PR #2 的分支，含全部 commit
pip install -r requirements.txt
# 环境三件套先落 Phase B 日志（R1 是版本相关风险，缺了没法归因）：
python -c "import torch, transformers, sys; print(torch.__version__, transformers.__version__, sys.version)"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python benchmarks/arr_ratio_sweep.py --dry_run   # 计划打印，零算力
python benchmarks/arr_ratio_sweep.py --smoke     # CPU 几分钟，零下载零 GPU
```

- AutoDL 拉不动 HuggingFace 时加镜像：`export HF_ENDPOINT=https://hf-mirror.com`
  （Kaggle 一般不需要）；
- 服务器测试**不必等 PR 评审/合并**——直接测分支即可，与评审并行；§2.1 的
  "合并后"路径是评审通过后的协议动作，两条路不冲突；
- 全量结果回传：`benchmarks/results/arr_ratio_*.{json,md}` commit 回本分支
  （或打包发负责人），运行日志一并附上——尤其要记录 R1 是否触发、
  `--no_cache` 是否用到。

### 2.1 本地门（合并后约 15–20 分钟）

前提：主 worktree `/Users/aricredemption/Projects/M1`，分支
`iter/irregular-streaming-edge`，用其 .venv。

```bash
# 0) 合并（fast-forward：只动本分支 8 个文件，与在改的 session_state.py 零冲突；
#    反悔用 git reset --keep e87919e，未提交改动保留）
git merge iter/o-series-hybrid-ratio

# 1) 本分支的门：5 用例（内含四比例一步 KD 冒烟），CPU 预算 <2min
python -m pytest tests/test_arr_hybrid_ratio.py -v

# 2) 全量回归（R3 归因注意）
python -m pytest tests/ -q

# 3) 扫描冒烟：随机小教师、CPU 几分钟、零下载零 GPU
python benchmarks/arr_ratio_sweep.py --dry_run   # 先看计划，不烧钱
python benchmarks/arr_ratio_sweep.py --smoke
```

**冒烟通过的判定清单**：
- 打出 4 行 `[smoke] ratio 0 / 1:8 / 1:4 / 1:2`，val PPL 有限正值
  （随机教师，**不是**质量信号）；
- 汇总表里 ratio 0 的 KV 列为 **0**，且只有它为 0；
- 恒定循环状态字节随比例**递减**（转换的层变少）、KV@fp16 **递增**；
- `promotion` 块 `promotable: false`（2 seeds 的预置）。

### 2.2 远程全量（完整 runbook 见 ARR_RATIO_PARETO.md §7）

配置：TinyLlama-1.1B 教师，steps_a 1000 + steps_b 2000，seq 512，
batch 1 × grad_accum 4，4 比例 × 2 seeds = 8 次独立子进程。

| 档 | 单次 | 8 次合计 |
|---|---|---|
| AutoDL A100（推荐） | ~20–40 min（含 3 次全测试集 PPL 评估） | **~4–6 h，一个下午** |
| Kaggle P100 / T4（免费） | ~2.5–5 h | 20–40 GPU·h，拆 3–4 个会话（单会话 12h 上限） |

> 预估为**折算值**（锚点：`docs/CLOUD_P4_RUNBOOK.md` 的 A100 实测——
> 1.5B × 5000 步 ≈ 1.5–2 h/臂、小学生 5000 步 21 min），非 distill_arr.py
> 自身实测；落地后以实际墙钟替换本表。
> 8 次相互独立，可拆会话并行：`--ratios 0,1:8` 与 `--ratios 1:4,1:2` 各占一个。

最优比例补第 3 seed：`python benchmarks/arr_ratio_sweep.py --ratios <best> --seeds 0,1,2 ...`
（约 +1 次单跑；这是进 RESULTS.md 的前提）。

### 2.3 通过标准（"全绿"的定义）

1. `tests/test_arr_hybrid_ratio.py` 5/5 通过；
2. 全量 pytest 无**本分支引入**的失败（R3 的 WIP 失败单独归因、单独记录）；
3. `--smoke` 满足 §2.1 判定清单；
4. 远程 8 个 JSON + 帕累托 md 落盘 `benchmarks/results/`，字段齐全（§3），
   1:4 行给出 ≥15% 门限的 PASS / 负结果判定。

## 3. 关键值：测什么、记什么、落在哪

**参考常量**（已写死在脚本里，报告时不得改动）：teacher PPL 基准 **11.8**；
1:4 门限 = 相对 ratio-0 对照改善 **≥15%**（达不到记"回填无效"负结果，
不许事后改口径）；进 RESULTS.md 需 **≥3 seeds**；参考上下文 T=32768
（全阶梯 512 → 1,048,576）。

| 字段（`arr_ratio_pareto.json` 路径） | 含义 | 判读 |
|---|---|---|
| `rows[k].val_ppl` / `val_ppl_per_seed` | seed 均值 / 分 seed | 均值进表；`val_ppl_spread` 看稳定性 |
| `rows[k].ppl_gain_vs_control` | `(ppl_0 − ppl_k) / ppl_0` | **主口径**；1:4 行过 15% 门 |
| `rows[k].gap_closed_vs_teacher` | `(ppl_0 − ppl_k) / (ppl_0 − 11.8)` | 副口径，仅报告 |
| `rows[k].memory.recurrent_state_bytes` | 恒定循环状态（解析式，fp16） | O(1) 项，随比例递减 |
| `rows[k].memory.kv_bytes["16"/"2"][T]` | 保留注意力层 KV @fp16 / @2bit | **O(T) 项**；与状态列永不合并 |
| `rows[k].verdict` | 预注册判定串 | 1:4 行含 PASS / NEGATIVE RESULT |
| `promotion.promotable` | 够不够格进 RESULTS.md | 2 seeds 必为 false |
| （分析值）KV 与恒定状态的交点 T | O(T) 项追平 O(1) 项的上下文 | 帕累托前沿形状的核心读数 |

**产物文件**：`benchmarks/results/arr_ratio_{0,1_8,1_4,1_2}.json`（分比例、
含全 seed 与单次 run 明细）、`arr_ratio_pareto.json`（账本）、
`arr_ratio_pareto.md`（双列帕累托表 + fp16/2bit 两张上下文曲线表，表头自带
"回填后不再是 O(1)"边界声明）。

## 4. 改动明细与设计理由（审阅对照用）

| 改动 | 为什么这样改 |
|---|---|
| `convert_to_arr(model, layer_indices=...)` 子集语义 | 之前只有"全转"一条路。集合 = 只转指定层，其余层的预训练注意力**逐位不变**（同一模块对象、同一权重、同一 KV 路径），保证 ratio>0 行的对照是没被动过的注意力 |
| `use_cache` 只在全转时关闭 | 修一个真 bug：混合模型保留的注意力层真的需要 KV cache，原实现无条件关掉是错的 |
| `select_attention_layers` 中点铺开 | 保留层沿全深度均分（等分桶取中点）。堆在一端会退化成"注意力编码器 + 循环解码器"，那是另一个模型 |
| `recurrent_state_elems` 解析式 + `measured_state_bytes` 实测对账 | 账本无 GPU 也能算；测试对真实 forward 的状态张量逐项核对，防"没人验过的公式" |
| 扫描按 (ratio, seed) 子进程调 `distill_arr.py`，只动 `--keep_ratio` | 转换是原地不可逆的，天然要进程隔离；单旋钮保证控制行与混合行走同一份蒸馏代码、配方不漂 |
| KV 复用 `kv_frontier.kv_bytes`（fp16 / 2bit，含 KIVI scale） | 对自己的 hybrid 用和给竞争对手记账**同一个会计**，不搞双标 |
| 判读门（15%、3-seed）在数字产生前写死 | 防事后挑有利口径；1:4 不达标即负结果入档，不重新协商 |
| `distill_arr.py` 新参数全部默认关闭 + 几何落盘 | 默认路径逐位不变，历史 O-series 数字不受影响；层/头几何写进 `arr_result.json`，扫描侧零手工抄写 |

**5 个测试各钉死什么**：
① 子集转换正确性（该转的成 mixer、该留的 q/k/v/o 逐位 `torch.equal`、
hybrid `use_cache=True`）；② `None`=全转的历史位等价（隐式默认 / 有序 list /
无序 set 三种传法 mixer 参数逐位相等）；③ 解析状态账 vs 真实流式张量逐项
相等；④ 比例规划单调、铺满全深度、不前置堆积、越界层号报错；⑤ 四比例一步
KD 冒烟 + 内存账形状（ratio-0 是唯一零 KV 行、状态递减、2 seeds 不可晋升）。

## 5. 附录：PR 状态（已提交，待评审）

- **PR #2**：https://github.com/everest-an/M1/pull/2
  - head = `iter/o-series-hybrid-ratio`，base = `iter/irregular-streaming-edge`
    （当初的方案 A：diff 干净 = 本线 commit；基线分支已同步推上远程）；
  - 状态 OPEN，**未合并**；正文含动机 / 交付 / 诚实状态（零执行、数字全
    TBD）/ 红线三条 / Phase B 后续。
- 后续顺序：评审 → 合 PR #2 进 edge 线 → 主 worktree 按 §2.1 过测试门 →
  远程全量（§2.2）→ 回填数字与 ≤200 字结论。edge 线进 main 另立 PR，
  不在本任务范围。
