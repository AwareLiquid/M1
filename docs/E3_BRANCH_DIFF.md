# E3 实现差异报告：consciousness-m1-v2 分支 vs main

> 分析日期：2026-08-15 · 分支：`origin/experiment/consciousness-m1-v2`（未合并，只做 diff 分析）

## 核心结论

分支上的 **parity 3/3 满分**（commit `4341694`）的实现 = 两个机制**同时开启**：
**输入相关转移（input-dependent decay）+ 负特征值（negative eigenvalues）**，
两者缺一不可（Khavari et al., arXiv 2508.07395 的标题结论）。

main 上 `selective_decay`（config.py:155）是两者的 **superset**（每步输入驱动的带符号
转移），已含负特征值能力。因此 **main 已经拥有分支的 parity 能力**，无需合并分支。

## 分支 2×2 消融表（5000 步 × 3 seeds，训 ≤24 测 25-64）

| 配置 | in_dist | extrapolate | 特征值范围 |
|---|---:|---:|---|
| legacy_neither | 0.342 | 0.000 | [+0.000, +0.994] |
| negative_only | 0.339 | 0.000 | [+0.000, +0.994] |
| input_dep_only | 0.477 | 0.000 | [+0.000, +0.999] |
| **both_khavari** | **1.000** | **1.000** | **[-1.000, +0.992]** |

**关键**：both 配置不仅 in_dist 满分，还做到**完美长度外推**（训 ≤24，测 25-64 全对）。
这是 E5 长度外推实验的直接依据——main 的 selective_decay 应复刻此结果。

## 失败根因（分支 commit 4341694 的代数检查）

- 液核转移 `decay = exp(-dt/tau)`，tau 是学习参数 → 对角 / 输入无关 / 特征值严格正
- 满足 Sarrof NeurIPS 2024 Thm 2 的非负前提 → 理论上做不了 parity
- **第一版只补负特征值 → 失败**：常数 a=-1 只给 `(-1)^t`（**位置**的奇偶），
  永远不是**比特**的奇偶——符号必须由输入驱动
- Khavari et al. 标题即结论：两者缺一不可

## 边界（不得越界声称，分支已正确标注）

- 这**不是**逃逸 TC⁰ —— parity 本就 ∈ TC⁰，修好的是**参数化缺陷**而非复杂度类
- **A5/S5 仍解不掉**（commit `bd19580`）——需要**非对角输入相关转移**
  （Merrill Thm 5.2 / IDS4 / DeltaProduct），现有对角修复不提供
- 玩具规模，LM 规模是否伤 PPL 未验证（分支最新 commit `ee42697` 已测 46M/6000步
  PPL 224.21 vs 228.69，状态追踪改动不伤 PPL）

## A5 状态追踪任务（分支 `benchmarks/state_tracking_a5.py`）

这是 **E2 任务桥接的现成起跑线**（LSTM 正对照已就位）：

| 配置 | in_dist | in_dist_tok | extrapolate | extrapolate_tok |
|---|---:|---:|---:|---:|
| LSTM 对照 | 0.958 | 0.992 | **0.390** | 0.819 |
| liquid_legacy | 失败（loss 3.84） | — | — | — |

- LSTM 在 A5 上 in_dist 学到 0.958，但外推崩溃到 0.39 —— **A5 是真正的
  状态追踪外推挑战，LSTM 也做不好**
- 当前对角修复解不了 A5 —— 需要非对角输入相关转移，这是 MT-LNN 的**差异化机会**

## 分支独有资产（main 没有的，若 E2 需要可 cherry-pick）

| 文件 | 内容 | 价值 |
|---|---|---|
| `benchmarks/state_tracking_parity.py` | parity 消融 + 长度外推协议 | E1 长度外推评估的直接参考 |
| `benchmarks/state_tracking_a5.py` | A5 状态追踪 + LSTM 对照 | E2 任务桥接起跑线 |
| `benchmarks/results/state_tracking_{parity,a5}.json` | 完整结果 | 已有基线 |
| `mt_lnn/adaptive_compute.py` | 自适应算力/可学停机 | P2 效率方向 |

## 行动指引

1. **E1**：main 的 `--selective_decay` 应复刻分支 both_khavari 的 1.000/1.000（验证中）
2. **E5**：把分支的"训 ≤N 测 N+1..2N-4"长度外推协议集成到 main（当前 reasoning_depth.py
   是固定 seq_len，无外推评估）
3. **E2**：A5 任务是现成的"状态追踪外推"基准，LSTM 已证明这是真挑战；
   MT-LNN 要解它需要**非对角输入相关转移**（下一架构演进方向）
