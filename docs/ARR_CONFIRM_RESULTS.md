# ARR 确认跑证据轨 — 6-seed 稳定化配方全曲线（2026-09-06 A100）

> **轨道**：evidence 证据轨（PR #21 三轨拆分）。本页零主张——只登记数据来历、
> 协议、环境与臂清单；数值账本在 `benchmarks/results/rebuilt/`（由
> `rebuild_arr_ledger.py` 以冻结判定逻辑重建），判定行由判定轨 PR 承载。
> 对应预注册草案：分支 `iter/o-series-hybrid-ratio` 的 `docs/ARR_CONFIRM_PREREG.md`。

## 1. 为什么这批（替换 8/31 的 9/12 臂）

8/31 批次（commit 1ed4625，PR #22 入库的 9/12 臂）运行于两个已知缺陷之下：
① 子集运行 bug（后经 4d4c48f 修复）；② `--warmup_b` 旋钮从未在真实路径生效——
stage-A 优化器缺 `initial_lr`，任何 warmup>0 必 KeyError（修复于
`iter/o-series-hybrid-ratio` e265875，随 PR #2 合并）。8/31 批次 1:4 s2 两次
NaN/714.9、1:2 系统性 1365/16136、ratio-0 对照 220.1——按 ARR_CONFIRM_PREREG
"8/31 教训：发散臂与子集 bug 都必须在跑之前写清处置"的约定，本批以稳定化配方
重出全曲线，**8/31 的 9/12 臂不再引用**（其副本仍存 `benchmarks/results/arr_ratio_*.json`
与 `benchmarks/arr_ratio_out/pre_pb_0830/`，作为历史留档）。

## 2. 协议（与草案一致，唯一变更 = warmup）

- 配方：TinyLlama-1.1B-Chat 教师；steps_a 1000 / steps_b 2000；seq 512；
  batch 1 × grad_accum 4；lr_a 1e-3 / lr_b 5e-4；d_proto 96 / proj_rank 384 /
  fw_dim 96；**--warmup_b 200**（stage B 线性 warmup，占 10%）；grad clip 1.0 在位。
- 臂：4 比例 {0, 1:8, 1:4, 1:2} × 6 seeds {0..5} = 24 臂。
- 设备：单一 A100-80GB，bf16 自动档（dtype 由脚本按 cuda+bf16 能力自动选择）。
- 发散规则 R-rerun：NaN 或 PPL>500 判发散，每臂允许重跑 1 次；重跑仍发散记
  DIVERGED.marker，均值只计有效种子。
- 环境三件套（R1 记录）：torch 2.5.1+cu121 / transformers 4.57.3 / datasets 4.8.5
  （= requirements.lock.gpu），Python 3.10.12。**R1 未触发**：transformers
  4.57.3 可消费循环层 cache 槽，全程未用 `--no_cache`。

## 3. 臂清单

| 比例 | 有效臂 | 发散臂（2 次尝试后） |
|---|---|---|
| 0 | 6/6 | — |
| 1:8 | 6/6 | — |
| 1:4 | 6/6 | — |
| 1:2 | 3/6（s0/s2/s4） | s1（550.5）、s3（1335.6）、s5（2274.7）→ `DIVERGED.marker` |

逐臂原始产物：`benchmarks/arr_ratio_out/ratio_<tag>_seed<n>/arr_result.json`
（含 teacher_ppl / student_ppl_after_a / student_ppl_final / mixer_params /
args 回显 / 层几何）。编排日志：`benchmarks/arr_ratio_out/confirm_run.log`。

## 4. 账本

`benchmarks/results/rebuilt/arr_ratio_pareto.{json,md}` 由
`python benchmarks/rebuild_arr_ledger.py <repo>` 从逐臂产物重建
（判定逻辑 = 8/29 冻结版零改动；NaN 臂自动剔除）。rebuild 输出
`promotion` 块与各比例均值/增益——判读由判定轨 PR 承担，本页不转述数字。

## 5. 登记

- 假设：H002（`kb/hypotheses/H002-arr-backfill-pareto.md`）
- 前置判负/零主张纪律：本批数字仅经预注册口径（1:4 ≥15% 门）判读；
  判定行写入 RESULTS.md 属判定轨，另立 PR。
- owner：EverestAn；本次执行：2026-09-06 A100-80GB（租期 24h 内完成 21/24 有效臂）。
