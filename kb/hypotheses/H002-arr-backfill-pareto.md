---
id: H002
status: PROPOSED
thread: T001
created: 2026-08-29
refs: [docs/ARR_RATIO_PARETO.md, benchmarks/results/decode.json, benchmarks/results/kv_frontier_ledger.json]
---

# H002 — 存在中间注意力回填比例,在质量-内存帕累托上同时优于 0:1 与 3:1

## Statement

ARR(O 系列,纯递归)全去掉注意力省 KV 但 PPL 付出 2.67× 同同步数代价;
hybrid(全注意力,M 系列)质量好但 KV O(T) 增长。存在某个中间回填比例
r ∈ {1:8, 1:4, 1:2}(注意力层数 : 递归层数),其 (质量, 推理内存)
帕累托点同时支配 r=0:1 与 r=3:1。

## Pre-registered judgment(脚手架内写死,PR #2;不事后移动)

四比例 × 3 seeds 同协议(WikiText,统一 token budget)各出 (val PPL, decode 峰值内存):
- 主张成立 ⇔ 存在 r* ∈ {1:8, 1:4, 1:2} 使 r* 的 PPL 显著低于 0:1
  (配对 p<0.05)且 decode 内存低于 3:1 至少 2×;
- 否则 NULL,记录"中间比例无帕累托优势"为负结果,ARR 定位不变。

## Evidence basis(立项依据,只引不测)

- ARR 蒸馏轨迹:25.4 = 2.15× teacher PPL 且随 token 仍降
  (RESULTS.md, Round 2/3 distillation)——质量差距是预算问题,不是结构上限。
- KV 前沿账本(`kv_frontier_ledger.json`):ARR 平线 0.381 MB vs hybrid
  KV@1M 3 GB——内存维度两端差 4 个数量级,中间比例的曲线形状未知。

## Verdict log(追加式)

- 2026-09-01(登记):Phase B 抢救入库 4 比例 9/12 臂(A100 提前回收),
  数据不足以判读;缺 3 臂补齐后按上列标准判。见 ADJ-004。
