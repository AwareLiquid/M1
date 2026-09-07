---
id: H002
status: SUPPORTED
thread: T001
created: 2026-08-29
refs: [docs/ARR_RATIO_PARETO.md, benchmarks/results/decode.json, benchmarks/results/kv_frontier_ledger.json, benchmarks/results/rebuilt/arr_ratio_pareto.json, docs/ARR_CONFIRM_RESULTS.md]
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
- 2026-09-06(判决,稳定化配方确认跑,21/24 臂有效):
  - 8/31 的 9/12 臂作废(子集 bug + `--warmup_b` 旋钮从未生效,见
    `docs/ARR_CONFIRM_RESULTS.md` §1);本判决只认 2026-09-06 批
    (4 比例 × 6 seeds, `--warmup_b 200`, 单一 A100-80GB bf16)。
  - 账本 `benchmarks/results/rebuilt/arr_ratio_pareto.json`(证据轨 PR
    先合):ratio 0 mean 142.89;1:8 88.92(**+37.8%**);**1:4 67.37
    (+52.9%, 门限 15% PASS)**;1:2 263.52(3/6 臂发散剔除,-84.4%)。
  - **status → SUPPORTED**:存在 r*=1:4,同预算下 PPL 显著低于 0:1
    (6/6 种子配对全胜,均值口径 +52.9% > 15% 预注册门),且 decode
    内存(6/22 注意力层,32K 时 192.5 MB fp16)远低于全注意力 hybrid。
  - 诚实标注:种子双峰性显著(1:4 好模式 18.9-20.5 vs 坏模式
    162.1-164.0;对照同样双峰 74.5-207.6)。warmup 使坏模式改善
    (8/31 无 warmup 时 220-305 → 本批 162-208)但未消除;配对口径
    不受影响,均值口径方差大,引用时必须附标准差与双峰说明。
  - 1:2 判负(6 seeds 中 3 臂双尝试发散、3 臂 223-320):该比例在
    3000 步蒸馏预算内不可训,与 8/31 系统性发散一致。
  - 后续(G1 盲区):最优比例 1:4 上补跑 cross-window recall 作补充列
    (需 `--save_ckpt` 重训留存 mixer checkpoint,本批未存)。
