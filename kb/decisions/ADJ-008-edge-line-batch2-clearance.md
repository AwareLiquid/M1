---
id: ADJ-008
date: 2026-09-06
refs: [benchmarks/results/ADJUDICATION_LOG.md]
---

# ADJ-008 · 2026-09-06 · 边缘证据线批次2清偿:生产者代码 + 6 份判负/加固证据 + 判定轨回填

**调整**:将仅存于未合并分支 `iter/irregular-streaming-edge` 的边缘线剩余
产物按三轨搬入(本 PR 为 ADJ-007/判定轨 #26 之后的第二批):
(1) 机制轨——判定/证据已引用但 main 缺失的生产者代码:`battery_soh_edge.py`
的 LiquidDT/LiquidAD 探针臂、`synth_ct_control.py` 的 mt_lnn_dt/mt_lnn_ad
臂与预注册、`dt_mixture_pilot.py`/`synth_mixture_recipe.py`/
`diagnose_decay_probes.py`、判定测试 `tests/test_liquid_dt.py`/
`test_liquid_ad.py`(ADJ-006"机制代码照常落地"条款);
(2) 证据轨——6 份零主张 JSON+log:`synth_ct_control_ad`(自适应衰减探针
R2'/R1' 双负)、`dt_mixture_pilot`(混合训练试点)、`synth_mixture_recipe`
(预注册正式配方实验,R1 判负)、`diagnose_decay_probes`(根因解剖+锚点
消融判负)、`airquality_irregular_10seed`(Dingling 10 seeds 加固)、
`airquality_irregular_gucheng`(第二留出站跨站 NULL);
(3) 判定轨——RESULTS.md 电池边缘 proven 行按预注册撤回 + air 单域/
Gucheng NULL/部署 profile/合成控制 NULL/1-of-3 域判定 NULL/mt_lnn_ad
终局 NULL 各行 + BENCHMARKS 章节 + HANDOFF §2.10。同源事故(单分支丢档)
第 4 笔(ADJ-003 H002、ADJ-006 #2、ADJ-007 后续)。

**根因证据**:判定轨 PR #26 引用 `benchmarks/battery_soh_edge.py` 的
LiquidDTRegressor,而该探针臂自分叉点 887a4a8 起仅存于未合并分支;
air 加固与 ad/锚点/配方判定的 JSON 均系 2026-08-29/30 产出、仅存未合并
分支,丢档风险与 ADJ-007 同源。预注册规则写死于生产者 docstring 并回显
于 JSON `pre_registered_rule` 字段(本 PR 落地后 main 在库可查)。

**验证**:生产者+测试过 CI `full-test`(本地全量 1509 passed / 4 skipped);
6 份 JSON 零主张入库(`audit-results` 绿);判定轨数字与 JSON 一致,可由
`benchmarks/analyze_irregular_line.py` 一键复算;本 PR 堆叠于批次1
(merge/evidence-closure-batch1, PR #30)之上,依赖其先行合入。

---

> 本文件为决策 ADR 工件（一决策一文件，正文不可变；勘误走新 ADJ 并在本文件尾追加指向）。
> 索引见 benchmarks/results/ADJUDICATION_LOG.md。
