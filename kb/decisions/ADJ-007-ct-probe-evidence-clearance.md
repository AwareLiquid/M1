---
id: ADJ-007
date: 2026-09-04
refs: [benchmarks/results/ADJUDICATION_LOG.md]
---

# ADJ-007 · 2026-09-04 · CT 机制探针判决证据清偿入库(单点丢档第 3 次)

**调整**:将仅存于未合并分支 `iter/process-train` 的 mt_lnn_dt 机制探针
判决证据(`benchmarks/results/synth_ct_control_dt.json` 723 行 + 同名
`.log` 322 行,commit `c503fdc`,2026-08-29)以 evidence 证据轨搬入 main,
零主张、零 RESULTS/BENCHMARKS 变更(判定轨行随后以独立 verdict PR 落地)。
同源事故第 3 笔(ADJ-003 H002、ADJ-006 #2 后续)。

**根因证据**:R1/R2 预注册规则写死于生产者 `benchmarks/synth_ct_control.py`
docstring(main 在库可查)并回显于 JSON `pre_registered_rule` 字段;
5 seeds × 6 架构 × 6 grid 档 + Δt 偏移档,per-seed 值 + config 回显齐备。
判决已出(CLOSED 双判负),且 ADJ-006 明示证据轨优先放行——数据扣在
未合并分支即丢档风险。注:该探针系 kb 制度生效前完成判定,无 H 前身
工件,其判决承载物为 RESULTS.md Null 区行(verdict PR 落地),不补造假设工件。

**验证**:JSON 通过 CI `audit-results` 结构校验;判定轨 PR 合并后,
RESULTS.md Null 区行 + BENCHMARKS.md §6 与本 JSON 数字一致(复核人可在
合并窗口对表)。

---

> 本文件为决策 ADR 工件（一决策一文件，正文不可变；勘误走新 ADJ 并在本文件尾追加指向）。
> 索引见 benchmarks/results/ADJUDICATION_LOG.md。
