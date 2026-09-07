---
id: ADJ-004
date: 2026-09-01
refs: [benchmarks/results/ADJUDICATION_LOG.md]
---

# ADJ-004 · 2026-09-01 · arr Phase B 缺臂补齐判据钉死

**调整**:H002(注意力回填比例)判读推迟至 12/12 臂齐;缺 3 臂的补齐
用分片预算重排(单实例 ≤2 臂),不再单实例全押。

**根因证据**:Phase B 抢救入库 9/12 臂(A100 提前回收,commit 1ed4625);
单实例 12 臂全押 = 一次性敞口等于全部实验价值。教训与 HANDOFF §5.3
"长跑必须 checkpoint/resume"同源:不可靠算力上的无断点长跑。

**验证**:补臂采用 resume-safe 脚手架(`latent_recursion.py` 的
pending_configs/原子写模式);单实例回收最大损失 ≤2 臂;齐臂后按 H002
预注册标准判读,不降低门槛。

---

> 本文件为决策 ADR 工件（一决策一文件，正文不可变；勘误走新 ADJ 并在本文件尾追加指向）。
> 索引见 benchmarks/results/ADJUDICATION_LOG.md。
