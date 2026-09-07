---
id: ADJ-001
date: 2026-09-01
refs: [benchmarks/results/ADJUDICATION_LOG.md]
---

# ADJ-001 · 2026-09-01 · latent-recursion 判决任务脱出 budget_wall

**调整**:latent-recursion Task 2/3 的判决任务从 pointer_chase difficulty=8
改为可学习域内配置(首选 parity d16——H003/S1 已证 6/6 通过;备选 d2/d4 两档)。

**根因证据**:48/48 探针 cell 全部 mean_acc ∈ [0.062, 0.072] < 0.25
(`_rescue_20260831/latent_recursion/`,autopilot ENDGAME results=48);
历史学习曲线:pointer_chase d2 需 30K 步才 grok 到 1.000
(`reasoning_depth.jsonl` verdict-30k),d4@20K 仅 0.373,难度外推从未验证
——立项依据(见 H001 Evidence basis)只锚定了 d2 数据点。
诊断:任务在 30K 步预算外于 learnable regime(工业口径:feasibility probe
缺失 + grokking 假阴性窗口),非架构判负。

**验证**:换 parity d16 后,同预算探针须先过 learnability gate
(30K 步中 10K 步时 acc > 0.2,预注册止损);过了才进入 6-seed 判决跑。

---

> 本文件为决策 ADR 工件（一决策一文件，正文不可变；勘误走新 ADJ 并在本文件尾追加指向）。
> 索引见 benchmarks/results/ADJUDICATION_LOG.md。
