---
id: ADJ-006
date: 2026-09-04
refs: [benchmarks/results/ADJUDICATION_LOG.md]
---

# ADJ-006 · 2026-09-04 · 追溯登记三笔"无结论合并"违规 + 门禁三轨拆分

**调整**:登记 PR #9/#10/#12 为结论级违规存量,限期清偿(owner:
AricRedemption);PR_MERGE_POLICY 增设三轨拆分(mechanism / evidence /
verdict)与结论级门禁(§5);kb 假设 UNDER_TEST 补 owner/deadline 强制
字段;PR 模板增"结论状态"节。另据证据轨条款,将 #2(iter/o-series-hybrid-ratio)
已抢救的 H002 9/12 臂数据以 evidence PR 落入 main(消除单分支副本)。

**根因证据**:三笔均 2026-08-30 合并,早于本政策生效日 2026-09-01(PR
#16),当时口头纪律只禁"无证据主张"(RESULTS RETRACTED 纪律),不禁
"无判定落地"——缺口由 owner 于 2026-09-04 确认为违规。对标业界治理:
vLLM 合入门禁只验机制不验结论;HF 新架构经玩具规模 CI 验证即合入、
效果主张不进代码审查;FDAAA/临床试验把"无结论"设为合法生命周期状态、
把"超期无结果且不声明"定为违规。据此,治理对象修正为"无证据主张"与
"无状态搁置",机制代码照常落地、判定内容拆轨后置。

**清偿路径(关闭条件 = kb verdict log + RESULTS 状态位回填)**:
- #12 HF 原生:Phase B 四项在主 worktree 执行(全量 pytest + hf_load_bench
  + SERVING_ROADMAP §7 + README 三行定稿),零 GPU,下一合并窗口前;
- #10 modern-trunk:四配置 × ≥3 seeds 筛选判定(CONFIRM/NULL)回填
  MODERN_TRUNK.md §3,下一合并窗口前;筛选数字维持 screening-only 红线;
- #9 latent-recursion Task 3:按 ADJ-001 换 parity d16、过 learnability
  gate(10K 步 acc>0.2)后 6-seed 判决,列入下一次 GPU 会话(与 H002
  缺 3 臂补齐同批,ADJ-004 分片纪律)。

**验证**:三笔清偿落地后由合并窗口门禁复核 kb 状态与 RESULTS 一致;
此后新 PR 按"结论状态"节审查,ADJ-007 起不得再出现无 owner 无期限的
判定欠账;证据轨 PR(audit-results 绿)不得被判定欠账阻塞。

---

> 本文件为决策 ADR 工件（一决策一文件，正文不可变；勘误走新 ADJ 并在本文件尾追加指向）。
> 索引见 benchmarks/results/ADJUDICATION_LOG.md。
