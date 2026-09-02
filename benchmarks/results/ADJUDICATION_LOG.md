# ADJUDICATION_LOG — 决策日志(人写,追加式)

> 每条记录一个关键决策/调整:调整了什么、根因证据、如何验证。
> 与 `kb/hypotheses/`(假设档案)、`docs/DECISION_TRACE_SPEC.md`(字段契约)配套。
> 纪律:必写"根因证据"(文件/数据/commit),必写"验证"(改后怎么确认是对的)。

---

## ADJ-001 · 2026-09-01 · latent-recursion 判决任务脱出 budget_wall

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

## ADJ-002 · 2026-09-01 · verdict n_rows=1 残留重跑

**调整**:在 PR #9 分支用完整 48 行行 JSON 重跑 `write_verdict`
(数据逐配置原子写未受损,判负标准未动)。

**根因证据**:rescue 目录 `latent_recursion_verdict.json` n_rows=1、
depths=[4]、gate tag=`stack_d4_vs_d4` —— core 单模式进程收尾
`judge_decision` 无条件访问 `tiers['stack']` → KeyError 崩出残留;
已由 commit 304ed35 修复(has_stack 保护 + no_stack_arms 降级)。

**验证**:重跑后 n_rows=48、depths=[1,2,4,8]、diagnosis 仍为 budget_wall
(内容不变,结构修正);回归测试 `test_latent_recursion_driver.py` 两例通过。

## ADJ-003 · 2026-09-01 · 采纳决策血缘规范(kb/ + DECISION_TRACE_SPEC)

**调整**:引入 kb/ 假设层(H/T 工件)、`docs/DECISION_TRACE_SPEC.md`
血缘字段、本日志;**不引入** aexp 运行时(signac/Claude Code hook/kb 全量模板)。

**根因证据**:交接丢档事故(REGISTRY 回收记录:"潜空间递归判决实际存在
但交接时以为丢失")暴露假设层缺位;pointer_chase d8 误立项暴露
"决策依据只存在于当时会话记忆"。外部参考:HEP(协议)、aexp(工程实现,
560 测试通过但 Beta + 单人 + signac 3.0 悬崖 + hook 仅 Claude Code 生效)、
PROV-AGENT(W3C PROV 语义)。

**验证**:新判决级 JSON 自 2026-09 起带 `lineage` + `protocol` 字段;
合并窗口门禁(PR #14 落地后)校验 H→JSON 链完整;首批 H001–H003 建档。

## ADJ-004 · 2026-09-01 · arr Phase B 缺臂补齐判据钉死

**调整**:H002(注意力回填比例)判读推迟至 12/12 臂齐;缺 3 臂的补齐
用分片预算重排(单实例 ≤2 臂),不再单实例全押。

**根因证据**:Phase B 抢救入库 9/12 臂(A100 提前回收,commit 1ed4625);
单实例 12 臂全押 = 一次性敞口等于全部实验价值。教训与 HANDOFF §5.3
"长跑必须 checkpoint/resume"同源:不可靠算力上的无断点长跑。

**验证**:补臂采用 resume-safe 脚手架(`latent_recursion.py` 的
pending_configs/原子写模式);单实例回收最大损失 ≤2 臂;齐臂后按 H002
预注册标准判读,不降低门槛。

## ADJ-005 · 2026-09-01 · 决策血缘迁移的"不迁移清单"审计

**调整**:对 aexp 38 个模块 + HEP/PROV-AGENT 全机制逐项审计,确认未迁移
项分为"有据不迁移"与"观察名单"两类(见下)。

**根因证据**:aexp 源码逐模块读检(workpool/airgapped/linklease/sandbox
等);对照 M1 实际算力形态(Kaggle 会话静态切分、AutoDL 单机多进程、
有互联网)。

**不迁移且有据(每条可复审)**:
- `airgapped/`(SSH 中继):M1 算力均有互联网,Kaggle 还能 API 回传——问题不存在;
- `hooks/` + `install.py` + `mcp_server.py`:强制点已收敛到合并窗口门禁,
  平台耦合(Claude Code)与弊端清单第 1 条同源;
- `queue.py` + signac run 范式:M1 用 shell autopilot + Kaggle 静态分片,
  引入队列框架是替换而非增强;
- `sandbox.py`/`jupyter.py`/`trackers/wandb`:M1 探索性工作在
  kaggle_kernels/ 与探针脚本,无 notebooks 约定;wandb 保持"非事实源"纪律;
- `runs_index.py`:aexp 自己已废弃(被 ledger 取代);
- HEP 信念 P(H) 数值化:单人仓无多 agent 消费者,markdown 追加式
  Verdict log 足够;hash-chain 事件日志:git commit 已提供等价防篡改;
- PROV-AGENT 全本体:单人仓过重,SPEC 已取其命名语义。

**观察名单(触发条件出现即升级)**:
- `utils/linklease.py` + `workpool.py`(NFS 安全 link() 租约 + 无守护
  work-stealing):**触发条件 = ADJ-004 的静态分片升级为多 Kaggle 会话
  动态认领**(多个会话从一个共享清单互相认领配置、死亡会话的认领过期
  被同伴回收)。当前 48-cell 靠 runbook 手工切分会话,暂无用武之地;
- ledger 终态自动投影:触发条件 = 结果开始落在 `benchmarks/results/` 之外
  的目录(当前 save_row 已直写,投影层是重复);
- `backlinks.py` + frontmatter 解析依赖:触发条件 = kb/ 假设数 >10
  (当前 3 个,grep 足够)。

**验证**:上述触发条件任一成为现实时,对应条目从本日志升级为 ADJ 决策
并实施;每季度(合并窗口)复审一次本清单。
