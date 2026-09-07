---
id: ADJ-005
date: 2026-09-01
refs: [benchmarks/results/ADJUDICATION_LOG.md]
---

# ADJ-005 · 2026-09-01 · 决策血缘迁移的"不迁移清单"审计

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

---

---

> 本文件为决策 ADR 工件（一决策一文件，正文不可变；勘误走新 ADJ 并在本文件尾追加指向）。
> 索引见 benchmarks/results/ADJUDICATION_LOG.md。
