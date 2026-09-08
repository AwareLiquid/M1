---
id: ADJ-011
date: 2026-09-08
refs: [docs/PR_MERGE_POLICY.md, benchmarks/results/ADJUDICATION_LOG.md]
---

# ADJ-011 — main 分支保护放宽批准门槛 (1 approval → 0)

**决策**：`main` 分支保护的 `required_approving_review_count` 从 1 降为 0。
PR 仍需提交（`required_pull_request_reviews` 对象保留），CI 状态门
（strict：`full-test` + `audit-results`）、linear history、禁止 force-push/
删除、`enforce_admins` 全部保留不变。

**理由**：仓库实际为双账号运营（everest-an 开发、AricRedemption 复核），
GitHub 平台硬规则禁止 PR 作者自批，导致单账号时段（如 A100 训练日）开发
PR 卡死等待第二身份，历史 3 次都是人工到浏览器点批。CI 三件套门禁
（full-test / audit-results / lint-and-import）+ 三轨纪律（机制/证据/判定）
已是实质性质量闸门；批准计数从 1→0 只移除"第二身份在场"这一排程约束，
不削弱内容门禁。

**约定（非机器强制，靠纪律）**：
- everest-an 的 PR：CI 绿即可合并（AI 助手可代为执行）；
- AricRedemption 的 PR：合并前仍需 Everest/Aric 人工复核，不因规则放宽
  而跳过评审；
- 若未来出现新协作者且单账号流程被滥用（绿测直合不合格内容），恢复
  `required_approving_review_count=1` 并另行 ADJ 登记。

**替代方案（被否）**：建 AI 评审账号并加入协作者——需注册新账号 + 常驻
写权限 token，安全暴露面更大；当前双账号纪律已覆盖复核需求。
