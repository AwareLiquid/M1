---
id: L007
status: ACTIVE
created: 2026-09-09
refs: [HANDOFF.md]
origin: HANDOFF §5.7（迁移自 2026-07 版血泪教训清单）
---

# L007 — 公开仓库披露红线

## Statement

自述"勿推公开仓库"或含内部信息的文档，不得被公开仓库 git 跟踪；
向公开仓提交前必须做披露审计。

## Trigger

- 提交涉及私有仓库路径、内部文档引用、未完成项与风险清单时；
- 迁移文档跨公私仓库时。

## Countermeasure

提交前 grep 私有仓库路径与敏感文件名；`git rm` 不清除历史，
迁移时需评估历史可见性。当前已知红线：PUBLICATION_READINESS.md
（已迁私有仓 AwareLiquid-Web `internal/`）；本 HANDOFF 本身在公开仓，
含未完成项与风险，提交前确认内容可公开。

## Source

HANDOFF.md §5.7（PUBLICATION_READINESS.md 曾长期被跟踪的历史教训）。
