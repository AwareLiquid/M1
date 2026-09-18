---
id: L004
status: ACTIVE
created: 2026-09-09
refs: [HANDOFF.md, .gitignore]
origin: HANDOFF §5.4（迁移自 2026-07 版血泪教训清单）
---

# L004 — 大二进制产物不进 git

## Statement

模型 checkpoint、权重、原始数据集等大二进制文件（单文件可到 GB 级）
只存本地/服务器恢复路径，不进版本库；仓库只提交结果 JSON 与日志摘要。

## Trigger

- 准备 `git add` 任何 `.pt`/`.bin`/`.safetensors`/数据集文件时；
- 想用 git 备份训练产物时。

## Countermeasure

大文件路径写入 `.gitignore`；恢复所需信息以 README/命令形式记录，
产物本身留在算力节点的固定目录。

## Source

HANDOFF.md §5.4。
