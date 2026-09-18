---
id: L005
status: ACTIVE
created: 2026-09-09
refs: [HANDOFF.md]
origin: HANDOFF §5.5（迁移自 2026-07 版血泪教训清单）
---

# L005 — 单卡显存预算单占

## Statement

单 GPU 上已有训练任务占用大部分显存时，不并行启动第二个训练任务；
新任务排队或换节点。

## Trigger

- 本地小显存卡（如 8GB 级）上已有任务运行时；
- OOM 原因不明时先查并发占用。

## Countermeasure

启动训练前 `nvidia-smi --query-compute-apps=pid,used_memory` 查占用；
确认剩余显存 ≥ 新任务峰值再启动。

## Source

HANDOFF.md §5.5。
