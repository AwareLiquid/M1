---
id: L003
status: ACTIVE
created: 2026-09-09
refs: [HANDOFF.md, kb/decisions/ADJ-004-arr-phase-b-arm-discipline.md]
origin: HANDOFF §5.3（迁移自 2026-07 版血泪教训清单）
---

# L003 — 长跑必须 checkpoint/resume

## Statement

任何超出单会话时长（Kaggle 12h 上限、SSH 断连、机器睡眠、进程重启）
的训练/评测长跑，无断点恢复支持就禁止启动。

## Trigger

- 预计运行时间超过所在算力会话剩余时长时；
- 依赖远程 SSH 会话或共享机器存活时。

## Countermeasure

启动前确认脚本支持 `--ckpt_every N` + `--resume`（如
`benchmarks/scaling_comparison.py`、`latent_recursion.py` 的
pending_configs/原子写模式）；不支持就先补断点逻辑再跑。
批量任务采用分片预算，单实例敞口设上限（ADJ-004 纪律）。

## Source

HANDOFF.md §5.3；ADJ-004（A100 提前回收损失臂的同源教训）。
