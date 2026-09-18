---
id: L002
status: ACTIVE
created: 2026-09-09
refs: [HANDOFF.md, RESULTS.md]
origin: HANDOFF §5.2（迁移自 2026-07 版血泪教训清单）
---

# L002 — 不同口径的数字不混用

## Statement

参数量、训练步数、精度（fp32/AMP）、数据集任一不同的两组实验结果，
不得进入同一张对比表或支撑同一条主张；口径差异只能文字注明，不能数值并排。

## Trigger

- 跨实验、跨论文、跨仓库搬数字进主表时；
- 用旧基线数字为新架构背书时。

## Countermeasure

主表数字必须同口径同协议；确实需要对齐口径时，先跑同 setup 的对照实验
（这正是 E1 立项的原因）。历史混口径结论一律标注限定语或撤回。

## Source

HANDOFF.md §5.2；RESULTS.md RETRACTED 区（×42 撤回事件的直接教训）。
