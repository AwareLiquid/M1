---
id: L008
status: ACTIVE
created: 2026-09-11
refs: [notes/overnight/backlog.md, notes/overnight/2026-09-09-morning2.md]
origin: B-1 事件（2026-09-09/10，text_selective 32K 本地 7h 长跑）
---

# L008 — 本机禁止 >1h 的训练/批量任务（CPU 时间片 + swap 双污染）

## Statement

本地工作机（M2 Max，32GB）禁止运行预计超过 1 小时的训练或批量任务。
此类任务一律迁 Kaggle（纯 CPU 批量推 CPU notebook 4 核免费额度；GPU 任务
推 T4）。本机只保留：探针、分析、判读、验收门、编排。

## Trigger

- 计划在本地跑任何预计 >1h 的训练/批量/网格任务时；
- 长任务结束后出现系统级卡顿、应用被杀、swap 高位残留时。

## Countermeasure

- 预注册时写明算力去处；>1h → Kaggle CPU notebook（数据走 dataset 挂载）
  或 T4；通道阻塞则方向标 PAUSED 并浮现替代方向；
- 长跑结束后检查 `sysctl vm.swapusage` 与 `vm_stat`：swap 用满属正常残留，
  重启清空；若不重启，避免立刻再起大内存任务。

## Source

2026-09-09/10 text_selective 32K 本地 7h 长跑：swap 涨至 26.1/26.6GB 用满、
压缩器 66GB 存储、系统级反复换页；停止进程后 swap 残留不清零，需重启恢复。
同日用户定调"≤1h 政策"。
