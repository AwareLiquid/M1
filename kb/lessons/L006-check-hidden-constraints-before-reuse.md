---
id: L006
status: ACTIVE
created: 2026-09-09
refs: [HANDOFF.md]
origin: HANDOFF §5.6（迁移自 2026-07 版血泪教训清单）
---

# L006 — 复用历史脚本前先读隐性约束

## Statement

复用他人或历史 baseline 脚本时，先读源码确认非显式约束
（硬编码维度、整除要求、路径假设、环境依赖），再传参运行。

## Trigger

- 给历史脚本传新参数组合时；
- 脚本报错或结果异常但命令行参数看起来合法时。

## Countermeasure

读 argparse 与常量定义段；把隐性约束写成调用命令注释或参数校验。
示例：transformer baseline 硬编码 `n_heads=13`，`--d_model` 必须能被
13 整除（用 832/104，不能用 128）。

## Source

HANDOFF.md §5.6。
