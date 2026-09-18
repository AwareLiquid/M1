---
id: L011
status: ACTIVE
created: 2026-09-16
refs: [notes/overnight/backlog.md B-2, kaggle_kernels/tau_ladder_probe/kernel-metadata.json]
origin: B-2 V14 僵尸事件（2026-09-15/16，API 推送缺 dataset 挂载，4 秒崩溃 + RUNNING 假象 12h）
---

# L011 — kernels status RUNNING ≠ 脚本存活：推送后必须验证启动日志

## Statement

`kaggle kernels status` 的 RUNNING 可能是**僵尸/队列假象**：B-2 V14 的脚本
在 4 秒内即崩溃（metadata 缺 `dataset_sources`，私有 bundle 未挂载），但
status 端点连续 12+ 小时报告 RUNNING。**唯一可信的存活判据是启动日志**。

## Enforcement

- 任何 Kaggle kernel 推送后 **4 分钟内**必须拉取启动日志验证：
  `kaggle kernels output <owner>/<slug> -p /tmp/check` —— 若脚本在数十秒内
  崩溃，日志会立即可拉（V14/V15 实例）；若仍在 RUNNING 且日志不可拉，
  视为"已越过快速崩溃窗口"，间隔 30min 再抽查。
- API 推送的 metadata **必须自查** `dataset_sources` / `kernel_sources` 是否
  覆盖脚本依赖的所有挂载（UI 会话手动挂载的数据集不会自动进入 API 推送）。
- 长跑 kernel 的吞吐核对：以日志内步进速率为准，不以状态端点计时。

## Instances

- B-2 V14（2026-09-15 23:04）：4 秒崩溃 + RUNNING 假象 12h，一夜零推进。
- B-2 V15（2026-09-16 10:44）：重写 metadata 重蹈覆辙（漏 dataset_sources），
  4 秒同因崩溃——同一晚两连击。
