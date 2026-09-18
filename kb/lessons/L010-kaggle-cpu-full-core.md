---
id: L010
status: ACTIVE
created: 2026-09-16
refs: [notes/overnight/backlog.md B-2, kaggle_kernels/tau_ladder_probe/run_tau_probe.py]
origin: B-2 V14 提速诊断（2026-09-16，用户质询"4 核是否用满"）
---

# L010 — Kaggle CPU 容器满核铁律（4 核 = 4 单线程 lane + 显式环境钉死）

## Statement

Kaggle CPU notebook（4 vCPU）上的并行批量任务必须满足：

1. **lane 数 = 核数**：worker 进程数 = 4。会话是 12h 硬窗，吞吐 = lane 数 ×
   单 lane 步速——lane 不足就是花钱买核不开机（V14 用 3 lane，25% 算力闲置，
   每会话少 ~2.2 配置）。
2. **线程显式钉死，用环境变量**：launcher 必须在 spawn 任何子进程**之前**
   导出 `OMP_NUM_THREADS=1 / MKL_NUM_THREADS=1 / OPENBLAS_NUM_THREADS=1 /
   NUMEXPR_NUM_THREADS=1`。父进程里的 `torch.set_num_threads(1)` 不会传给
   子进程——V14 的 3 子进程各自默认开 4 OMP 线程 = 12 线程挤 4 核。
3. **按配置原子落盘 + resume-safe**：会话截断只损失在途的那一个配置；
   重推同脚本自动跳过已完成配置。

## Instances

- B-2 V14（2026-09-15/16）：3 lane 跑 4 核 + 子进程 OMP 未钉死；
  12h 会话产出 ~6 配置。V15 修为 4 lane × 单线程（预期 ~8.8 配置/会话）。
- 本地 A/B（200 步，M2 Max）：OMP=4 比 OMP=1 快 24%——workload 对线程数
  敏感，"不钉死"既可能超订也可能欠用，唯一稳态是显式 1 线程/lane。

## Enforcement

- 任何新 Kaggle launcher 的 review 检查项：lane 数 == 核数？env 钉死在
  Popen 之前？粒度是否支持 resume-safe？
- 更细粒度拆分（如 probe 加 --ladder 单配置粒度）可以进一步平衡 lane 负载，
  需 bundle dataset 版本升级（v7），按需立项。
