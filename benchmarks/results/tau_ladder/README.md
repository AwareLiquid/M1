# tau_ladder 结果 — B-2 τ 阶梯 × parity 探针（A100 通道）

> 通道：**A100 服务器**（2026-09-16，用户指令接管 Kaggle 通道——协议不变，仅换算力通道；
> 原始预注册见 `notes/overnight/backlog.md` B-2 条目）。
> 文件：18 个配置 JSON（`tau_probe_ladder{on|off}_d{1|2|4}_s{0|1|2}.json`）
> + `tau_ladder_verdict.json`（judge_verdict 预注册判据）。

## 判决摘要

| 项 | 值 |
|---|---|
| h_supported | **false** |
| gain_off (d4−d1) | 0.0001 |
| gain_on (d4−d1) | 0.0001 |
| delta_gain | 0.0（阈值 2σ=0.0002） |
| 诊断 | **flat**（"τ 阶梯增益与 OFF 无可辨差异"） |
| 配对符号检验 | p=1.0（2 胜 1 负，噪声） |
| 全部 18 配置 | mean_acc ≈ 1.0（全饱和） |

## 诚实标注（重要）

**此 flat 是天花板效应，不是"τ 阶梯无效"的干净判负**：parity d16 在 depth=1 即达
acc 1.0，深度维度无区分度（depth 1 = depth 4 = 1.0），τ 阶梯自然"无可辨增益"。
与 M2 mod_chain 判负同因（"任务对 probe 规模过简单 → 天花板效应，无法区分机制"）。

**结论：该协议无法检验 H004**（"τ 阶梯使深度增益更大"）。要真正检验 τ 阶梯的深度利用
效率，需要 depth=1 学不会、depth=4 能学会的**硬化任务**（更高 difficulty 或
pointer_chase 类查表域），判据同款预注册后重测。
