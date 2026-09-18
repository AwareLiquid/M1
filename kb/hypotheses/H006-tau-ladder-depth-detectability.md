---
id: H006
status: DORMANT
thread: T001
created: 2026-09-18
refs: [benchmarks/results/tau_ladder_k32/tau_ladder_verdict.json, benchmarks/results/tau_staircase/, notes/overnight/backlog.md, kb/lessons/L012-single-seed-depth-gain-grokking-artifact.md, kb/hypotheses/H001-stack-depth-monotonic.md]
---

# H006 — τ 阶梯（连续时间步长）使"深度→准确率"增益在可学区间内可检

**撞号追溯登记（本件存在的直接原因）**：backlog B-2/B-2′、2026-09-16/17 晨报、
commit `0f1dbb7`/`ab10bd5`/PR #62 中所有"H004"引用实指本工件（τ 阶梯深度可检性，
T001 线）；kb 既有 H004 为 1M 流式记忆（T002，2026-09-06 立项），两者无关。
H004 号已被占用，故追溯立为 H006；历史文档不回写（工件不可变纪律，先例 ADJ-006
追溯登记）。今后 overnight 文档必须用 H006 指代本假设。

## Statement

在存在深度敏感带（depth=1 学不会、depth=4 学得会）的任务域内，resonance τ 阶梯
（ladder=on，把连续时间步长离散化为校准阶梯）能把深度增益从噪声中分离出来，
使 H001 的核心/整块单调性主张可判定。

## Pre-registered judgment（写死于 B-2′ 立项，2026-09-16）

本假设可检 ⇔ 存在任务点使 d1 显著低于"学得会"线（0.55）且 d4 显著高于该线；
增益判定 ⇔ ladder=on 相对 off 的 depth 斜率差在 ≥2 个独立 seed 下同号且过
2σ（cooked 后读数，遵守 L012）。任一环节不满足 → 记 flat/underpowered，
不得以单 seed 欠火读数转正（L012）。

## Evidence basis

H001 的 parity 域筛选证明该族"全配置共享天花板或墙"，无法提供敏感带
（backlog B-2′ 结案：d16 天花板 + d32 深度平坦，parity 家族无法检验本假设）；
pointer_chase medium 域 B-2″ 探雷（进行中）即为寻找敏感带的授权续段。

## Verdict log

- 2026-09-17（追加）：staircase 校准 k=32 处出现"深度增益转正"形状
  （读数见 `benchmarks/results/tau_staircase/`）——事后定性为 grokking
  位置伪影（单 seed、欠火窗），按 L012 降级为线索，不得作效应量引用。
- 2026-09-18（追加）：PR #62 k=32 18 配置全矩阵正式判定 **flat/underpowered**
  （正向读数低于 2σ 判据阈值，数字见 `tau_ladder_k32/tau_ladder_verdict.json`）。
  判定：非证伪（检验功效不足），转 **DORMANT**。
- 复燃条件（任一）：① B-2″ 判 BAND_HIT（pointer_chase medium 存在
  d1<0.55≤d4 敏感带）→ 在该域以 ≥2 seed、cooked 判据重试本假设；
  ② TPU core audit 放行后出现更便宜的全矩阵复核通道，使功效补足成本可接受。
