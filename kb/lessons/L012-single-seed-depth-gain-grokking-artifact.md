---
id: L012
status: ACTIVE
created: 2026-09-18
refs: [kb/hypotheses/H001-stack-depth-monotonic.md, notes/overnight/backlog.md, docs/COMPUTE_TIERS.md]
origin: B-2' staircase 校准轮（2026-09-17 收割登记，2026-09-18 判读定性）
---

# L012 — 欠火区单 seed"深度增益"读数是 grokking 位置伪影，不是可学性证据

## Statement

在 grokking 型任务上，训练尚未过半程（欠火区）时，"深度↑=增益↑"或反向的
单 seed 读数大概率是**读出时刻的位置伪影**：不同深度配置 grokking 时刻不同，
任一切片上看到的差值随步数窗口正负翻转。因此：
① 探真轮的低保真读数**只能用于筛配置**（R3），不得当作效应量登记；
② 深度增益类结论必须 ≥2 独立 seed 且训练过预注册的 cooked 线后才允许进
verdict log（R4）；
③ 引用此类旧读数须带"位置伪影风险"注记——这是 L009（跨时点对照须同跑）
在**时间轴内侧**的对应物：同一 run 内不同时点的读数同样不可跨档互比。

## Instances

- B-2'（2026-09-17）：staircase 校准 k=32 处 τ 阶梯使深度增益"转正"
  （+0.054 vs ladder-off −0.024，单 seed 40K 步窗）→ 次日 PR #62 全量
  验证 A-E 与正式跑判读为 **flat/underpowered**，+0.054 定性为 grokking
  位置伪影；深度-单调主线维持 undecided。该读数一度被用作"转正式跑"的依据，
  是 R1/R4 与 ADJ-012 复用规则的直接催生事件。

## Enforcement

- 预注册模板 gate 栏须写明 cooked 判据（步数下限/loss 平台确认），探针窗
  未过线一律只出"继续/杀"二值，禁报效应量；
- backlog 信号对账条目引用此类读数时标注 (单 seed, 欠火) 双降级；
- 常驻判读工具（check_progress/judge 系）后续可加"深度增益读数须第二 seed"
  的软提醒（待排期，不阻塞）。
