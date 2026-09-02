---
id: T001
status: EXPLORING
created: 2026-07-28
refs: [docs/ROADMAP_M2.md, benchmarks/reasoning_depth.py, docs/LATENT_RECURSION.md]
---

# T001 — M2 推理深度线(递归深度 = 用计算换参数)

## Concern

ROADMAP_M2 核心命题之二:HRM/Geiping 证明循环递归深度可替代参数/层数,
M1 的液体核心天生递归——该契合点未被利用。本线程覆盖:单环深度扫描、
stack×core 对照、τ 阶梯、anytime 前沿、全局头配额(C1)。

## Spawned hypotheses

- H001 — stack 循环深度使深度-准确率单调上升(core 不升)

## Log

- 2026-07-28: pointer_chase 深度扫描立项(depth_stack_k4 / p0c-run1)。
- 2026-08-16: 云 A100 parity 外推规律确立(外推极限 ≈ 训练长度 × 8)。
- 2026-08-29: PR #9 立项,协议/驱动/账本就绪。
- 2026-08-31: Phase B Task 2 判决回填 → H001 探针 budget_wall(见 ADJ-001)。
