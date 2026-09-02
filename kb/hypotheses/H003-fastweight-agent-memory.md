---
id: H003
status: UNDER_TEST
created: 2026-08-29
refs: [docs/PARAMETRIC_MEMORY.md, mt_lnn/parametric_memory.py, benchmarks/results/parametric_memory_summary.json]
---

# H003 — 快权重 (F, z) 矩阵可充当 agent 记忆本体(Mem0 形 API 下四维可测)

## Statement

已证实的快权重记忆(跨窗口召回 0.56 vs attention/LoRA 结构零)不仅是一个
LM 附加能力,而是可以**就是** agent 记忆本体:O(1) 状态、代数单绑定遗忘
(`F ← F − k(kᵀF)`)、bit-exact 快照,在 MemoryAgentBench 四维
(D1 检索 / D2 跨窗口 / D3 冲突消解 / D4 快照迁移)上对比外部存储(BM25)
与结构零对照可测出差异。

## Pre-registered judgment(登记于 PR #11 四维基准;不事后移动)

- D1/D2/D4:mechanism-vs-structural-zero,3 seeds,`publishable()` 门;
- D3(delta 规则可修正绑定):预测 mt_v2_delta 显著高于 mt_v2;
- 外部存储对照:BM25 同题。

## Evidence basis(立项依据,只引不测)

- 跨窗口召回 0.56 vs 0.000(BENCHMARKS.md,多 seed 复制);
- 快照/恢复 bit-exact(HANDOFF §7 引用线);
- 外部存储路线固有 O(n) 存储、进程迁移非 bit-exact(PARAMETRIC_MEMORY.md §Objective)。

## Verdict log(追加式)

- 2026-08-30(全量入库,main b99dd91 判读):D1 mt_v2 0.9896 ✓(lora_only
  0.0026 / delta 0.0013 均塌到冻结线以下);D2 mt_v2 0.0911 唯一非零 ✓;
  D4 快照→磁盘→子进程→恢复 0.0677 ✓(no-restore 对照 0);**D3 全配置
  ≤0.0013 判负**——delta 规则"修正绑定"预测在该预算未复现。
  状态:机制证明(D1/D2/D4),幅度不主张,D3 负结果入档。
