# DECISION_TRACE_SPEC — 结果 JSON 血缘字段规范 v1.0

> 数据层 ↔ 假设层(kb/)的契约。语义对齐 HEP(arXiv:2607.09195)与
> W3C PROV;实现为零依赖的 JSON 可选字段——不引 signac/pydantic,
> 不改任何既有字段,旧文件缺字段视为合法(P0 兼容)。

## 字段定义(全部可选,加在新结果 JSON 顶层)

```jsonc
{
  // —— 既有字段不动(verdict / task / steps / mean_acc / ...)——

  "lineage": {                       // 血缘:这个结果从哪来(≈PROV wasDerivedFrom)
    "hypothesis": "H001",            //   判定对象假设(kb/hypotheses/H###-*.md)
    "parent": "latent_recursion_verdict.json@<short-sha>",  // 父结果/父判决
    "generation": 2                  //   HEP generation:de-novo=0,refine=parent+1,merge=max(parents)+1
  },
  "evidence": [                      // 证据清单:本结论依赖的文件(≈PROV wasGeneratedBy 的输入面)
    {"file": "latent_recursion_stack_d4_s0.json", "claim": "acc 0.068 < 0.25 budget_wall"}
  ],
  "protocol": {                      // 预注册协议锚:判负标准不可事后移动的机器凭证
    "preregistered": "judge_decision@304ed35",   // 函数@commit
    "criteria": "strict_monotonic + d8-d1>=2sigma + E0(6 seeds)"
  },
  "config_snapshot": {               // 完整配置快照:任何数字可复现的最小集合
    "task": "pointer_chase", "difficulty": 8, "n_values": 16,
    "steps": 30000, "batch": 128, "lr": 0.0003, "seed": 0
  }
}
```

## 规则

1. **单向引用**:数据层 → 假设层(`lineage.hypothesis` 指向 `H###`);
   H 工件不回写结果清单。REGISTRY 仍是你读账本的唯一入口。
2. **缺字段合法**:存量 55+ 个结果 JSON 不强制回填;新文件
   (2026-09 起的判决级 JSON)应带 `lineage` + `protocol`。
3. **PROBE 合法**:探针/抢救数据照常落盘,`lineage` 照常填——
   门禁要求的是"可追溯",不是"已完成"。
4. **门禁**(PR #14 的 `harvest_registry.py --check` 落地后扩展):
   - 每个带 `lineage.hypothesis` 的 JSON,其 H 工件必须存在;
   - 每个 `status: UNDER_TEST` 以后的 H 工件,必须能解析到 ≥1 个证据文件;
   - orphan 判定不变(文件未被 REGISTRY 人写区引用)。

## 参考实现(第一个带血缘字段的判决文件)

`latent_recursion_verdict.json`(PR #9 分支)重跑后应长这样:

```json
{
  "h_supported": false,
  "diagnosis": {"kind": "budget_wall", "note": "全程 <0.25 ..."},
  "n_rows": 48,
  "lineage": {
    "hypothesis": "H001",
    "parent": "reasoning_depth.jsonl@verdict-30k",
    "generation": 1
  },
  "evidence": [
    {"file": "latent_recursion_core_d8_s0.json", "claim": "mean_acc 0.0692 < 0.25"},
    {"file": "latent_recursion_stack_d8_s0.json", "claim": "mean_acc 0.0723 < 0.25"}
  ],
  "protocol": {
    "preregistered": "judge_decision@304ed35",
    "criteria": "strict_monotonic + d8-d1>=2sigma + E0(6 seeds)"
  }
}
```

## 来源与取舍(为什么是这四个字段)

| 来源 | 采纳 | 不采纳(原因) |
|---|---|---|
| HEP | lineage/generation、evidence 附件、协议锚 | 信念 P(H) 数值化(人写判定更适合 markdown 追加) |
| aexp | RunLink→experiment_id 绑定思想、config 冻结 | signac 范式、Claude Code hook、kb/ 全量 H/E/F(双事实源) |
| PROV-AGENT | wasDerivedFrom 语义命名 | W3C 全本体(对单人仓过重) |
| LLNL co-scientist | parent_ids 血缘 | Elo 锦标赛(本仓用 publishable() 门) |
