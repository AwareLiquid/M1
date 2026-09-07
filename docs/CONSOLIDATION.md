# CONSOLIDATION — 固化数据流(写规则 / 遗忘钩子 / 睡眠周期)

`mt_lnn/memory_broker/consolidation_policy.py` — 固化策略的两个**纯函数**,
外加既有 sleep-cycle 路径的一个**可选**挂点。本页只讲数据流;能力对照见
[MEMORY_BROKER.md](MEMORY_BROKER.md)。

## 诚实边界(先读这个)

**冲突消解不在这里。** 四维基准 D3(经写规则的冲突消解)在 v0 预算
全配置判负(RESULTS.md 既有行;kb/H003 verdict log 按预注册记档),
负结果照记、标准不移动。本模块**不判定冲突双方谁为真、不合并数值、
不直接改任何存储**——它只回答两件事:一条记录该沉淀成哪些绑定
(`policy_write`),一场**已被检出**的冲突让哪些旧键过期
(`policy_forget`)。检出冲突是 SDK 检测层的职责(兜底方);对返回的旧键
做什么(遗忘 / supersede / 无视)是调用方的决定。

## 数据流

```
                        ┌────────────────────────────────────────────┐
                        │  Awareness-SDK (产品仓, 不在本仓 import)     │
                        │                                            │
                        │  SDK-C 事件钩子 ─── 检出冲突 ──┐             │
                        │        │                     │             │
                        │  policy_forget(冲突事件)      │ (纯函数调用, │
                        │        │                     │  跨进程安全) │
                        │        ▼                     │             │
                        │  [旧键...]                   │             │
                        │        │                     │             │
                        └────────┼─────────────────────┼─────────────┘
                                 │ 调用方决定:            │
                                 │ forget / supersede /  │
                                 │无视(都由它执行)         │
                                 ▼                       ▼
   ┌────────────────────────── 本仓 M1 ────────────────────────────┐
   │                                                              │
   │  经验 buffer ──sample──▶ 睡眠周期 SleepWakeConsolidator        │
   │  (ReservoirBuffer)      .nrem_replay(                        │
   │                             buffer, knowledge_memory,        │
   │                 可选 ───▶    write_policy=policy_write)      │
   │                             │                                │
   │                ┌────────────┴─────────────┐                  │
   │                │ write_policy=None (默认)   │ write_policy=    │
   │                │  每集一条写:               │ policy_write:    │
   │                │  write(key, content, meta)│ 逐记录返回        │
   │                │  (原行为,逐字节等价)        │ [(key,value)...] │
   │                └────────────┬─────────────┘  (salience 门可清零)│
   │                             ▼                                │
   │              长期存储 (duck-typed write)                      │
   │   ├─ PersistentKnowledgeMemory (平铺)                         │
   │   └─ GraphKnowledgeMemory (6deb3ee 起: 边随写建,               │
   │      SUPERSEDES/CONTRADICTS 生命周期照常生效)                  │
   └──────────────────────────────────────────────────────────────┘
```

## 挂点纪律(off 位等价)

`nrem_replay` 新增的 `write_policy` 参数**默认 None**:采样、salience 排序、
写入次数与内容与挂点引入前逐字节一致(测试
`test_off_position_is_byte_equivalent` /
`test_policy_with_explicit_keys_matches_plain_path` 钉死)。传
`policy_write` 后,排序仍不变,只有**每条记录写什么**交给策略——salience
门(`min_salience`)可以在记录级清零绑定,补充(不是替代)既有的
`consolidate_fraction` 批级门。

## policy_write 规则(依序)

1. **salience 门**:设了 `min_salience` 且记录带数值 salience 低于门 → `[]`
   (不沉淀;无 salience 或非数值 = 不设门,绝不抛错);
2. **显式键**:`keys`(列表)或 `key`(单值)→ 每键一条 `(key, value)`;
   睡眠路径把 key/content 字段预填进记录,所以策略产出的键与默认路径一致;
3. **内容自寻址兜底**:无任何键 → `[(content, content)]`(与知识/图谱库对
   自由文本的处理一致:嵌入文本、存文本)。

空记录 → `[]`。键值原样返回,不做归一化、不做编码(编码是
store/broker 的职责)。

## policy_forget 规则

| 冲突事件 | 返回 |
|---|---|
| 无 `old.key` | `[]` |
| `type == "duplicate"` | `[]` |
| old 内容 == new 内容(重述) | `[]` |
| 其余(update / contradiction / 未知类型) | `[old.key]` |

## 纯函数契约

确定性、全域(缺字段不抛错)、零 IO、不改输入、不 import torch——SDK-C
事件钩子可在任意线程/进程直接调用。
