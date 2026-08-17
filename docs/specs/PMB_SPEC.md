# PMB — Persistent Memory Benchmark v0（持久记忆基准规格）

> 2026-07-16 · 定位：附录 B 裁决的"最高杠杆动作"——该生态位（跨会话持久记忆）当前没有受信任基准（Titans Revisited, arXiv:2510.09551 点名此缺口），由本项目定义任务族与协议，让后来者在我们的坐标系里报数。
> 同时它就是 E2 的升级版：对照组从"无记忆"升级为 RAG 检索基线（附录 B：0.56 vs 0.00 的对照太弱，必须对打检索系统）。

## 1. 与现有基准的差异（存在的理由）

| 基准 | 测什么 | 不测什么（PMB 补位） |
|---|---|---|
| NIAH/RULER | 窗口内检索 | 跨会话；窗口外 |
| BABILong | 超长单序列 | **会话边界**（状态序列化/恢复）；更新与遗忘 |
| LongMemEval | 检索式记忆产品 | 权重内/状态内记忆；状态大小-召回权衡 |

PMB 的三个独有轴：**会话边界**（每个 session 之间状态必须序列化到磁盘再恢复——bit-exact 持久化是参赛资格）、**更新语义**（后写覆盖先写）、**遗忘曲线**（召回率 vs 间隔会话数/token 数的显式报告）。

## 2. 任务族（v0 三个，全部程序化生成、可复现种子）

### T1 · 跨会话保持（Cross-Session Retention）
- Session 1 注入 N 个键值事实（人物-属性对，模板多样化），随后 K 个无关会话（干扰内容），Session K+2 提问。
- 变量：N ∈ {4, 16, 64}，K ∈ {0, 4, 16}。
- 指标：exact-match 召回率，按 (N, K) 网格报告。

### T2 · 流式更新召回（Streaming Update Recall）
- 同一键在多个 session 中被更新（"Alice 的电话改为 X"），查询必须返回**最新**值。
- 指标：最新值召回率 + 过期值误答率（stale-answer rate，单独报告——这是检索系统的典型弱点）。

### T3 · 遗忘曲线（Forgetting Curve）
- 固定注入量，测召回率随间隔增长的衰减：间隔 ∈ {0, 2, 8, 32} 个干扰会话（每个 ~2K tokens）。
- 输出：完整曲线，不许只报单点。容量诚实条款：固定态系统必须报告此曲线（HOLA 已证明竞争性关联下固定态必然覆写）。

## 3. 参赛系统协议（统一接口）

每个系统实现三个方法，会话之间**必须经过磁盘序列化**：
```
ingest(session_id, text)        # 逐会话喂入
snapshot(session_id) -> bytes   # 会话结束，状态落盘
answer(question) -> str          # 恢复状态后作答
```

v0 内置四个参照系统：
1. **none**：无记忆（下界；也是旧 0.56 vs 0.00 的对照，保留供连续性）
2. **oracle-context**：全部历史塞进上下文（能塞下时的上界；超窗即失效——这本身就是数据点）
3. **rag**：句级切分 + 向量检索 top-k 注入（迷你版 mem0/Zep 代表；编码器可插拔，默认 e5-small，CPU 可跑）
4. **fastweight**：MT-LNN adapter 的 (F,z) 会话快照 + FastWeightSessionStore（本项目的参赛者）

## 4. 报告格式（三列强制）

每个系统每个任务报告：**召回率 · 持久状态字节数 · 答题延迟**。缺一不可——这个基准的核心论点就是三者的权衡面，只报召回率的提交视为无效。

## 5. v0 范围与诚实声明

- 生成器为合成模板（英文 v0；中文 v1），真实对话语料（LongMemEval 风格）列为 v1。
- fastweight 参照系统需要 GPU 微调过的 adapter（E2 的产出）；none/oracle/rag 纯 CPU 可跑。
- 本项目发布 PMB 时必须同时公布自己在全部三列上的数字，包括 T3 遗忘曲线的不利区间——基准的可信度来自定义者先亮出自己的短板。

## 6. 文件布局

```
benchmarks/persistent_memory/
  __init__.py
  tasks.py        # T1/T2/T3 生成器（种子可复现）
  systems.py      # 四个参照系统（统一 ingest/snapshot/answer 协议）
  run_pmb.py      # CLI：--task --system --seed --out_json
  README.md       # 对外文档（协议 + 提交规范）
```

## 7. 评分防御（v0.1）

背景：一个"作弊系统"（ingest 什么都不存、answer_context 直接返回全部 900,000 个可能的 6 位码，约 5.4MB 文本）曾把三列指标全部打满。v0.1 引入三层防御：

1. **评分预算截断（harness 强制）**：评分只看 harness 截断后的前缀——evidence 模式截断到 `--context_char_budget`（默认 4000 字符，最多容纳约 570 个码，枚举命中率 ≈ 0.06%/题）；generative 模式对 `answer()` 输出截断到 200 字符（≈ 28 个码，≈ 0.003%/题）。截断由 harness 执行，系统自己无法绕过。截断前的原始长度以 `context_chars_mean` 写进报告，灌水行为直接暴露。
2. **评分模式声明**：系统通过类属性 `mode = "evidence" | "generative"` 声明评分路径。generative 系统（如 fastweight）由 harness 调 `answer()` 计分与计时，绝不调 `answer_context()`。
3. **任务加难（防饱和）**：每个被查询人物额外携带 2 条其他属性的 confuser 事实（各带码）——命中错属性 = 检索粒度停留在实体级；每个干扰会话植入 8 条"其他人物-属性-码"诱饵事实，措辞与真实事实同模板——"找 6 位数字"不再是策略。全部码（gold/stale/confuser/decoy）在单次全局无放回抽样中产生，不变量保持：gold 全局唯一、oracle 无限窗口 recall = 1、none = 0。实测（seed 0，rag/hash topk=8）：T1 n=64,k=16 recall 从 1.0 降至 0.78，出现真实区分度。
