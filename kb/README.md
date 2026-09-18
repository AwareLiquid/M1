# kb/ — 假设层工件(决策血缘)

> 语义对齐 HEP(Hypothesis Evolution Protocol, arXiv:2607.09195)与 W3C PROV;
> 载体是纯 markdown + 简易 frontmatter,零运行时依赖。
> 参考:`docs/DECISION_TRACE_SPEC.md`(数据层血缘字段)、`benchmarks/results/ADJUDICATION_LOG.md`(决策日志)。

## 工件类型(当前四种)

| 前缀 | 类型 | 目录 | 生命周期 |
|---|---|---|---|
| `H###` | Hypothesis — 一个可判定的假设,带预注册判负标准 | `kb/hypotheses/` | PROPOSED → UNDER_TEST → SUPPORTED / REFUTED / DORMANT |
| `T###` | Thread — 一条研究线/方向,可孵化多个 H | `kb/threads/` | PROPOSED → EXPLORING → PROMOTED / CLOSED |
| `ADJ###` | Decision ADR — 一个关键决策/调整的正文工件(2026-09-06 起,一决策一文件) | `kb/decisions/` | 追加式:新决策开新文件;正文不可变,勘误开新 ADJ 并双向注记。索引 = benchmarks/results/ADJUDICATION_LOG.md(每条 1 行) |
| `L###` | Lesson/复盘 — 错误反馈体系(2026-08 起,一教训一文件) | `kb/lessons/` | ACTIVE / SUPERSEDED(被新 L 或规则取代时注记,文件不删)。正文节:Statement / Instances / Enforcement。**勘误先例**:撞号或误引按追溯登记处理(ADJ-006;H006 对 overnight"H004"的脱钩;ADJ-014 对 PR #52 迟到件 ADJ-012 撞号回溯改号) |

## 知识户口总表(新文档先找家)

方向→本目录 H/T + backlog;决策→ADJ;错误反馈→L;规则→`docs/` 协议类 +
backlog 协议节;运维 how-to→`docs/*_RUNBOOK*`、`docs/COMPUTE_TIERS.md` §三;
参考标定→`docs/COMPUTE_TIERS.md`、BENCHMARKS/RESULTS。调研流水与出处登记在
本地不入库的 `notes/searches/`(苗圃;毕业方向即上表)。

## 三条铁律

1. **假设层只写信念与判据,不写数字。** 所有实测值只活在
   `benchmarks/results/*.json` 与 `BENCHMARKS.md`/`RESULTS.md`/REGISTRY。
   假设文件引用数据时只给文件名 + 定性结论(如 "48/48 cell acc < 0.25, budget_wall")。
2. **工件不可变,演进靠新工件。** 推翻 H001 不改 H001,而是新建 H002
   (frontmatter 标 `refine_of: H001`)。H001 的失败是可引用的历史。
   判负标准写死后不得移动——与 `judge_decision` docstring 同一纪律。
3. **单向引用。** 数据层(结果 JSON 的 `lineage.parent`)指向 H###;
   H### 不维护结果清单(那是 REGISTRY 的事)。两边不互相回写,防止漂移。

## 强制点(唯一)

入库门禁:`python benchmarks/harvest_registry.py --check`(PR #14 落地后)
扩展校验——每个 verdict JSON 的 `lineage.parent` 必须指向存在的 H 工件;
每个 UNDER_TEST 及以后的 H 必须能解析到至少一个证据文件。
**在合并窗口强制,不在实验时强制**——实验在 GPU 机器上,拦不住也不该拦;
抢救数据合法,但必须补登记(状态 PROBE/CLOSED 而非丢失)。

## 命名与格式

- 文件名:`H###-短横线摘要.md`(如 `H001-stack-depth-monotonic.md`)。
- frontmatter 字段:`id` / `status` / `thread`(可选)/ `refine_of`(可选)/
  `created` / `refs`(代码或数据路径列表)/ `owner` + `deadline`(UNDER_TEST 必填,ADJ-006)。
- 必填正文节:`## Statement`(假设陈述)、`## Pre-registered judgment`(判负标准,写死)、
  `## Evidence basis`(立项依据,只引不测,**须含外部扫描:谁做过类似的事、失败在
  哪里——H 立项与判负后转向时强制,ADJ-014**)、
  `## Verdict log`(追加式判决记录)。
- L 工件:`L###-短横线摘要.md`,frontmatter `id`/`status`/`created`/`refs`/`origin`,
  正文节 `## Statement` / `## Trigger`(触发条件,§4.2 异常晋升的匹配依据)/
  `## Countermeasure`(对策)/ `## Source`(来源锚)。L 只写行为规则与触发条件,
  不写实验数字(铁律 1 同样适用);对策仍要走预注册,不是免检结论。

## 生命周期纪律(2026-09-04,ADJ-006)

- UNDER_TEST 的 H 必须登记 `owner` 与 `deadline`。期限用事件锚
  (如"下一次 GPU 会话"/"下一合并窗口"),不强制日历日期;
- 超期处理二选一,均以 Verdict log 追加条目落地:**续期**(写明新期限
  与卡点)或 **DORMANT**(写明原因与复燃条件)。跨合并窗口无任何 log
  条目的无声搁置 = 违规,由合并窗口门禁(`harvest_registry.py --check`)
  标记;
- 判定落地(frontmatter 改 SUPPORTED/REFUTED)只认三件套:预注册标准
  + 证据轨 JSON 入 `benchmarks/results/` + RESULTS.md 状态位回填。
  三件齐才允许改状态——与 PR_MERGE_POLICY §5 结论级门禁同一条纪律。

## 教训库演进纪律(L###,2026-09-09,ADJ-014)

- **受控演进**:每个合并窗口的教训复盘只允许 ≤4 个操作,操作集固定为
  `ADD / EDIT / REMOVE / SUPERSEDE`(取代开新条,只命名不删除,同 consolidation_policy
  `policy_forget` 的"只命名过期键"纪律)。禁止无操作的无声膨胀。
- **准入判据**:"删掉这条会导致再犯同类错吗?不会就删"(Claude Code 官方
  best-practices 的教训文件准入)。不满足即不得 ADD。
- **来源要求**:每条 L 必须有 `Source` 锚(实际事故/H/ADJ 路径);
  同一异常第 2 次出现(§4.2 异常晋升)时,评审现有 L 的 Trigger 是否已覆盖,
  未覆盖即 ADD 并在条目里注记两次出现的锚。
