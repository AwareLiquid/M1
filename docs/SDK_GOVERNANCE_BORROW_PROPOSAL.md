# 提案 · 把 M1 的证据文化机械化 — 借鉴 Awareness-SDK 的三台治理机器

- **状态**: Proposed(待治理评审)
- **日期**: 2026-09-07
- **提出方**: `iter/claims-gate` 迭代线(本 PR)
- **性质**: 工程/治理提案,**零新增主张**;引用的外部数字全部标注出处,不构成 M1 主张。

## 0. 一页结论

M1 发明了这套证据文化(RESULTS 唯一事实源 / 预注册 / 诚实 Null / 撤回入档),下游的
Awareness-SDK 把同样的文化**机械化**成了三台接到 CI 上的机器:主张校验器、同环境
A/B 仪器、trace 聚合分析器。本仓的规范层已属一流,缺的是把"规范→断言"的最后一公里
交给机器。本文给出有据可查的差距清单与按杠杆排序的落地路线;**本 PR 落地第 ① 件**
(`scripts/check_claims.py` + CI 接入),其余列为后续立项。

## 1. 审计范围与证据

只读审计 main(`33d39d5`)与 9 个 `M1-*` worktree,2026-09-07 执行。核心差距(每条
可复核):

| # | 差距 | 证据 |
|---|---|---|
| 1 | 主张→产物无机械链接:RESULTS 行只绑 BENCHMARKS **章节**,不绑脚本/JSON/commit | `RESULTS.md` PROVEN 表列结构 |
| 2 | 文档数字无交叉校验门禁:CI 唯一的 byte 级断言只覆盖 kv_frontier 账本 | `.github/workflows/ci.yml` Job 3;`scripts/validate_results.py` 自述不校验数字 |
| 3 | 论文级主张游离于账本外(14.7% 摘要主张仅 HANDOFF 待办表追踪) | `HANDOFF.md` L359/L550 |
| 4 | 流程制度分叉:REGISTRY/harvest/process_window 停在 `iter/process-train`,main 未采纳;main 的 kb 文档却引用未落地的 `harvest_registry.py --check` | `git ls-tree main`;`kb/README.md` |
| 5 | 判决账本双轨:kb 称 REGISTRY 是唯一入口,main 上只有 ADJUDICATION_LOG | `main:kb/README.md` vs `git ls-tree main benchmarks/results` |
| 6 | 权重等大资产无清单/校验和/位置账(O1 48M 失踪) | `HANDOFF.md` L258/L368;`.gitignore`;ADJ-007"单点丢档第 3 次" |
| 7 | 结果 JSON 无统一 trace schema(git HEAD/env/时间戳缺位;DECISION_TRACE_SPEC 字段可选且不回填) | `docs/DECISION_TRACE_SPEC.md` 规则 2 |
| 8 | 对表靠人:#30 的"4 位小数逐字节一致"由 commit message 人工声明 | `git show 33d39d5` |
| 9 | 实验执行过程无日志制度(notes/ 近乎空,叙事靠 HANDOFF 手写) | `notes/` |

## 2. 落地路线(按杠杆排序)

| 优先 | 机器 | 对应差距 | 状态 |
|---|---|---|---|
| ① | **claims→证据机械链接**:`check_claims.py` 把 RESULTS 头部承诺("Every row maps to a reproducible table in BENCHMARKS.md")变成 CI 断言 | 1/2 | **本 PR 落地** |
| ② | **绝对门禁 vs 零回归拆分 + 同环境 A/B 仪器**(参照 SDK `run_ab.sh`:基线 worktree + 双跑 + 逐题位 diff;SDK F-070 ADR 把两个判据独立化) | 3/8 | 待立项;直接服务 14.7% 复核与 modern Transformer 对照 |
| ③ | **trace 读写闭环**(参照 SDK `analyze_trace.py`:JSONL→命中率/降级频率/p50-p95,方向变更必须消费聚合证据) | 7/8 | 待立项;M1 有写侧(JsonlMetricWriter)缺读侧 |
| ④ | **权重资产登记**(manifest + sha256 + 位置账) | 6 | 待立项 |
| ⑤ | **REGISTRY 与 ADJUDICATION_LOG 合流** | 4/5 | 待立项 |
| ⑥ | **合并前对齐纪律**:下游(SDK)已按 GOVERNANCE 式流程提交 memory_broker 合并前对齐提案(trace writer 注入/事件词表/session 语义/trace_id 透传);#40 携带 broker,建议合并前回复 | — | 时效项 |

外部参照(非 M1 主张,均为 Awareness-SDK 仓内登记或其 PR #1 报告,2026-09-06):
其主张登记簿(GOVERNANCE §7,机械校验 `check_claims.py`)、门禁双判据(F-070 ADR)、
LongMemEval 500Q A/B 实证(95.8 vs 96.0 归因环境漂移、逐题位算等价)。

## 3. 本 PR 改动清单

| 文件 | 改动 |
|---|---|
| `scripts/check_claims.py` | 新增:账本行可溯源性校验(解析/四级锚点匹配/主数字断言/%↔小数等价) |
| `tests/test_check_claims.py` | 新增:10 个合成契约用例 + 1 个真账本冒烟(锚点漂移会让 full-test 变红) |
| `.github/workflows/ci.yml` | `audit-results` 新增一步(check_claims) |
| `RESULTS.md` | 仅 1 行:Selective Copy 行锚点列补 `"What this shows"`(门禁首例真实抓漏,见 §4;数字零变化) |

校验规则要点:

- 每个账本行(含 BENCHMARKS.md section / Where 列的表格)必须解析到 **≥1 个可验证
  锚点**:BENCHMARKS.md 章节(精确→子串→词元→正文 四级匹配)或仓内证据文件
  (`benchmarks/results/*.json` 等)。零锚点 = FAIL。
- PROVEN 行(状态无 RETRACTED/NULL/INERT/INCONCLUSIVE/preview)的主数字必须出现在
  被引章节正文;`+13.3%` 与 `+0.133` 视为等价写法。
- 行有锚点但个别引用漂移 → PASS + WARN(见 §4 回填清单)。

## 4. 首轮运行产出:漂移回填清单(3 项 WARN,建议下一合并窗口处理)

1. `"Parametric memory four-competency bench"`(PROVEN 行)——BENCHMARKS.md 无对应
   章节,证据仅在 `benchmarks/results/parametric_memory_summary.json`。建议:在
   BENCHMARKS.md 建节,或在 RESULTS 行内显式标注 evidence-JSON-only。
2. `"Round 2/3 distillation"`(RETRACTED 行)——指向不存在的章节名;实际内容在
   `### Round 2 — teacher-forced alignment` 一节。
3. `"Irregular-sampling streaming edge — mechanism probe"`(NULL 行)——叙事已按
   预注册规则关闭("CT narrative is not told anywhere in this repo"),建议行内显式
   标注 evidence-JSON-only(`synth_ct_control_dt.json` 已存在且可校验)。

## 5. 与在途 PR 的交集

- 与 `merge/batch4-engineering`(#40):仅 `.github/workflows/ci.yml` 微量行交集
  (本 PR +4 行),rebase 即可;RESULTS.md 改动行与 #40 触及的行不重叠。
- 与 `merge/batch4-roadmap`(#41):零文件交集。
