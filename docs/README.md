# docs/ — M1 文档索引

2026-07-31 根目录整理后的文档归档。**评价链核心文档仍在仓库根目录**：`README.md`（入口）、`RESULTS.md`（事实来源）、`BENCHMARKS.md`（可复现表格）、`ABLATIONS.md`、`ARCHITECTURE.md`、`SPEC.md`、`PRD.md`、`HANDOFF.md`（活跃交接）。

投资人 deck、论文成品 PDF、营销材料已迁至私有仓库 [AwareLiquid/AwareLiquid-Web](https://github.com/AwareLiquid/AwareLiquid-Web)。

## 子目录

| 目录 | 内容 |
|---|---|
| `guides/` | 复现与使用指南：`CLOUD_RUN` / `COLAB_RUN` / `KAGGLE_RUN` / `CLOUD_TRAINING_GUIDE` / `RECIPES`（Recipe API）/ `VISUALIZATION_TOOLS` / `ITERATION_PRINCIPLES`（迭代方法论） |
| `reviews/` | 历史评审与分析快照：`V2_REVIEW` / `DEEP_REVIEW_2026_07_14` / `ARCHITECTURE_ANALYSIS_REPORT` / `PHASE5B_ANALYSIS` / `NEEDLE_FIX`（needle harness 修复记录）/ `VISUALIZATION_COMPLETE`。<br>注：`PUBLICATION_READINESS`（论文短板与未完成实验清单）已迁至私有仓库 `AwareLiquid/AwareLiquid-Web` 的 `internal/` —— 该文件自述"勿推公开仓库"，M1 是公开仓库。 |
| `specs/` | 规格与路线图：`AWARELIQUID_SYSTEM_MVP` / `AWARENESS_NETWORK_PRD` / `PMB_SPEC`（持久记忆基准）/ `OPERATOR_COMPRESSION_PRD`+`TODO` / `POSITION_FREE_ARCHITECTURE` / `NEW_BASES` / `BRAIN_INSPIRED_ROADMAP` / `MT_LNN_ARCHITECTURE_VISUAL` |
| （既有）`PRODUCT_LINES.md` | M 系列 vs O 系列产品线能力卡 |
| （既有）`ROADMAP_M2.md` | M2 路线图 |

## 其它整理去向

- 根目录 `_diag_*` / `_test_*` / `_bench_*` 等 scratch 脚本 → `scripts/diagnostics/`（从仓库根目录以 `python scripts/diagnostics/xxx.py` 运行，部分脚本假定 CWD 为仓库根）
- `llm-viz-QUICKSTART.md`、`llm-viz-mtlnn.html`、`mtlnn-chapter.html` → `llm-viz-integration/`
