#!/usr/bin/env python3
"""
scripts/check_doc_budget.py — 热区文档尺寸预算门禁

背景（2026-09-06 文档治理）：追加式活文档（交接/决策索引/路线图）若不设上限，
会退化成流水账（实测：ADJUDICATION_LOG 4 天 0→148 行、HANDOFF 出现 §3.75 式
考古章节）。本脚本在 CI audit-results job 中强制"热区"文档的行数预算：

- 热区 = 决策/状态/交接/路线图类活文档（读者需要"当前态"的文档）；
- 参考型文档（BENCHMARKS/ABLATIONS/ARCHITECTURE 的 lookup 表）只设宽松上限，
  防失控不防篇幅；
- 超预算 = CI 红。提高预算必须改本文件的 BUDGETS——即一次显式的 PR 决策
  （对应 kb 决策工件），不允许无声膨胀；
- 文件不存在则跳过（如 docs/RESEARCH_PLAN.md 随 #33 合入后自动纳入）。

Usage:
    python scripts/check_doc_budget.py            # 校验全部
    python scripts/check_doc_budget.py -v         # 显示每项实际行数
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 热区预算（行）。首次设定 = 2026-09-06 治理后的现状 + 少量余量；
# 之后每次上调都应伴随 kb/decisions/ 新决策工件。
BUDGETS = {
    "HANDOFF.md": 660,                        # 冷热分离后（快照 + 链接，剧情入归档）
    "benchmarks/results/ADJUDICATION_LOG.md": 60,   # 纯索引：每决策 1 行
    "RESULTS.md": 110,
    "README.md": 650,
    "docs/RESEARCH_PLAN.md": 220,             # §1 状态表 + pitch；决策记录去 kb/decisions/
    # 参考型（宽松上限，防失控不防篇幅）
    "BENCHMARKS.md": 1850,
}

VERBOSE = "-v" in sys.argv

failed = []
for path, budget in BUDGETS.items():
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        if VERBOSE:
            print(f"  SKIP  {path}: 不存在（随所属 PR 合入后纳入）")
        continue
    n = sum(1 for _ in open(full, encoding="utf-8"))
    status = "PASS" if n <= budget else "FAIL"
    print(f"  {status}  {path}: {n} / {budget} 行")
    if n > budget:
        failed.append((path, n, budget))

print()
if failed:
    print("FAIL: 热区文档超预算 —— 请先归档/瘦身，或在同 PR 中修改 "
          "scripts/check_doc_budget.py 的 BUDGETS 并附 kb/decisions/ 决策工件")
    sys.exit(1)
print("PASS: all hot-zone docs within size budgets")
