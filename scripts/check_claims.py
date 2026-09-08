#!/usr/bin/env python3
"""check_claims.py — RESULTS.md 账本行的可溯源性门禁 (claims → evidence linkage).

RESULTS.md 自我承诺 "Every row maps to a reproducible table in BENCHMARKS.md"。
本脚本把这句承诺变成 CI 断言, 规则如下:

  1. 账本行 = RESULTS.md 中含 "BENCHMARKS.md section" 或 "Where" 列的表格的行。
  2. 每行的引用 (引号章节名 / markdown 链接 / 仓内路径) 必须解析到至少一个
     可验证锚点:
       - 章节锚点: BENCHMARKS.md 标题 (精确 → 子串 → 词元 → 正文 四级匹配)
       - 文件锚点: 仓内真实存在的 *.json / *.md / *.log
     零锚点 = FAIL (纯漂移: 引用指向不存在的东西)。
  3. 行内所有引用都解析成功 → PASS; 行有锚点但个别引用解析失败 → PASS + WARN
     (漂移候选, 供回填); PROVEN 行的主数字必须出现在被引章节正文中。

v1 边界 (差距清单见 docs/SDK_GOVERNANCE_BORROW_PROPOSAL.md):
  - 数字断言只对"有已解析章节"的 PROVEN 行强制; JSON-only 行豁免并计入 WARN。
  - README 数字主张的交叉校验留待 v2。

用法:
  python scripts/check_claims.py [-v] [--report-only]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 行状态中出现这些词 → 非主张行 (RETRACTED/NULL/INERT), 不做数字断言
NON_ASSERT_STATUS = re.compile(
    r"RETRACTED|NULL|INERT|INCONCLUSIVE|research preview|not re-run|provisional|pending",
    re.IGNORECASE,
)

# 引用列里的仓内文件路径 (含反引号/圆括号包裹形态)
FILE_REF = re.compile(
    r"(?:benchmarks|docs|kb|scripts|tests|notes)/[\w./-]+\.(?:json|md|log|txt|jsonl)"
)
MD_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
QUOTED = re.compile(r'"([^"]{2,160})"')

# 数字令牌: 小数优先; 千分位逗号先剔除
NUM_TOKEN = re.compile(r"\d+\.\d+|\d+")

UNICODE_MAP = {
    ord("×"): "x", ord("÷"): "/", ord("％"): "%",
    ord("−"): "-", ord("–"): "-", ord("—"): "-", ord("‑"): "-",
    ord("“"): '"', ord("”"): '"', ord("‘"): "'", ord("’"): "'",
    ord("→"): "->", ord("≧"): ">=", ord("≤"): "<=", ord("≥"): ">=",
}
FULLWIDTH = {c: c - 0xFEE0 for c in range(0xFF01, 0xFF5F)}
STOP_TOKENS = {
    "the", "and", "for", "with", "not", "this", "that", "does", "show",
    "section", "table", "mode", "results", "result", "bench", "note",
}


def norm(text: str) -> str:
    """统一 unicode / 全角 / 空白 / markdown 强调, 供匹配与数字提取。"""
    text = text.translate(UNICODE_MAP).translate(FULLWIDTH)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"[*`_]{1,3}", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def tokens(text: str) -> set[str]:
    return {
        t for t in re.findall(r"[a-z0-9]+", norm(text))
        if len(t) >= 3 and not t.isdigit() and t not in STOP_TOKENS
    }


def numbers(text: str) -> list[str]:
    return NUM_TOKEN.findall(norm(text))


class Benchmarks:
    """BENCHMARKS.md 的标题索引 + 分节正文。"""

    def __init__(self, path: Path):
        self.lines = path.read_text(encoding="utf-8").splitlines()
        self.norm_all = norm(path.read_text(encoding="utf-8"))
        self.headings: list[dict] = []
        for i, line in enumerate(self.lines):
            m = re.match(r"^(#{1,6})\s+(.*)$", line)
            if m:
                title = m.group(2)
                self.headings.append({
                    "level": len(m.group(1)),
                    "title": title,
                    "norm": norm(title),
                    "toks": tokens(title),
                    "line": i,
                })
        # 每个标题的正文切片: 到下一个同级或更高级标题为止 (含子节)
        for idx, h in enumerate(self.headings):
            end = len(self.lines)
            for h2 in self.headings[idx + 1:]:
                if h2["level"] <= h["level"]:
                    end = h2["line"]
                    break
            body = "\n".join(self.lines[h["line"]:end])
            h["body_nums"] = set(numbers(body))

    def resolve(self, quoted: str) -> tuple[str, dict | None]:
        """四级匹配: 精确 → 标题子串 → 词元子集 → 全文正文。返回 (级别, 标题)。"""
        # norm() 把 "→"/"–"/弯引号等统一成 ASCII, 再取 "→ base" 的 base 段
        base = re.split(r"\s*->\s*", norm(quoted))[0].strip()
        base = re.sub(r"\s+in benchmarks\.md$", "", base).strip()
        base = re.sub(r"[\s\"]+$", "", base).strip()
        if not base:
            return "empty", None
        n_base, t_base = base, tokens(base)
        for h in self.headings:  # L1 精确
            if n_base == h["norm"]:
                return "exact", h
        for h in self.headings:  # L2 标题包含引用
            if len(n_base) >= 6 and n_base in h["norm"]:
                return "substring", h
        if t_base:  # L3 引用词元 ⊆ 某标题词元
            for h in self.headings:
                if t_base <= h["toks"]:
                    return "tokens", h
        if len(n_base) >= 8 and n_base in self.norm_all:  # L4 正文 (blockquote/表内文字)
            return "body", None
        return "unresolved", None


def find_tables(lines: list[str]) -> list[dict]:
    """找出所有 markdown 表格及其列头。"""
    tables, cur = [], []
    for i, line in enumerate(lines):
        if line.lstrip().startswith("|"):
            cur.append((i, line))
        else:
            if len(cur) >= 2:
                tables.append(cur)
            cur = []
    if len(cur) >= 2:
        tables.append(cur)
    out = []
    for tbl in tables:
        header = [c.strip() for c in tbl[0][1].strip().strip("|").split("|")]
        ncol = {norm(c) for c in header}
        ref_idx = next((j for j, c in enumerate(header)
                        if "benchmarks.md" in norm(c) or norm(c) == "where"), None)
        if ref_idx is None:
            continue
        num_idx = next((j for j, c in enumerate(header) if "number" in norm(c)), None)
        status_idx = next((j for j, c in enumerate(header) if "status" in norm(c)), None)
        out.append({
            "header_line": tbl[0][0],
            "rows": tbl[2:],  # 跳过分隔行
            "ref_idx": ref_idx,
            "num_idx": num_idx,
            "status_idx": status_idx,
        })
    return out


def parse_row(raw: str) -> list[str]:
    # 表内字面量管道写作 \|, 不能当作列分隔符
    cells = [c.strip() for c in re.split(r"(?<!\\)\|", raw.strip().strip("|"))]
    return cells


def check(args: argparse.Namespace) -> int:
    results_p = Path(args.results)
    bench_p = Path(args.benchmarks)
    benches = Benchmarks(bench_p)

    failures: list[str] = []
    warnings: list[str] = []
    rows_checked = 0

    lines = results_p.read_text(encoding="utf-8").splitlines()
    for tbl in find_tables(lines):
        for line_no, raw in tbl["rows"]:
            cells = parse_row(raw)
            if len(cells) <= max(tbl["ref_idx"], tbl["num_idx"] or 0):
                continue
            if not any(cells) or set(cells[0]) <= set("-: "):
                continue
            rows_checked += 1
            ref_cell = cells[tbl["ref_idx"]]
            status_cell = cells[tbl["status_idx"]] if tbl["status_idx"] is not None else ""
            num_cell = cells[tbl["num_idx"]] if tbl["num_idx"] is not None else ""

            # ── 解析所有引用 ──────────────────────────────────────────
            resolved_sections: list[dict] = []
            body_anchors = 0
            missing_files: list[str] = []
            unresolved_quotes: list[str] = []
            for q in QUOTED.findall(ref_cell):
                level, h = benches.resolve(q)
                if h is not None:
                    resolved_sections.append(h)
                elif level == "body":
                    body_anchors += 1  # 正文级命中: 有效锚点, 但无分节数字域
                else:
                    unresolved_quotes.append(q)
            # 引号外的裸文字片段 (如 Correction note (2026-07-04)) 走正文级匹配
            leftover = QUOTED.sub(" ", ref_cell)
            for frag in re.split(r"[+/;]| table\b", leftover):
                frag = frag.strip(" \t\"'")
                if len(norm(frag)) >= 8 and norm(frag) in benches.norm_all:
                    body_anchors += 1
                    break
            file_refs = set(FILE_REF.findall(ref_cell)) | {
                p for p in MD_LINK.findall(ref_cell)
                if FILE_REF.fullmatch(p.strip())
            }
            for f in file_refs:
                if not (REPO_ROOT / f).exists():
                    missing_files.append(f)

            anchors = (len(resolved_sections) + body_anchors
                       + len(file_refs) - len(missing_files))
            where = f"L{line_no + 1}"

            # ── 判定 ──────────────────────────────────────────────────
            if anchors == 0:
                failures.append(
                    f"{where} 零可验证锚点: 引用={unresolved_quotes or ref_cell!r} "
                    f"缺文件={missing_files}"
                )
                continue
            for f in missing_files:
                failures.append(f"{where} 引用文件不存在: {f}")
            for q in unresolved_quotes:
                warnings.append(f"{where} 章节引用漂移(行有其他锚点): {q!r}")

            # ── PROVEN 数字断言 ───────────────────────────────────────
            if (tbl["num_idx"] is not None and resolved_sections
                    and not NON_ASSERT_STATUS.search(status_cell)):
                nums = numbers(num_cell)
                if nums:
                    primary = nums[0]
                    body_floats: set[float] = set()
                    for h in resolved_sections:
                        body_floats |= {round(float(n), 9) for n in h["body_nums"]}
                    # '%' 单位等价: RESULTS 写百分比、BENCHMARKS 写小数 (13.3% ↔ 0.133)
                    candidates = {round(float(primary), 9)}
                    m = re.search(r"(\d+(?:\.\d+)?)\s*%", norm(num_cell))
                    if m and m.group(1) == primary:
                        candidates.add(round(float(primary) / 100, 9))
                    all_floats = {round(float(x), 9) for x in nums}
                    found = sum(1 for n in all_floats if n in body_floats)
                    if not (candidates & body_floats):
                        failures.append(
                            f"{where} PROVEN 主数字 {primary} 不在任何被引章节中 "
                            f"(命中 {found}/{len(all_floats)})"
                        )
                    elif args.verbose and found < len(all_floats):
                        warnings.append(
                            f"{where} 数字部分命中 {found}/{len(all_floats)} "
                            f"(主数字 {primary} ✓)"
                        )
                else:
                    warnings.append(f"{where} Number 列无可提取数字")

    # ── 汇总 ──────────────────────────────────────────────────────────
    print(f"账本行: {rows_checked}  失败: {len(failures)}  警告: {len(warnings)}")
    for f in failures:
        print(f"  ❌ {f}")
    for w in warnings:
        print(f"  ⚠️  {w}")
    if failures and not args.report_only:
        print("❌ check_claims: 账本存在不可溯源行")
        return 1
    if failures:
        print("⚠️  report-only 模式: 存在失败但不阻塞")
        return 0
    print("✅ check_claims: 每个账本行均可溯源 (章节锚点/证据文件)")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default=str(REPO_ROOT / "RESULTS.md"))
    p.add_argument("--benchmarks", default=str(REPO_ROOT / "BENCHMARKS.md"))
    p.add_argument("--report-only", action="store_true",
                   help="只报告不阻塞 (exit 0)")
    p.add_argument("-v", "--verbose", action="store_true")
    return check(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
