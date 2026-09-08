"""check_claims.py 的契约测试。

合成 fixture 锁定解析与判定行为; 最后一个用例对真实账本跑冒烟,
使 RESULTS/BENCHMARKS 锚点漂移在 CI 的 full-test 层也会被拦下。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import check_claims  # noqa: E402

PROVEN_HEADER = """# R

## PROVEN — reproducible

| Claim | Number | Status | BENCHMARKS.md section |
|---|---|---|---|
"""


def write(tmp_path: Path, results: str, benchmarks: str) -> tuple[Path, Path]:
    r = tmp_path / "RESULTS.md"
    b = tmp_path / "BENCHMARKS.md"
    r.write_text(results, encoding="utf-8")
    b.write_text(benchmarks, encoding="utf-8")
    return r, b


def run(tmp_path: Path, monkeypatch, results: str, benchmarks: str) -> int:
    r, b = write(tmp_path, results, benchmarks)
    monkeypatch.setattr(check_claims, "REPO_ROOT", tmp_path)
    return check_claims.main(["--results", str(r), "--benchmarks", str(b)])


def test_proven_row_with_resolvable_anchor_and_primary_number(tmp_path, monkeypatch):
    rc = run(
        tmp_path, monkeypatch,
        PROVEN_HEADER + '| Claim A | **0.56 mean** (0.6 per seed) | ✅ proven | "Cross-window recall" |\n',
        "# B\n\n## Cross-window recall (2026-01-01)\n\nmean 0.56, per-seed 0.6.\n",
    )
    assert rc == 0


def test_row_with_zero_anchors_fails(tmp_path, monkeypatch):
    rc = run(
        tmp_path, monkeypatch,
        PROVEN_HEADER + '| Claim A | 0.56 | ✅ proven | "No Such Section" |\n',
        "# B\n\n## Something else\n\n0.56\n",
    )
    assert rc == 1


def test_quoted_drift_with_existing_file_anchor_passes_with_warning(
    tmp_path, monkeypatch, capsys
):
    (tmp_path / "benchmarks" / "results").mkdir(parents=True)
    (tmp_path / "benchmarks" / "results" / "x.json").write_text("{}", encoding="utf-8")
    rc = run(
        tmp_path, monkeypatch,
        PROVEN_HEADER
        + '| Claim A | 0.56 | ✅ proven | "Stale Section Name" + `benchmarks/results/x.json` |\n',
        "# B\n\n## Real section\n\n0.56\n",
    )
    assert rc == 0
    assert "漂移" in capsys.readouterr().out


def test_missing_file_reference_fails(tmp_path, monkeypatch):
    rc = run(
        tmp_path, monkeypatch,
        PROVEN_HEADER
        + '| Claim A | 0.56 | ✅ proven | "Real section" + `benchmarks/results/gone.json` |\n',
        "# B\n\n## Real section\n\n0.56\n",
    )
    assert rc == 1


def test_proven_primary_number_missing_from_section_fails(tmp_path, monkeypatch):
    rc = run(
        tmp_path, monkeypatch,
        PROVEN_HEADER + '| Claim A | **9.999** | ✅ proven | "Section A" |\n',
        "# B\n\n## Section A (2026-01-01)\n\nmeasured 0.56 only.\n",
    )
    assert rc == 1


def test_percentage_matches_decimal_in_section(tmp_path, monkeypatch):
    rc = run(
        tmp_path, monkeypatch,
        PROVEN_HEADER + '| Claim A | **+13.3%** (25/30) | ✅ proven | "Section A" |\n',
        "# B\n\n## Section A (2026-01-01)\n\nuplift +0.133 (25/30).\n",
    )
    assert rc == 0


def test_retracted_row_skips_number_assertion(tmp_path, monkeypatch):
    rc = run(
        tmp_path, monkeypatch,
        PROVEN_HEADER
        + '| ~~Claim A~~ | 9.999 | ❌ **RETRACTED** — wrong baseline | "Section A" |\n',
        "# B\n\n## Section A (2026-01-01)\n\nsuperseded.\n",
    )
    assert rc == 0


def test_escaped_pipe_inside_cells_does_not_break_parsing(tmp_path, monkeypatch):
    rc = run(
        tmp_path, monkeypatch,
        PROVEN_HEADER
        + '| Claim A | tied (every pairwise \\|t\\| < 1), mt 0.56 best | ✅ proven | "Section A" |\n',
        "# B\n\n## Section A (2026-01-01)\n\nmt 0.56 best.\n",
    )
    assert rc == 0


def test_arrow_reference_resolves_to_base_section(tmp_path, monkeypatch):
    rc = run(
        tmp_path, monkeypatch,
        PROVEN_HEADER + '| Claim A | flat **2.6 KB** | ✅ proven | "Edge pilot → streaming memory" |\n',
        "# B\n\n## Edge pilot: battery SoH (2026-01-01)\n\n### streaming memory\n\nflat 2.6 KB.\n",
    )
    assert rc == 0


def test_where_table_without_number_column_only_needs_anchor(tmp_path, monkeypatch):
    results = """# R

## PENDING

| Line | Status | Where |
|---|---|---|
| thing | pending | "Latent recursion" in BENCHMARKS.md + `docs/LATENT.md` |
"""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "LATENT.md").write_text("x", encoding="utf-8")
    rc = run(
        tmp_path, monkeypatch, results,
        "# B\n\n## Latent recursion — ledger\n\npending.\n",
    )
    assert rc == 0


def test_real_repo_ledger_smoke():
    """真实账本冒烟: 任何 RESULTS→BENCHMARKS 锚点漂移都会让 CI 的 full-test 红。"""
    rc = check_claims.main([])
    assert rc == 0, "RESULTS.md 账本存在不可溯源行 — 见上方 check_claims 输出"
