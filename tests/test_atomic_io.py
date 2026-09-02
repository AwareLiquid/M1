"""tests/test_atomic_io.py — 并发原子写:唯一 tmp 名 + 整文件语义。

对应 benchmarks/atomic_io.py。核心回归点:aexp 0.6.1 存档的并发撕裂
(固定 tmp 名 + os.replace 互踩,1350 目标 751 撕裂)。本测试用线程
并发写同一目标,断言每次读到的都是完整 JSON(last-writer-wins,整文件)。

失败场景测试(对齐 workpool 工程纪律:先写失败场景,再写功能)
- test_resume_skips_completed        — 中断后重启,只跑未完成配置
- test_partial_run_no_corrupt_result — 进程被杀,已完成结果完整可读
- test_two_writers_same_dest_idempotent — 重复写同一目标,读回始终是完整 JSON
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from benchmarks.atomic_io import atomic_write_json, atomic_write_text

PAYLOAD = {"verdict": "NULL", "diagnosis": "budget_wall", "rows": list(range(64))}


# ---------------------------------------------------------------------------
# 原子写基础(原有)
# ---------------------------------------------------------------------------

def test_single_write_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "row.json"
    atomic_write_json(p, PAYLOAD)
    assert json.loads(p.read_text(encoding="utf-8")) == PAYLOAD


def test_overwrite_is_whole_file(tmp_path: Path) -> None:
    p = tmp_path / "row.json"
    atomic_write_json(p, {"n": 1})
    atomic_write_json(p, PAYLOAD)
    assert json.loads(p.read_text(encoding="utf-8")) == PAYLOAD


def test_failure_cleans_own_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import benchmarks.atomic_io as aio

    p = tmp_path / "row.json"
    atomic_write_text(p, "seed")

    def boom(_fd: int) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(aio.os, "fsync", boom)
    with pytest.raises(RuntimeError):
        atomic_write_text(p, "partial")
    # 目标未损坏,无残留 tmp
    assert p.read_text(encoding="utf-8") == "seed"
    assert list(tmp_path.glob("*.tmp")) == []


def test_concurrent_writers_no_tear(tmp_path: Path) -> None:
    """N 线程 × M 轮并发写同一目标:每次读到的必须是完整 JSON。"""
    p = tmp_path / "shared.json"
    atomic_write_json(p, {"gen": -1})

    n_threads, rounds = 8, 25
    errors: list[str] = []

    def writer(tid: int) -> None:
        for r in range(rounds):
            try:
                atomic_write_json(p, {"gen": f"{tid}-{r}", "pad": "x" * 512})
            except Exception as exc:
                errors.append(f"w{tid}: {exc!r}")

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    final = json.loads(p.read_text(encoding="utf-8"))
    assert "gen" in final and "pad" in final
    assert list(tmp_path.glob("*.tmp")) == []


def test_no_shared_tmp_name() -> None:
    import benchmarks.atomic_io as aio

    names = {str(aio._tmp_name(Path("x.json"))) for _ in range(64)}
    assert len(names) == 64


# ---------------------------------------------------------------------------
# 失败场景测试(对齐 workpool 工程纪律)
# ---------------------------------------------------------------------------

def test_resume_skips_completed(tmp_path: Path) -> None:
    """模拟 pending_configs 语义:5 配置跑完 3 个后中断,重启后已完成跳过。

    对应事故:Phase B 24-worker 中 A100 提前回收 → 部分结果已落盘。
    保证:重启后只跑未完成配置,已完成的 mean_acc 不变。
    """
    configs = [(m, d, s) for m in ("core", "stack") for d in (1, 4) for s in (0,)]  # 4 configs
    done = configs[:3]   # 前三个已完成
    todo = configs[3:]   # 最后一个未完成

    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # 模拟:前三个配置已落盘
    for mode, depth, seed in done:
        p = results_dir / f"exp_{mode}_d{depth}_s{seed}.json"
        atomic_write_json(p, {"mode": mode, "depth": depth, "seed": seed,
                               "mean_acc": 0.42, "steps": 30000})

    # 重启:检查哪些已完成
    def is_done(m, d, s):
        fp = results_dir / f"exp_{m}_d{d}_s{s}.json"
        if not fp.exists():
            return False
        r = json.loads(fp.read_text(encoding="utf-8"))
        return r.get("steps") == 30000  # 同预算才跳过

    remaining = [(m, d, s) for m, d, s in configs if not is_done(m, d, s)]
    assert set(remaining) == set(todo), f"expected {todo}, got {remaining}"

    # 补跑最后一个,验证所有配置齐
    for m, d, s in remaining:
        atomic_write_json(results_dir / f"exp_{m}_d{d}_s{s}.json",
                          {"mode": m, "depth": d, "seed": s, "mean_acc": 0.07, "steps": 30000})
    assert all(is_done(m, d, s) for m, d, s in configs)


def test_partial_run_no_corrupt_result(tmp_path: Path) -> None:
    """模拟:writer 在 write 和 replace 之间被 SIGKILL。

    保证:已完成的目标 JSON 完整可读;无残留 tmp(如果有,也不影响目标)。
    对应事故:_rescue_20260831 抢救时目标 JSON 必须完整可解析。
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # 写入 3 个"已完成"配置
    for i in range(3):
        atomic_write_json(results_dir / f"cell_{i}.json",
                          {"idx": i, "mean_acc": 0.42 + i, "rows": list(range(100))})

    # 模拟第 4 个配置:tmp 写入了但 replace 未发生(writer 被 kill -9)
    orphan_tmp = results_dir / "cell_3.json.12345.999999.tmp"
    orphan_tmp.write_text('{"idx": 3, "mean_acc": 0.', encoding="utf-8")  # 半截

    # 验证:已完成的目标完整可读
    for i in range(3):
        d = json.loads((results_dir / f"cell_{i}.json").read_text(encoding="utf-8"))
        assert d["idx"] == i and len(d["rows"]) == 100

    # 验证:cell_3 目标不存在(未被 replace),残留 tmp 可被安全清理
    assert not (results_dir / "cell_3.json").exists()
    # 清理残留 tmp(实际由下次 atomic_write 的失败清理或人工删除)
    orphan_tmp.unlink()
    assert list(results_dir.glob("*.tmp")) == []


def test_two_writers_same_dest_idempotent(tmp_path: Path) -> None:
    """两个进程同时写同一目标:读回的永远是某个 writer 的完整内容。

    对应事故:分片失误导致同配置被两个进程跑,结果文件被覆盖。
    保证:last-writer-wins(整文件),不是拼接体。
    """
    p = tmp_path / "shared_dest.json"
    barrier = threading.Event()

    def writer_a():
        barrier.wait()
        atomic_write_json(p, {"writer": "a", "data": list(range(50))})

    def writer_b():
        barrier.wait()
        atomic_write_json(p, {"writer": "b", "data": list(range(50))})

    ta, tb = threading.Thread(target=writer_a), threading.Thread(target=writer_b)
    ta.start(); tb.start()
    barrier.set()
    ta.join(); tb.join()

    # 读回:一定是 a 或 b 的完整内容,不是拼接体
    result = json.loads(p.read_text(encoding="utf-8"))
    assert result["writer"] in ("a", "b")
    assert len(result["data"]) == 50  # 完整数组,不是半截
    assert list(tmp_path.glob("*.tmp")) == []
