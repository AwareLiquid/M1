"""atomic_io.py — 并发安全的原子落盘(每 writer 独占 tmp 名)。

tmp 名为什么必须带 nonce(实测,不是推理)
-------------------------------------------------------
``os.replace`` 只保证"发布"原子——它完全不说被发布的 tmp 是谁填的。
固定 tmp 名(path + ".tmp")被同一目标的所有并发 writer 共享:两个 writer
以 O_TRUNC 打开同一 tmp,各自写不同偏移,各自把对方留下的内容 publish
出去——目标文件变成两个 writer 的拼接体。

这不是推理,是实测。aexp CHANGELOG 0.6.1 存档了同形态实现的 A/B 对比
(Linux 5.14, xfs + NFS, 2/4 并发 writer, 150 目标 × 3 种 flush 形态 = 1350):

  ==================  ======================  ================
  实现                 撕裂目标数               replace 异常
  ==================  ======================  ================
  固定 .tmp           751 / 1350(全配置)     ~1 次/轮
  每 writer 独占 tmp  0 / 1350                0
  ==================  ======================  ================

per-config 撕裂率随文件系统/writer 数/负载波动(1.3%–98%),但稳定的结论
是:每一配置都撕裂了,修后全零。本仓 test_atomic_io.py::concurrent_writers_no_tear
(8 线程 × 25 轮写同一目标)从本模块交付起 0 torn。

nonce 为什么用 time_ns 而不是 uuid4
------------------------------------
aexp 用 ``uuid.uuid4().hex[:8]``(由构造保证唯一,但需要 import uuid)。
本模块用 ``time.time_ns()``:同进程两线程在同一纳秒调用的概率为
~1/10⁹ × 核心数,实测 8 线程 × 25 轮 = 200 次调用无碰撞。如果未来扩展到
多机共享文件系统(当前不存在),再换 uuid4——uuid 的唯一性是构造性的,
time_ns 是概率性的,多机时钟不同步时概率性不够。

fsync 为什么写(aexp 没写)
---------------------------
aexp 的场景是"多 writer 争抢 last-writer-wins"——fsync 对争抢无帮助。
本模块的场景是"实验结果必须扛住机器崩溃"——OS buffer 里的数据在
crash 时丢失,fsync 确保 replace 前数据已落盘。多一次系统调用换
crash-safety,对本仓(24 进程并行 × 长跑 × 不可靠算力)是对的取舍。

失败清理为什么用 BaseException 而非 Exception
------------------------------------------------
KeyboardInterrupt 打断在 write 与 replace 之间,恰好是唯一命名的 tmp
永久残留的场景。Exception 不捕获 KeyboardInterrupt——BaseException 捕获。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

__all__ = ["atomic_write_bytes", "atomic_write_text", "atomic_write_json"]


def _tmp_name(path: Path) -> Path:
    """每次调用生成独占 tmp 名:<dest>.<pid>.<nonce>.tmp。

    pid 供人排查硬杀残留;time_ns 提供实际唯一性(见模块 docstring)。
    """
    nonce = time.time_ns()
    return path.with_name(f"{path.name}.{os.getpid()}.{nonce}.tmp")


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    """原子写二进制:独占 tmp → fsync → os.replace;失败清理自身 tmp。"""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_name(dest)
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | os.PathLike[str], obj: Any, indent: int = 1) -> None:
    atomic_write_text(path, json.dumps(obj, indent=indent, ensure_ascii=False))
