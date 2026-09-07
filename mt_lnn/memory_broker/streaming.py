"""mt_lnn/memory_broker/streaming.py — H004 流式会话记忆层（机制轨，零主张）

把 memory_broker 后端包成跨会话流式会话对象，供 Bet A（docs/RESEARCH_PLAN.md
§2.1）的 1M 流式上下文评测使用。设计要点：

- 后端协议 BrokerProtocol（duck-type）：write/forget/recall/snapshot/restore/
  state_bytes。生产后端 = broker 既有的 parametric/graph/external；集成时以
  适配器对齐真实 API（本文件不 import 具体后端，避免循环依赖）。
- InMemoryBackend：纯 python 参考后端——协议钉版与测试用，**它的状态随写入
  增长，不是 O(1) 载体**（这一点被测试显式钉住，防止"参考实现冒充主张"）。
- FixedStateBackend：定长槽位演示后端——展示 state_bytes 恒定应满足什么契约，
  对齐 parametric 后端已测的"50 次写入前后 state_bytes 相等"性质。
- StreamingSession：会话命名空间 + salience 门 + bit-exact 快照回滚 + o1_check
  （H004 判据 2 的代码面检测器——数字判定仍在评测脚本，运行前写死）。

零主张：本文件不携带任何实验数字；判定内容归判定轨。
"""

from __future__ import annotations

import pickle
from typing import Any, Protocol

Hit = tuple[str, Any, float]  # (key, value, score)


class BrokerProtocol(Protocol):
    """memory_broker 后端最小协议（生产后端以适配器对齐）。"""

    def write(self, key: str, value: Any, salience: float = 1.0) -> bool:
        """写入一条记忆；重复 (key, value) 应短路返回 False。"""
        ...

    def forget(self, key: str) -> bool:
        ...

    def recall(self, query: str, k: int = 5) -> list[Hit]:
        ...

    def snapshot(self) -> bytes:
        """bit-exact 状态快照（restore(snapshot()) 后行为逐字节一致）。"""
        ...

    def restore(self, blob: bytes) -> None:
        ...

    def state_bytes(self) -> int:
        """当前状态占用字节数（O(1) 检测的对象）。"""
        ...


class InMemoryBackend:
    """纯 python 参考后端：dict 存储，状态随写入增长（**非 O(1)**，测试钉住）。

    recall = 关键词重叠计分（确定性，无随机性），只保证协议可用性，
    不承载检索质量主张。
    """

    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float]] = {}
        self.dedup_count = 0

    def write(self, key: str, value: Any, salience: float = 1.0) -> bool:
        old = self._store.get(key)
        if old is not None and old[0] == value:
            self.dedup_count += 1
            return False  # dedup 短路
        self._store[key] = (value, salience)
        return True

    def forget(self, key: str) -> bool:
        return self._store.pop(key, None) is not None

    def recall(self, query: str, k: int = 5) -> list[Hit]:
        q = set(str(query).split())
        scored: list[Hit] = []
        for key, (value, sal) in self._store.items():
            overlap = len(q & set(str(value).split()))
            if overlap:
                scored.append((key, value, overlap * max(sal, 0.0)))
        scored.sort(key=lambda h: (-h[2], h[0]))
        return scored[:k]

    def snapshot(self) -> bytes:
        return pickle.dumps({"sorted_items": sorted(self._store.items()),
                             "dedup_count": self.dedup_count})

    def restore(self, blob: bytes) -> None:
        data = pickle.loads(blob)
        self._store = dict(data["sorted_items"])
        self.dedup_count = data["dedup_count"]

    def state_bytes(self) -> int:
        return len(self.snapshot())


class FixedStateBackend:
    """定长槽位演示后端：state_bytes 与写入条数无关（到容量上限为止）。

    契约对齐 parametric 后端已测性质（写入前后 state_bytes 相等）；
    容量外的写入被拒绝（返回 False）——诚实于固定容量的边界。
    """

    def __init__(self, capacity: int = 64) -> None:
        self._capacity = capacity
        self._slots: dict[str, Any] = {}
        self.dedup_count = 0

    def write(self, key: str, value: Any, salience: float = 1.0) -> bool:
        if key in self._slots:
            if self._slots[key] == value:
                self.dedup_count += 1
                return False
            self._slots[key] = value
            return True
        if len(self._slots) >= self._capacity:
            return False  # 定长：容量满即拒绝，不假装 O(1) 还能扩
        self._slots[key] = value
        return True

    def forget(self, key: str) -> bool:
        return self._slots.pop(key, None) is not None

    def recall(self, query: str, k: int = 5) -> list[Hit]:
        q = set(str(query).split())
        scored = [(key, v, len(q & set(str(v).split())))
                  for key, v in self._slots.items()]
        scored = [h for h in scored if h[2] > 0]
        scored.sort(key=lambda h: (-h[2], h[0]))
        return scored[:k]

    def snapshot(self) -> bytes:
        return pickle.dumps({"cap": self._capacity,
                             "sorted_items": sorted(self._slots.items())})

    def restore(self, blob: bytes) -> None:
        data = pickle.loads(blob)
        self._capacity = data["cap"]
        self._slots = dict(data["sorted_items"])

    def state_bytes(self) -> int:
        # 定长结构：header + capacity×槽位上限，与当前条数无关
        return 64 + self._capacity * 128


class StreamingSession:
    """跨会话流式会话：会话命名空间 + salience 门 + 快照回滚 + O(1) 检测。"""

    def __init__(self, backend: BrokerProtocol, *,
                 salience_floor: float = 0.0, session: str = "default") -> None:
        self.backend = backend
        self.salience_floor = salience_floor
        self.session = session

    def _ns(self, key: str) -> str:
        return f"{self.session}::{key}"

    def write(self, key: str, value: Any, *, salience: float = 1.0) -> bool:
        if salience < self.salience_floor:
            return False  # salience 门：低显著度不入记忆（写入纪律）
        return self.backend.write(self._ns(key), value, salience)

    def recall(self, query: str, *, k: int = 5) -> list[Hit]:
        hits = self.backend.recall(query, k=k)
        prefix = f"{self.session}::"
        return [(key[len(prefix):] if isinstance(key, str) and key.startswith(prefix)
                 else key, v, s)
                for key, v, s in hits]

    def snapshot(self) -> bytes:
        return self.backend.snapshot()

    def restore(self, blob: bytes) -> None:
        self.backend.restore(blob)

    def state_bytes(self) -> int:
        return self.backend.state_bytes()

    def o1_check(self, n_writes: int = 200) -> dict:
        """H004 判据 2 的代码面检测：n 次写入前后 state_bytes 是否恒定。

        返回原始测量，不做"是否 O(1)"的裁决（数字判定在评测脚本，运行前写死）。
        """
        before = self.state_bytes()
        written = 0
        for i in range(n_writes):
            if self.write(f"o1probe::{i}", f"probe payload {i} 0123456789", salience=1.0):
                written += 1
        after = self.state_bytes()
        return {
            "n_writes": n_writes,
            "written": written,
            "state_bytes_before": before,
            "state_bytes_after": after,
            "growth_bytes": after - before,
            "constant": before == after,
        }
