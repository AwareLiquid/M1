"""mt_lnn/memory_broker/adapters.py — 后端 → 流式协议适配器（机制轨，零主张）

把 session 作用域的 MemoryBroker 后端适配成 StreamingSession（streaming.py）
消费的 BrokerProtocol 形状。目前只有 parametric 一个适配器——graph/external
按需再补（ADJ-009 纪律：不做没人用的预设计）。

文档化的形状差异（诚实边界，不是丢弃）：
- parametric 的 recall 读面天然返回 (value, score)，**不返回 key** —— 适配后
  Hit = (None, value, score)；
- parametric 的 write 无 dedup 短路、无 salience 存储字段 —— write 恒返回
  True，salience 被忽略（对齐 base.py 对 meta 的处理口径）；
- 快照是 JSON-safe dict（base64 fp32，bit-exact），适配成 canonical JSON
  bytes（sort_keys）以匹配 BrokerProtocol 的 bytes 语义；
- state_bytes：空会话为 0，首次写入后恒定（d_mem²+d_mem 字节）——O(1) 主张
  以"会话存在"为前提，测试里以 warmup 写入钉住此前提。
"""

from __future__ import annotations

import json
from typing import Any

from mt_lnn.memory_broker.parametric_broker import ParametricBroker


class ParametricStreamingBackend:
    """ParametricBroker → BrokerProtocol（streaming.py）单会话适配器。"""

    def __init__(self, session_id: str = "default",
                 broker: ParametricBroker | None = None, **memory_kwargs) -> None:
        self.session_id = session_id
        self.broker = broker if broker is not None else ParametricBroker(**memory_kwargs)

    # -- BrokerProtocol: write/forget/recall -------------------------------

    def write(self, key: Any, value: Any, salience: float = 1.0) -> bool:
        del salience  # parametric (F, z) 无 salience 存储字段（文档化差异）
        self.broker.write(self.session_id, key, value)
        return True

    def forget(self, key: Any) -> bool:
        return self.broker.forget(self.session_id, key)

    def recall(self, query: Any, k: int = 5) -> list[tuple]:
        hits = self.broker.recall(self.session_id, query, top_k=k)
        return [(None, h.value, h.score) for h in hits]

    # -- BrokerProtocol: snapshot/restore/state_bytes -----------------------

    def snapshot(self) -> bytes:
        snap = self.broker.snapshot(self.session_id)
        return json.dumps(snap, sort_keys=True, ensure_ascii=False).encode("utf-8")

    def restore(self, blob: bytes) -> None:
        self.broker.restore(self.session_id, json.loads(blob.decode("utf-8")))

    def state_bytes(self) -> int:
        return self.broker.state_bytes(self.session_id)
