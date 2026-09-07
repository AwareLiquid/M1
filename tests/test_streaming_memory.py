"""tests/test_streaming_memory.py — 流式记忆层协议测试（CPU，零主张）"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mt_lnn.memory_broker.streaming import (  # noqa: E402
    FixedStateBackend,
    InMemoryBackend,
    StreamingSession,
)


def test_write_recall_roundtrip():
    s = StreamingSession(InMemoryBackend())
    assert s.write("alice", "likes tea beijing") is True
    hits = s.recall("tea")
    assert hits and hits[0][0] == "alice" and hits[0][1] == "likes tea beijing"


def test_dedup_short_circuit():
    b = InMemoryBackend()
    s = StreamingSession(b)
    assert s.write("k", "same value") is True
    assert s.write("k", "same value") is False, "重复 (key,value) 必须短路"
    assert b.dedup_count == 1


def test_salience_floor_filters():
    s = StreamingSession(InMemoryBackend(), salience_floor=0.5)
    assert s.write("low", "v", salience=0.2) is False, "低于 salience 门不入记忆"
    assert s.write("high", "v", salience=0.9) is True
    assert s.recall("v") == [] or all(k != "low" for k, _, _ in s.recall("v"))


def test_snapshot_restore_bit_exact():
    s = StreamingSession(InMemoryBackend())
    for i in range(10):
        s.write(f"k{i}", f"value {i} alpha")
    blob = s.snapshot()
    s.write("pollute", "post snapshot junk")
    s.restore(blob)
    assert s.state_bytes() == len(blob), "restore 后 state_bytes 必须与快照一致"
    hits = s.recall("alpha", k=20)
    assert len(hits) == 10 and all(k != "pollute" for k, _, _ in hits)
    s2_blob = s.snapshot()
    assert s2_blob == blob, "恢复后再快照应与原快照逐字节一致（bit-exact）"


def test_session_namespace_isolation():
    a = StreamingSession(InMemoryBackend(), session="A")
    b = StreamingSession(InMemoryBackend(), session="B")
    a.write("mood", "session A secret blue")
    b.write("mood", "session B secret red")
    ha = a.recall("secret")
    hb = b.recall("secret")
    assert len(ha) == 1 and "blue" in ha[0][1]
    assert len(hb) == 1 and "red" in hb[0][1]
    assert all(k == "mood" for k, _, _ in ha), "会话命名空间对调用方透明"


def test_o1_fixed_backend_constant():
    s = StreamingSession(FixedStateBackend(capacity=64))
    r = s.o1_check(n_writes=50)
    assert r["constant"] is True, f"定长后端 state_bytes 必须恒定: {r}"
    assert r["written"] == 50


def test_o1_growing_backend_not_constant():
    s = StreamingSession(InMemoryBackend())
    r = s.o1_check(n_writes=50)
    assert r["constant"] is False, (
        "参考后端状态随写入增长——这个 False 是诚实的：O(1) 主张只属于"
        "定长状态后端（parametric），参考实现不得冒充")
    assert r["growth_bytes"] > 0


def test_o1_fixed_capacity_refuses_overflow():
    b = FixedStateBackend(capacity=4)
    for i in range(4):
        assert b.write(f"k{i}", f"v{i}") is True
    assert b.write("k-overflow", "v") is False, "容量满必须拒绝，不假装还能扩"


def test_recall_no_match_returns_empty():
    s = StreamingSession(InMemoryBackend())
    s.write("k", "totally different words")
    assert s.recall("zzz-no-match") == []
