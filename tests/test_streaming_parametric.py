"""tests/test_streaming_parametric.py — parametric 后端 × 流式协议集成测试（CPU，零主张）

钉三件事：
1. o1_check 对 parametric 后端 constant=True（会话存在前提下 state_bytes 恒定
   ——H004 判据 2 的代码面实测载体；空会话=0 的边缘单独断言）；
2. 快照/回滚经 canonical JSON bytes 往返后 bit-exact；
3. recall 返回 (None, value, score) 形状（parametric 读面无 key 的文档化差异）。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

torch = pytest.importorskip("torch", reason="parametric 后端依赖 torch")

from mt_lnn.memory_broker.adapters import ParametricStreamingBackend  # noqa: E402
from mt_lnn.memory_broker.streaming import StreamingSession  # noqa: E402


@pytest.fixture()
def session():
    backend = ParametricStreamingBackend(d_mem=32, vocab_size=2000)
    s = StreamingSession(backend)
    s.write("warmup", "warmup entry establishes the session", salience=1.0)
    return s


def test_state_bytes_empty_session_is_zero():
    b = ParametricStreamingBackend(d_mem=32, vocab_size=2000)
    assert b.state_bytes() == 0, "空会话 state_bytes=0（懒创建边缘，文档化）"


def test_o1_constant_on_parametric(session):
    r = session.o1_check(n_writes=50)
    assert r["constant"] is True, (
        f"parametric state_bytes 必须与写入次数无关: {r}")
    assert r["written"] == 50
    assert r["state_bytes_before"] > 0, "warmup 后会话应已建立"
    # 恒定值 = (d²+d)×元素字节数：d=32, fp32 → (1024+32)*4 = 4224
    assert r["state_bytes_before"] == r["state_bytes_after"] == (32 * 32 + 32) * 4


def test_snapshot_restore_bit_exact(session):
    for i in range(8):
        session.write(f"fact{i}", f"项目 alpha{i} 的登记码是 {i:04d}", salience=1.0)
    blob = session.snapshot()
    session.write("pollute", "post snapshot junk entry", salience=1.0)
    session.restore(blob)
    # state_bytes 是 (F,z) 张量本体(定长), 与快照 JSON 字节数是两个量, 不应相等
    assert session.snapshot() == blob, "恢复后再快照应逐字节一致（bit-exact）"


def test_recall_roundtrip_shapes(session):
    session.write("doc1", "quantum entanglement overview", salience=1.0)
    hits = session.recall("entanglement", k=3)
    assert hits, "召回不应为空"
    key, value, score = hits[0]
    assert key is None, "parametric 读面不返回 key（文档化差异）"
    assert "entanglement" in str(value)
    assert score is not None, "parametric 暴露原生相似度"
    # 分数符号不保证：cosine 对累积 (F,z) 可为负——只断言数值存在


def test_forget_removes_binding(session):
    """forget 后按 key 召回 = []（"no memory of it"，padding 塌缩）——对齐
    test_memory_broker.py:130 的房内口径；键含 StreamingSession 命名空间前缀。"""
    session.write("temp", "temporary entry to forget", salience=1.0)
    pre = session.backend.recall("default::temp", k=3)
    assert pre and pre[0][1] == "temporary entry to forget"
    assert session.backend.forget("default::temp") is True
    assert session.backend.recall("default::temp", k=3) == []
    # 会话存在但键不存在：delta 投影 ~no-op，代数上无法检测缺席 → 返回 True
    # （parametric_memory.py forget docstring 口径）；False 仅在会话不存在时出现
    assert session.backend.forget("default::ghost") is True

