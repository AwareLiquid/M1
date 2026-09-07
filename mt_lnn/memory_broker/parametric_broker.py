"""ParametricBroker — MemoryBroker protocol adapter over ParametricMemory.

PURE PROTOCOL ADAPTATION, ZERO ALGORITHM CHANGES: every call forwards to the
existing :class:`mt_lnn.parametric_memory.ParametricMemory` (the fast-weight
(F, z) runtime; docs/PARAMETRIC_MEMORY.md). No write rule, decay, read path,
or state layout is touched here — this file exists so callers program against
the :class:`~mt_lnn.memory_broker.base.MemoryBroker` interface can use the
parametric backend without seeing its concrete class.

Capability notes (full matrix in docs/MEMORY_BROKER.md):

* ``snapshot``/``restore`` are BIT-EXACT — raw fp32 bytes base64 through the
  session envelope, unit-tested in tests/test_parametric_memory.py and
  re-asserted here through the broker face.
* ``state_bytes`` is CONSTANT in the number of writes — the O(1) claim.
* ``forget(key)`` is single-binding algebraic removal (delta projection);
  no binding index exists.
* ``write`` returns a DERIVED handle (the canonical string of the key), not
  a stored id: the parametric store keeps no id index by design (an index
  would be O(n) state and undermine the O(1) claim).
"""

from __future__ import annotations

from typing import Any, List, Optional

from ..parametric_memory import ParametricMemory
from .base import BrokerError, MemoryBroker, MemoryKey, RecallHit, Snapshot


class ParametricBroker(MemoryBroker):
    """MemoryBroker face over the fast-weight parametric store."""

    kind = "parametric"

    def __init__(self, memory: Optional[ParametricMemory] = None,
                 **memory_kwargs) -> None:
        """Wrap an existing ParametricMemory, or build one from kwargs.

        Passing an existing instance is the intended reuse path (the caller
        owns the hyperparameters — d_mem, update_rule, decay); kwargs are a
        convenience that forwards untouched to the ParametricMemory
        constructor.
        """
        self.memory = memory if memory is not None else ParametricMemory(**memory_kwargs)

    # ------------------------------------------------------------------ ...

    def write(self, session_id: str, key: MemoryKey, value, *,
              meta: Optional[Any] = None) -> str:
        """Forward to ParametricMemory.write; ``meta`` has no storage field
        in the (F, z) state and is dropped (documented limitation, not a
        silent design choice — see docs/MEMORY_BROKER.md)."""
        try:
            self.memory.write(session_id, key, value)
        except (TypeError, ValueError) as exc:
            raise BrokerError(f"parametric write rejected: {exc}") from exc
        return key if isinstance(key, str) else str(key)

    def recall(self, session_id: str, query: MemoryQuery,
               top_k: int = 5) -> List[RecallHit]:
        pairs = self.memory.recall(session_id, query, top_k=top_k)
        # [(None, 0.0), ...] padding means "no memory of it" — collapse to [].
        return [RecallHit(value=v, score=s) for v, s in pairs if v is not None]

    def forget(self, session_id: str, key: Optional[MemoryKey] = None) -> bool:
        return self.memory.forget(session_id, key)

    def snapshot(self, session_id: str) -> Snapshot:
        snap = self.memory.snapshot(session_id)
        if snap is None:
            raise KeyError(f"no session {session_id!r} to snapshot")
        return snap

    def restore(self, session_id: str, snap: Snapshot) -> None:
        self.memory.restore(session_id, snap)

    def state_bytes(self, session_id: str) -> int:
        return self.memory.state_bytes(session_id)

    def close(self) -> None:   # no external resources
        pass
