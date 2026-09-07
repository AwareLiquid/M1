"""MemoryBroker — the single memory interface M1 exposes to Awareness.

The dependency contract between the two repos (see docs/MEMORY_BROKER.md):
Awareness-SDK (the product/distribution repo) may call INTO this package over
its installed interface, but this repo never imports SDK source — the only
cross-repo runtime channel is the SDK local daemon's HTTP surface, wrapped by
:class:`mt_lnn.memory_broker.external_broker.ExternalBroker`. Keeping every
backend behind ONE abstract interface is what makes the two-repo split safe:
the recall chain on the SDK side can swap backends without code changes, and
the four-competency benchmark (benchmarks/persistent_memory/) can score
parametric vs external-daemon vs graph with an identical workload.

The method signatures deliberately mirror the calling habits both sides
already have:

* M1's :class:`mt_lnn.parametric_memory.ParametricMemory` —
  ``write(session_id, key, value)`` / ``recall(session_id, query, top_k)`` /
  ``forget`` / ``snapshot`` / ``restore`` / ``state_bytes``;
* the SDK's memory tools — an explicit session scope, a free-form query
  string, and a hit limit (its ``awareness_recall`` query/limit shape).

Key/value domain: ``int`` (token id) or ``str`` (text) — the domain
ParametricMemory defines. Backends that are natively text-only flatten the
same domain into their record shape (see each backend's docstring for the
exact mapping). Capabilities differ by backend BY DESIGN — the O(1)-state
backend cannot offer unbounded storage and the unbounded store cannot offer
bit-exact state. Methods a backend cannot honour raise ``NotImplementedError``
(with the honest reason in the message); the capability matrix lives in
docs/MEMORY_BROKER.md so callers can dispatch on documented facts instead of
exception handling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Union

MemoryKey = Union[int, str]
MemoryQuery = Union[int, str]

# A snapshot is a JSON-safe dict; bit-exactness is a PER-BACKEND property,
# stated in the backend docstring and docs/MEMORY_BROKER.md, not by this
# type alias.
Snapshot = Dict[str, Any]


class BrokerError(RuntimeError):
    """A broker backend failed at runtime (transport, storage, daemon down).

    Raised instead of letting raw urllib/sqlite errors leak through the
    interface, so callers can catch one exception type. The original
    exception is chained (``raise BrokerError(...) from exc``)."""


@dataclass(frozen=True)
class RecallHit:
    """One recall result.

    ``value`` is the decoded payload (the ``value`` a write bound, or the
    backend's native record content). ``score`` is backend-native similarity
    when the backend exposes one (cosine for the parametric read, accumulated
    activation for the graph walk); ``None`` when the backend's retrieval
    surface returns no numeric score over its API (the external daemon's
    cascade returns ranked items only).
    """

    value: Any
    score: Optional[float] = None
    id: Optional[str] = None
    meta: Optional[Any] = None


class MemoryBroker(ABC):
    """Abstract interface every memory backend presents to callers.

    Sessions are explicit first-class strings on every method — both the
    parametric (per-session (F, z) state) and external (daemon ``session_id``
    column) backends are session-scoped natively, and the graph backend
    implements the same scope with one store per session.
    """

    #: registry key used by :func:`mt_lnn.memory_broker.factory.create_broker`
    kind: ClassVar[str] = ""

    @abstractmethod
    def write(self, session_id: str, key: MemoryKey, value: Any, *,
              meta: Optional[Any] = None) -> str:
        """Bind ``key -> value`` in ``session_id``; return a record handle.

        The handle is backend-native where the backend has records (the
        daemon's ``mem_...`` id, the graph's node id) and derived where it
        does not (the parametric store keeps no id index — its handle is the
        canonical string of the key). ``meta`` rides along where the backend
        has a metadata field; it is silently dropped where it does not.
        """

    @abstractmethod
    def recall(self, session_id: str, query: MemoryQuery,
               top_k: int = 5) -> List[RecallHit]:
        """Return up to ``top_k`` hits for ``query``, best first.

        An empty or never-written session returns ``[]`` — "no memory of
        it", never a stale guess (the ParametricMemory contract).
        """

    @abstractmethod
    def forget(self, session_id: str, key: Optional[MemoryKey] = None) -> bool:
        """Erase ONE binding (``key`` given) or the WHOLE session (``None``).

        Selective forgetting is the parametric backend's algebraic
        differentiator; backends without the capability raise
        ``NotImplementedError`` (see docs/MEMORY_BROKER.md). Returns True if
        the session existed, False if it did not.
        """

    @abstractmethod
    def snapshot(self, session_id: str) -> Snapshot:
        """Serialize one session's state to a JSON-safe dict.

        Bit-exactness (or its absence) is documented per backend. Raises
        KeyError for an unknown session where the backend can tell.
        """

    @abstractmethod
    def restore(self, session_id: str, snap: Snapshot) -> None:
        """Inverse of :meth:`snapshot` (same backend, same version)."""

    @abstractmethod
    def state_bytes(self, session_id: str) -> int:
        """Size of the session's memory body in bytes.

        The comparison metric between backends: constant-in-writes for the
        parametric state (the O(1) claim), the SQLite file size for the
        graph, and the stored content bytes for the external daemon. Returns
        0 for an unknown session.
        """

    def close(self) -> None:
        """Release backend resources. Safe to call twice; default no-op."""

    def __enter__(self) -> "MemoryBroker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
