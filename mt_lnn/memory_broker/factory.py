"""create_broker — the one constructor callers are meant to remember.

    from mt_lnn.memory_broker import create_broker

    broker = create_broker("parametric", d_mem=128, update_rule="sum")
    broker = create_broker("graph", key_dim=64, auto_link=True)
    broker = create_broker("external", base_url="http://127.0.0.1:37800")

Kwargs forward untouched to the backend constructor; see each backend's
docstring for its parameters and docs/MEMORY_BROKER.md for the capability
matrix that should drive the choice.
"""

from __future__ import annotations

from typing import Any, Dict, Type

from .base import MemoryBroker
from .external_broker import ExternalBroker
from .graph_broker import GraphBroker
from .parametric_broker import ParametricBroker

_REGISTRY: Dict[str, Type[MemoryBroker]] = {
    ParametricBroker.kind: ParametricBroker,
    GraphBroker.kind: GraphBroker,
    ExternalBroker.kind: ExternalBroker,
}


def create_broker(kind: str, **kwargs: Any) -> MemoryBroker:
    """Build a broker by registry kind; kwargs forward to the backend."""
    try:
        cls = _REGISTRY[kind]
    except KeyError:
        raise ValueError(
            f"unknown broker kind {kind!r}; known kinds: {sorted(_REGISTRY)}"
        ) from None
    return cls(**kwargs)
