"""mt_lnn.memory_broker — one memory interface, three backends.

Public surface:

    MemoryBroker, RecallHit, BrokerError   (the contract — base.py)
    ParametricBroker                        (fast-weight (F, z) state, O(1))
    GraphBroker                             (knowledge graph, multi-hop + lifecycle)
    ExternalBroker                          (Awareness local daemon HTTP)
    create_broker(kind, **kwargs)           (registry constructor)

This package is the interface M1 exposes to Awareness-SDK (SDK→M1 one-way;
M1 reaches the SDK only over its local daemon HTTP — see base.py's module
docstring and docs/MEMORY_BROKER.md).
"""

from .base import BrokerError, MemoryBroker, RecallHit, Snapshot
from .external_broker import ExternalBroker
from .factory import create_broker
from .graph_broker import GraphBroker
from .parametric_broker import ParametricBroker

__all__ = [
    "BrokerError",
    "ExternalBroker",
    "GraphBroker",
    "MemoryBroker",
    "ParametricBroker",
    "RecallHit",
    "Snapshot",
    "create_broker",
]
