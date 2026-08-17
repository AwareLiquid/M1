"""
mt_lnn/adapter_homeostasis.py -- SHY synaptic downscaling for MT adapters.

What this is for (sleep cycle, plan B)
--------------------------------------
The sleep module's NREM stage (plan A, ``session_consolidation.py``) ships salient
working-state into long-term memory. Its complementary stage is **synaptic
downscaling / SHY** (Tononi & Cirelli 2014, "Synaptic Homeostasis Hypothesis"):
wake *potentiates* synapses, and sleep renormalises them *proportionally* downward
-- preserving the relative weight pattern (the memory) while shrinking total
synaptic weight (restoring capacity and signal-to-noise).

This module applies that downscaling to the M1 route's **trainable MT residual
adapters ONLY** -- never the frozen base model. That separation is the whole point
on the adapter route: the base keeps its language ability untouched (frozen), and
homeostasis acts only on the small adapter that learned the new dynamics.

Design discipline (same as session_consolidation.py)
----------------------------------------------------
  * Reuses, does not duplicate -- the multiplicative-rescale math lives in
    :meth:`mt_lnn.sleep_consolidation.SleepWakeConsolidator.synaptic_downscale`;
    adapter identification lives in ``llama_adapter``. This file only selects the
    right tensors and forwards them.
  * The frozen base is protected BY CONSTRUCTION (only ``MTResidualAdapter``
    parameters are collected) AND explicitly (any ``requires_grad=False`` tensor is
    skipped even if nested in an adapter). Normalisation/bias terms are spared by
    default, since SHY targets excitatory synaptic *weights*.
  * Pure multiplicative rescale -- direction-preserving (cosine before/after == 1),
    and exactly invertible if the caller keeps the factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import torch.nn as nn

from .llama_adapter import MTResidualAdapter
from .sleep_consolidation import SleepWakeConsolidator

__all__ = [
    "default_protect",
    "AdapterDownscaleReport",
    "downscale_adapters",
]


def default_protect(name: str) -> bool:
    """Spare normalisation and bias terms -- SHY targets excitatory *weights*.

    ``name`` is a parameter's ``named_parameters`` name *within* an adapter.
    """
    low = name.lower()
    return name.endswith("bias") or "norm" in low


@dataclass(frozen=True)
class AdapterDownscaleReport:
    """Outcome of one adapter SHY pass."""

    n_adapters: int        # MTResidualAdapter modules found in the model
    factor: float          # multiplicative rescale applied
    n_tensors: int         # adapter weight tensors actually rescaled
    pre_norm: float        # global L2 norm of the rescaled weights before
    post_norm: float       # ...and after (== factor * pre_norm)


def downscale_adapters(
    model: nn.Module,
    *,
    factor: float = 0.9,
    protect: Optional[Callable[[str], bool]] = default_protect,
) -> AdapterDownscaleReport:
    """Multiplicatively downscale every trainable MT-adapter weight in ``model``.

    Walks the model for :class:`mt_lnn.llama_adapter.MTResidualAdapter` modules and
    rescales their excitatory weight tensors in place by ``factor`` (in ``(0, 1]``;
    ``1.0`` is a no-op). The frozen base is never touched: only adapter parameters
    are collected, and any ``requires_grad=False`` tensor is skipped regardless.

    Parameters
    ----------
    factor :
        Multiplicative rescale in ``(0, 1]``. Smaller = stronger downscaling.
    protect :
        Predicate on an adapter parameter's name; when it returns True the
        parameter is left untouched. Defaults to :func:`default_protect`
        (spares normalisation/bias). Pass ``None`` to downscale every adapter
        weight.

    Returns an :class:`AdapterDownscaleReport`. A no-op (all-zero counts, unit
    norms) when the model has no MT adapters, so it is always safe to call.
    """
    n_adapters = 0
    tensors: List[torch.Tensor] = []
    for module in model.modules():
        if not isinstance(module, MTResidualAdapter):
            continue
        n_adapters += 1
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue                    # never touch a frozen weight
            if protect is not None and protect(name):
                continue                    # spare norm/bias
            tensors.append(p)

    if not tensors:
        return AdapterDownscaleReport(
            n_adapters=n_adapters, factor=float(factor),
            n_tensors=0, pre_norm=0.0, post_norm=0.0,
        )

    # Reuse the SHY rescale math; pass an explicit tensor list so the module-level
    # frozen base can never be reached even by accident.
    consolidator = SleepWakeConsolidator(downscale_factor=factor)
    res = consolidator.synaptic_downscale(tensors, factor=factor)
    return AdapterDownscaleReport(
        n_adapters=n_adapters,
        factor=res.factor,
        n_tensors=res.n_tensors,
        pre_norm=res.pre_norm,
        post_norm=res.post_norm,
    )
