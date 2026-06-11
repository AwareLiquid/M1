"""
causal_decoding.py — apply L3 causal activation steering inside real decoding.

Why this module exists (closing the L3 loop)
---------------------------------------------
``causality.CausalConsistencyChecker`` *detects* abrupt jumps in the LNN
recurrent-state trajectory; ``causal_steering.CausalActivationSteerer`` *projects*
a drifting state back onto the legal causal subspace. Until now those two were
only wired as **read-only diagnostics** (``SpatialReasoner``) or exercised in a
standalone demo — the corrected state never fed back into the model that was
actually generating. This module closes that loop: it turns the detector +
actuator into a ``step_callback`` for :meth:`MTLNNModel.generate`, so a detected
break is corrected *in the live KV/recurrent cache* and the fix conditions the
**next** token. The actuator finally acts on the model it monitors.

How it plugs in (zero model coupling)
-------------------------------------
``MTLNNModel.generate`` exposes a generic ``step_callback(cache, step)`` hook
(it imports nothing causal — the model stays decoupled). A
:class:`CausalDecodeSteerer` *is* such a callback. Each decode step it:

  1. reads each target layer's recurrent state ``cache.layers[i][1]``  (B,P,S,D);
  2. feeds it to a per-layer :class:`CausalConsistencyChecker` (one trajectory
     per layer — layers drift independently);
  3. asks the shared :class:`CausalActivationSteerer` to project it back onto the
     legal subspace **iff** a break is detected (score < ``floor``);
  4. writes the corrected state back: ``cache.layers[i] = (kv, steered, gwtb)``.

On a smooth (healthy) trajectory the score stays high, nothing fires, and
generation is bit-for-bit identical to vanilla decoding — steering is a *gated
correction*, never a constant nudge. This is the safety property: it can only
help on a detected break, and is a no-op otherwise.

Coupling / classification
-------------------------
* Imports only the public ``causality`` + ``causal_steering`` APIs. It never
  imports ``model.py``; the dependency arrow points model → (generic hook), not
  the reverse, so the backbone has no idea this module exists.
* **Zero learnable parameters** — an inference-time actuator, not a layer.
* Fully opt-in: pass ``step_callback=CausalDecodeSteerer(...)`` to ``generate``.
  Omit it and behaviour is unchanged.

Batching note
-------------
The checker/steerer flatten the *whole* ``(B,P,S,D)`` state to one vector, so the
clean case is single-sequence decoding (``B=1``) — the usual server path. With
``B>1`` the per-step states of all rows are pooled into one trajectory and a
single correction is shared; the signal is coarser but still well-defined.

Typical use::

    from mt_lnn.causal_decoding import CausalDecodeSteerer

    steerer = CausalDecodeSteerer(strength=1.0, floor=0.3, layers="all")
    tokens = model.generate(prompt_ids, max_new_tokens=64, step_callback=steerer)
    print(steerer.n_corrections, "states steered back onto the legal subspace")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import torch

from .causality import CausalConsistencyChecker
from .causal_steering import CausalActivationSteerer

__all__ = ["SteerEvent", "CausalDecodeSteerer"]


@dataclass
class SteerEvent:
    """One per-layer steering decision logged during decoding.

    Attributes
    ----------
    step : decode step index (0 = after prompt prefill, then 1, 2, …).
    layer : index of the backbone block whose recurrent state was inspected.
    score : the consistency score that gated the decision (∈ [0, 1]).
    applied : True iff a correction was actually written back to the cache.
    correction_norm : L2 norm of the applied correction (0.0 when not applied).
    subspace_dim : dimensionality k of the legal subspace used.
    """

    step: int
    layer: int
    score: float
    applied: bool
    correction_norm: float
    subspace_dim: int


class CausalDecodeSteerer:
    """A ``generate`` ``step_callback`` that steers the live recurrent cache.

    Parameters
    ----------
    steerer : a pre-built :class:`CausalActivationSteerer`. When omitted one is
        constructed from ``strength`` / ``floor`` / ``adaptive``.
    strength, floor, adaptive : forwarded to the default steerer (ignored if an
        explicit ``steerer`` is given). ``floor`` also seeds each checker's
        ``threshold`` so detector and actuator share one gate.
    layers : which blocks to monitor — ``"all"`` (default), ``"last"``, or an
        explicit list of block indices.
    method, window, energy_keep : per-layer :class:`CausalConsistencyChecker`
        settings. ``method`` defaults to ``"subspace"`` because the steerer needs
        a principal subspace to project onto (the ``"cosine"`` detector exposes
        none, so steering would never fire — choosing subspace here keeps the
        actuator from being silent dead weight).

    Notes
    -----
    One checker is created lazily per monitored layer, so each block's trajectory
    is judged on its own history. Call :meth:`reset` between independent prompts
    (a fresh instance also starts clean).
    """

    def __init__(
        self,
        *,
        steerer: Optional[CausalActivationSteerer] = None,
        strength: float = 1.0,
        floor: float = 0.3,
        adaptive: bool = True,
        layers: Union[str, Sequence[int]] = "all",
        method: str = "subspace",
        window: int = 8,
        energy_keep: float = 0.9,
    ):
        self.steerer = steerer or CausalActivationSteerer(
            strength=strength, floor=floor, adaptive=adaptive,
        )
        # Keep each checker's threshold aligned with the steerer's gate so
        # "break" means the same thing to detector and actuator.
        self._checker_kwargs = dict(
            method=method, window=window,
            threshold=self.steerer.floor, energy_keep=energy_keep,
        )
        self.layers = layers
        self._checkers: Dict[int, CausalConsistencyChecker] = {}
        self.events: List[SteerEvent] = []

    # ------------------------------------------------------------------
    def _checker(self, layer: int) -> CausalConsistencyChecker:
        chk = self._checkers.get(layer)
        if chk is None:
            chk = CausalConsistencyChecker(**self._checker_kwargs)
            self._checkers[layer] = chk
        return chk

    def _target_layers(self, cache) -> List[int]:
        n = len(cache.layers)
        if self.layers == "all":
            return list(range(n))
        if self.layers == "last":
            return [n - 1] if n else []
        return [int(i) for i in self.layers if 0 <= int(i) < n]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def __call__(self, cache, step: int) -> None:
        """Inspect (and possibly correct) each target layer's recurrent state."""
        for i in self._target_layers(cache):
            lc = cache.layers[i]
            if lc is None:
                continue
            h = lc[1]                                   # (B, P, S, D) or None
            if h is None:
                continue
            chk = self._checker(i)
            score = chk.update(h)
            # exclude_last=True: update() just appended h, so the legal subspace
            # must be built from the trajectory BEFORE h — otherwise the suspect
            # break-direction pollutes its own reference and the correction ~0.
            res = self.steerer.steer(h, chk, score=score, exclude_last=True)
            if res.applied:
                # Tuples are immutable — rebuild the LayerCache with the
                # corrected recurrent state so the next forward reads it as
                # h_prev. KV (lc[0]) and GWTB KV (lc[2]) are left untouched.
                cache.layers[i] = (lc[0], res.steered, lc[2])
            self.events.append(SteerEvent(
                step=step, layer=i, score=float(score),
                applied=bool(res.applied),
                correction_norm=float(res.correction_norm),
                subspace_dim=int(res.subspace_dim),
            ))

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear per-layer histories and the event log (e.g. between prompts)."""
        for chk in self._checkers.values():
            chk.reset()
        self.events.clear()

    @property
    def n_corrections(self) -> int:
        """How many states were actually steered back this run."""
        return sum(1 for e in self.events if e.applied)

    @property
    def total_correction_norm(self) -> float:
        """Summed L2 norm of all applied corrections (a crude 'how much work')."""
        return float(sum(e.correction_norm for e in self.events if e.applied))

    def __repr__(self) -> str:
        return (
            f"CausalDecodeSteerer(layers={self.layers!r}, "
            f"floor={self.steerer.floor}, strength={self.steerer.strength}, "
            f"corrections={self.n_corrections})"
        )
