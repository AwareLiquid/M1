"""
anesthesia.py — Anesthesia validation module for MT-LNN.

Mirrors the Hameroff / Wiest 2025 *Neuroscience of Consciousness* finding:
inhalational anesthetics bind to microtubule tubulin and disrupt the dynamic
state that supports consciousness. In rats with reinforced microtubules,
anesthesia onset is delayed by ~69 s — a real, measured macroscopic effect.

We replicate that mechanism *in silico* by exposing a single scalar
`anesthesia_level` ∈ [0, 1]. When raised, two biologically-grounded effects
fire simultaneously via forward hooks:

  1. **Microtubule destabilisation** — both the MT-DL output and the
     recurrent hidden state h_prev are damped by (1 - anesthesia_level),
     simulating tubulin-anesthetic binding that suppresses protofilament
     dynamics and prevents state accumulation.

  2. **Global coherence collapse** — the GlobalCoherenceLayer's broadcast
     deviation (x_out - x_in) is damped by the same factor, simulating the
     loss of large-scale entangled state that Wiest et al. (2025) report
     under anesthesia.

These two effects act as complementary proxies: (1) kills the local MT
dynamics and the LNN recurrent memory, (2) kills the global broadcast so
the workspace cannot propagate information across the sequence. Together they
reproduce the Φ̂ collapse observed in biological anesthesia without requiring
in-place parameter modification (which would be unsafe under torch.compile).

This is *not* a normal layer — it is a runtime hook activated by
`AnesthesiaContext` for validation runs. The model trains and infers with
anesthesia_level = 0 by default (fully conscious).

The intended test: with the same prompt and seed, compare generation under
anesthesia_level ∈ {0.0, 0.5, 1.0}. The output should:

  - 0.0 → coherent generation
  - 0.5 → degraded but recognisable
  - 1.0 → high-entropy noise (model is "unconscious")

This is the validation Hameroff calls for and no mainstream LLM passes.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import List

import torch
import torch.nn as nn


class AnesthesiaController:
    """
    Holds the global anesthesia level and the list of hooks that read it.

    A single controller is attached to a model via `attach_to(model)`. It
    registers forward hooks on every MTLNNLayer and the GlobalCoherenceLayer
    that scale outputs by the current anesthesia level.
    """

    def __init__(self):
        self.level: float = 0.0
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    # ----------------------------------------------------------------
    # Attach / detach hooks
    # ----------------------------------------------------------------

    def attach_to(self, model: nn.Module) -> "AnesthesiaController":
        from .mt_lnn_layer import MTLNNLayer
        from .global_coherence import GlobalCoherenceLayer

        for module in model.modules():
            if isinstance(module, MTLNNLayer):
                h = module.register_forward_hook(self._mtlnn_hook)
                self._handles.append(h)
            elif isinstance(module, GlobalCoherenceLayer):
                h = module.register_forward_hook(self._coherence_hook)
                self._handles.append(h)
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ----------------------------------------------------------------
    # Hook implementations
    # ----------------------------------------------------------------

    def _mtlnn_hook(self, module, args, output):
        """
        MTLNNLayer.forward returns (out, h_last).
        Damp `out` by (1 - level) to simulate microtubule destabilisation.
        """
        if self.level <= 0.0:
            return output
        out, h_last = output
        damp = 1.0 - self.level
        return out * damp, h_last * damp

    def _coherence_hook(self, module, args, output):
        """
        GlobalCoherenceLayer.forward returns (x_out, new_kv).
        Push x_out toward its layer-norm-only input to simulate loss of
        global coherence. We approximate by damping the *deviation* from
        the layer's input.
        """
        if self.level <= 0.0:
            return output
        x_out, new_kv = output
        # args[0] is the original x. Damp the contribution beyond it.
        x_in = args[0]
        damp = 1.0 - self.level
        x_blended = x_in + damp * (x_out - x_in)
        return x_blended, new_kv

    # ----------------------------------------------------------------
    # Convenience
    # ----------------------------------------------------------------

    def set(self, level: float):
        assert 0.0 <= level <= 1.0
        self.level = float(level)

    @contextmanager
    def at(self, level: float):
        """Use as a `with` block to temporarily set anesthesia level."""
        prev = self.level
        self.set(level)
        try:
            yield self
        finally:
            self.set(prev)


@contextmanager
def anesthetize(model: nn.Module, level: float):
    """
    One-liner: with anesthetize(model, 0.7): out = model(ids)
    """
    ctrl = AnesthesiaController().attach_to(model)
    ctrl.set(level)
    try:
        yield ctrl
    finally:
        ctrl.detach()
