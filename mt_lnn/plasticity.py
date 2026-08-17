"""
plasticity.py — Hebbian co-activation regularizer for MT-LNN.

Biological grounding
--------------------
Hebb's rule (1949): "When an axon of cell A is near enough to excite cell B
and repeatedly or persistently takes part in firing it, some growth process or
metabolic change takes place in one or both cells such that A's efficiency, as
one of the cells firing B, is increased."

Modern formulation: ΔW_ij ∝ x_i (pre-synaptic) × h_j (post-synaptic)
Maximising co-activation patterns reinforces the weights that produce them —
the neurobiological basis of memory consolidation and reduced catastrophic
forgetting.

Why NOT biological STDP in MT-LNN
----------------------------------
ProtofilamentLTC uses continuous-time state updates:
    h_t = exp(-dt/τ) * h_{t-1} + (1 - exp(-dt/τ)) * σ(W_in · x_t)
There are no discrete spikes and no pre/post synaptic timing events (STDP
requires identifying which neuron fired "before" vs "after" in discrete time).
Applying STDP would require artificially discretising the LTC dynamics and
destroying its core continuous-time property.

This module implements the equivalent phenomenon at the LOSS FUNCTION level:
a Hebbian co-activation term L_hebb that is added to the total training loss.
No change to the forward pass.

Algorithm
---------
For each MTLNNLayer in the model:
  1. During forward(), the layer stores:
       _hebb_signal = mean(out_d_model ⊙ x_norm)  — scalar, in graph
     where:
       x_norm      = LayerNorm(x)  — the normalised layer INPUT  (B,T,d_model)
       out_d_model = MTLNNLayer.out_proj(h_flat)  — the layer OUTPUT (B,T,d_model)
                     before the residual addition
     This is the post-synaptic × pre-synaptic co-activation proxy.

  2. HebbianRegularizer.compute_loss(model) collects all _hebb_signal values.

  3. α is modulated by a self-contained consolidation gate:
       α = hebbian_lr × sigmoid(temp × coactivation_mag)  [hebbian_lavi_gate=True]
       α = hebbian_lr                                      [hebbian_lavi_gate=False]

     coactivation_mag = mean(|_hebb_signal|) across blocks (detached), i.e. the
     gate is driven by Hebbian's OWN co-activation magnitude. Strong, consistent
     co-activation → stronger consolidation; weak co-activation → weaker.

     DECOUPLING (v2.2): the gate was previously driven by the rhythm module's
     resonance.last_lavi_mean, which meant `lavi_temperature` only received a
     gradient when use_rhythm was ALSO enabled — a hidden cross-module coupling
     that made Hebbian un-trainable in isolation. The gate is now self-contained,
     so every mechanism (Hebbian / predictive-coding / world-model / GWT / rhythm)
     is independently switchable and ablatable. The parameter keeps the name
     `lavi_temperature` for checkpoint / diagnostics back-compat.

  4. L_hebb = -α × mean(all _hebb_signals)
     Negative sign: minimising total_loss → maximises co-activation → Hebb rule.

  5. L_total = L_lm + [L_wm] + L_hebb

Integration in model.forward()
-------------------------------
The call is analogous to world_model_head:
    if self.hebbian_reg is not None and self.training:
        hebb_loss = self.hebbian_reg.compute_loss(self)
        if hebb_loss is not None:
            loss = loss + hebb_loss
            result["hebbian_loss"] = hebb_loss.detach()

Train.py does not need modification — the model handles it internally.

Coupling invariants
-------------------
- use_hebbian=False (default): no _hebb_signal stored, no loss term, zero impact.
- MTLNNLayer.forward() signature unchanged.
- HebbianRegularizer is an nn.Module (has lavi_temperature as a learnable
  parameter that softens the LAVI gate).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from .model import MTLNNModel


class HebbianRegularizer(nn.Module):
    """
    Hebbian co-activation regularizer.

    Collects per-layer co-activation signals (_hebb_signal) stored by
    MTLNNLayer.forward() and produces a scalar Hebbian loss term.

    Parameters
    ----------
    base_lr : float
        Base Hebbian learning rate α (pre-LAVI gate).
    lavi_gate : bool
        If True, α is multiplied by sigmoid(lavi_temperature × coactivation_mag),
        where coactivation_mag is Hebbian's own detached co-activation magnitude
        (NOT rhythm output — see DECOUPLING note in the module docstring).
        If False, α is constant.
    """

    def __init__(self, base_lr: float = 1e-4, lavi_gate: bool = True):
        super().__init__()
        self.base_lr = base_lr
        self.lavi_gate = lavi_gate
        # Learnable softening temperature for the LAVI gate sigmoid.
        # Initialised to 1.0; allows training to adjust how sharply
        # persistent vs transient mode affects consolidation rate.
        self.lavi_temperature = nn.Parameter(torch.ones(()))

    def compute_loss(self, model: "MTLNNModel") -> Optional[torch.Tensor]:
        """
        Compute -α × mean(co-activation signals) across all blocks.

        Returns None if no signals are available (e.g. inference-only mode,
        or use_hebbian=False in config).

        Parameters
        ----------
        model : MTLNNModel whose blocks may carry _hebb_signal attributes.
        """
        signals = []

        for block in model.blocks:
            sig = getattr(block.lnn, "_hebb_signal", None)
            if sig is not None:
                signals.append(sig)

        if not signals:
            return None

        hebb_stack = torch.stack(signals)
        hebb_mean = hebb_stack.mean()

        # Decoupled consolidation gate (see class docstring).
        # The gate driver is Hebbian's OWN co-activation magnitude, NOT the
        # rhythm module's resonance.last_lavi_mean. Consequence: lavi_temperature
        # is trainable whenever use_hebbian is on, with ZERO dependency on
        # use_rhythm -- every mechanism stays independently switchable/ablatable
        # and there is no hidden "must enable rhythm to train Hebbian" coupling.
        alpha = self.base_lr
        if self.lavi_gate:
            # Detach the driver so the gate is a pure multiplicative modulator
            # (mirrors the original detached-LAVI buffer); gradient still flows
            # to lavi_temperature through the sigmoid as long as co-activation
            # magnitude is non-zero.
            coact = hebb_stack.detach().abs().mean()
            gate = torch.sigmoid(self.lavi_temperature * coact)
            alpha = self.base_lr * gate

        # Negative sign: minimise total loss → maximise co-activation (Hebb rule)
        return -alpha * hebb_mean
