"""
imagination.py — latent multi-step "mental simulation" on top of the world model.

Biological grounding
--------------------
The predictive-coding / world-model view (Friston; Ha & Schmidhuber 2018,
"World Models"; Hafner et al. "Dreamer") holds that an agent does not only
predict the *immediate* next state — it can roll its learned forward model
forward in imagination, simulating how a trajectory unfolds over several steps
*before* committing to an action. This is the "in your head, picture how the
scene plays out" capability.

``mt_lnn.world_model.PredictiveStateHead`` already learns a **one-step** latent
transition (BYOL / V-JEPA style: it maps the current hidden state's latent to
the *next* position's latent, trained so the prediction matches the EMA target
encoding of the true next state). :class:`LatentImagination` is the natural
deep extension: it iterates that *same trained* one-step map autoregressively in
the head's normalised latent space to produce an **imagined trajectory** of
length ``horizon``, together with a per-step confidence that decays as the
imagined future drifts further from the present (we trust the near future more
than the far future).

Why this is a real mechanism, not a wrapper
-------------------------------------------
* It reuses the *trained* predictor and projector of the world model. A head
  that has learned a genuine transition produces a multi-step rollout that
  tracks the true k-step future markedly better than a static "assume nothing
  changes" baseline — that gap is what ``tests/test_imagination.py`` pins.
* It adds **zero trainable parameters**: it is a pure inference-time actuator
  over the existing head (``n_parameters == 0``).

Coupling / classification
-------------------------
* Imports only ``torch`` + ``dataclasses``. It NEVER imports ``model.py``. It is
  duck-typed on the head: anything exposing ``online_proj`` (d_model→P),
  ``predictor`` (P→P), ``proj_dim`` and a scalar ``last_pred_error`` works —
  so it composes with the real :class:`PredictiveStateHead` but does not depend
  on the backbone. The demo plugs it onto a live ``MTLNNModel`` final hidden
  state without any change to the model.
* Default-off: nothing in the model instantiates this; you opt in explicitly.

Returned object
---------------
:class:`ImaginedTrajectory` carries the normalised imagined latents
(``z_1 … z_H``), a ``confidence`` ∈ [0, 1] per step, and a ``novelty`` ∈ [0, 1]
per step (how much each imagined step departs from the previous one). These are
diagnostics / gating signals a caller can act on (e.g. abort a plan once
confidence falls below a floor) — they are not fed back into the backbone here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

__all__ = ["ImaginedTrajectory", "LatentImagination"]


@dataclass
class ImaginedTrajectory:
    """Outcome of :meth:`LatentImagination.imagine`.

    Attributes
    ----------
    latents : (B, H, P)
        The imagined next-state latents ``z_1 … z_H``, each L2-normalised
        (unit vectors in the world model's projection space). ``z_t`` is the
        model's "picture" of the latent ``t`` steps into the future.
    confidence : (B, H)
        Per-step confidence ∈ [0, 1]. Starts from the head's current trust
        ``1 - last_pred_error`` and decays with the horizon (``trust_decay``)
        and with imagined novelty — the far, fast-changing future is trusted
        less than the near, smooth future.
    novelty : (B, H)
        Per-step novelty ``(1 - cos(z_t, z_{t-1})) / 2`` ∈ [0, 1]: how much each
        imagined step departs from the one before it (``z_0`` for the first
        step). High novelty = the imagined scene is changing fast.
    horizon : int
        Number of imagined steps ``H``.
    """

    latents: torch.Tensor
    confidence: torch.Tensor
    novelty: torch.Tensor
    horizon: int

    @property
    def endpoint(self) -> torch.Tensor:
        """The final imagined latent ``z_H`` — (B, P)."""
        return self.latents[:, -1, :]

    def confidence_weighted_drift(self) -> torch.Tensor:
        """Confidence-weighted cumulative drift of the imagined trajectory.

        For each step ``t`` the displacement from the present is
        ``(1 - cos(z_t, z_0)) / 2`` where ``z_0`` is the latent of the current
        state; the per-step values are averaged weighted by ``confidence``.
        Returns a scalar per batch element — a single "how far do I expect the
        scene to move, discounted by how much I trust the rollout" summary.
        """
        z0 = self._z0
        if z0 is None:                                  # pragma: no cover - guard
            raise RuntimeError("z0 not recorded on this trajectory")
        cos = (self.latents * z0.unsqueeze(1)).sum(dim=-1)      # (B, H)
        disp = ((1.0 - cos) * 0.5).clamp(0.0, 1.0)             # (B, H)
        w = self.confidence
        denom = w.sum(dim=-1).clamp_min(1e-8)
        return (w * disp).sum(dim=-1) / denom                  # (B,)

    # z_0 (present latent) is stashed by LatentImagination for the drift summary.
    _z0: Optional[torch.Tensor] = None


class LatentImagination:
    """Autoregressive latent rollout of a one-step world-model head.

    Parameters
    ----------
    head : object
        A world-model head exposing ``online_proj`` (Linear d_model→P),
        ``predictor`` (P→P), ``proj_dim`` (int) and a scalar ``last_pred_error``
        buffer. :class:`mt_lnn.world_model.PredictiveStateHead` satisfies this.
    horizon : int
        Number of steps to imagine (``H``). Must be ≥ 1.
    trust_decay : float
        Geometric per-step discount applied to confidence (∈ (0, 1]). 1.0 =
        no horizon discount. Default 0.85.
    novelty_penalty : float
        How strongly per-step novelty further reduces confidence (∈ [0, 1]).
        ``conf_t *= (1 - novelty_penalty · novelty_t)``. Default 0.5.
    renorm : bool
        If True (default), each predicted latent fed back into the predictor is
        rescaled to the magnitude of the previous latent. The world-model
        predictor is trained on *unnormalised* projector outputs, so holding the
        magnitude steady keeps the iterated input in-distribution and prevents
        the rollout from collapsing toward zero or exploding.

    Notes
    -----
    Adds **zero** trainable parameters (``n_parameters == 0``); it is a pure
    inference-time actuator over the existing head. Runs under ``no_grad``.
    """

    def __init__(
        self,
        head,
        *,
        horizon: int = 4,
        trust_decay: float = 0.85,
        novelty_penalty: float = 0.5,
        renorm: bool = True,
    ):
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if not (0.0 < trust_decay <= 1.0):
            raise ValueError(f"trust_decay must be in (0, 1], got {trust_decay}")
        if not (0.0 <= novelty_penalty <= 1.0):
            raise ValueError(f"novelty_penalty must be in [0, 1], got {novelty_penalty}")
        for attr in ("online_proj", "predictor", "proj_dim"):
            if not hasattr(head, attr):
                raise TypeError(
                    f"head is missing required attribute '{attr}'; expected a "
                    f"PredictiveStateHead-like object."
                )
        self.head = head
        self.horizon = int(horizon)
        self.trust_decay = float(trust_decay)
        self.novelty_penalty = float(novelty_penalty)
        self.renorm = bool(renorm)

    @property
    def n_parameters(self) -> int:
        """Trainable parameters this actuator adds: always 0 (reuses the head)."""
        return 0

    def _present_latent(self, hidden: torch.Tensor) -> torch.Tensor:
        """Encode the current hidden state into the head's latent space.

        ``hidden`` may be (B, d_model) or (B, T, d_model); the *last* position
        is used as "now".
        """
        if hidden.dim() == 3:
            hidden = hidden[:, -1, :]                    # (B, d_model)
        elif hidden.dim() != 2:
            raise ValueError(
                f"hidden must be (B, d_model) or (B, T, d_model), got {tuple(hidden.shape)}"
            )
        return self.head.online_proj(hidden)             # (B, P), unnormalised

    @torch.no_grad()
    def imagine(self, hidden: torch.Tensor, horizon: Optional[int] = None) -> ImaginedTrajectory:
        """Roll the head's one-step map forward ``horizon`` steps in latent space.

        Parameters
        ----------
        hidden : (B, d_model) or (B, T, d_model)
            The current hidden state (e.g. a backbone's ``final_norm`` output).
        horizon : int, optional
            Override the configured horizon for this call.

        Returns
        -------
        ImaginedTrajectory
        """
        H = self.horizon if horizon is None else int(horizon)
        if H < 1:
            raise ValueError(f"horizon must be >= 1, got {H}")

        cur = self._present_latent(hidden)               # (B, P) unnormalised present latent
        z_prev = F.normalize(cur, dim=-1, eps=1e-8)      # (B, P) normalised present (z_0)
        z0 = z_prev

        # Trust scalar from the head's most recent one-step surprise (∈ [0,1]):
        # a head that predicts well right now (low error) earns high base trust.
        err = float(getattr(self.head, "last_pred_error", 0.0))
        if not (err == err):                             # NaN guard
            err = 1.0
        base_trust = max(0.0, min(1.0, 1.0 - err))

        latents = []
        confs = []
        novs = []
        for t in range(1, H + 1):
            raw = self.head.predictor(cur)               # (B, P) predicted next latent
            z_t = F.normalize(raw, dim=-1, eps=1e-8)     # (B, P) normalised

            # Novelty: how far this imagined step departs from the previous one.
            cos_step = (z_t * z_prev).sum(dim=-1).clamp(-1.0, 1.0)   # (B,)
            novelty = ((1.0 - cos_step) * 0.5).clamp(0.0, 1.0)       # (B,)

            # Confidence: base trust, geometric horizon discount, novelty penalty.
            decay = self.trust_decay ** (t - 1)
            conf = base_trust * decay * (1.0 - self.novelty_penalty * novelty)
            conf = conf.clamp(0.0, 1.0)

            latents.append(z_t)
            confs.append(conf)
            novs.append(novelty)

            # Prepare next iteration's predictor input. Feed the *unnormalised*
            # prediction (the space the predictor was trained on); optionally
            # rescale to the previous magnitude to stay in-distribution.
            if self.renorm:
                scale = cur.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                cur = F.normalize(raw, dim=-1, eps=1e-8) * scale
            else:
                cur = raw
            z_prev = z_t

        traj = ImaginedTrajectory(
            latents=torch.stack(latents, dim=1),         # (B, H, P)
            confidence=torch.stack(confs, dim=1),        # (B, H)
            novelty=torch.stack(novs, dim=1),            # (B, H)
            horizon=H,
        )
        traj._z0 = z0
        return traj

    # Convenience: an instance is directly callable.
    __call__ = imagine
