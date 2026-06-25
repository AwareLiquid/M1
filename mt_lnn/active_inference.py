"""
mt_lnn/active_inference.py — Expected-Free-Energy goal selection over imagination.

Why this module exists
----------------------
A passive predictor answers "what happens next?". An *agent* answers "which of
my possible goals should I pursue?". Active inference (Friston et al. 2015, 2017;
Da Costa et al. 2020) gives a single scalar for that choice — the **Expected
Free Energy** (EFE) of a policy — and says the agent selects the policy that
*minimises* it. EFE decomposes into two competing drives:

    G(goal)  =  -pragmatic_value(goal)  -  epistemic_value(goal)

* **Pragmatic (extrinsic) value** — how well the agent's *predicted* future
  serves the goal. Goals the world model expects to actually reach score high.
  This is the exploit / goal-seeking term.
* **Epistemic (intrinsic) value** — how much *new information* pursuing the goal
  is expected to yield. Goals far from the present state (novel, uncertain)
  score high. This is the explore / curiosity term.

Minimising ``G`` balances the two: an agent driven only by pragmatic value never
explores; one driven only by epistemic value never commits. EFE is the principled
trade-off, and it is what lets a model pick **autonomous goals** rather than only
react.

MT-LNN already has the imagination substrate:
:class:`mt_lnn.imagination.LatentImagination` rolls the world-model head forward
into an imagined latent trajectory with per-step confidence and novelty. This
module scores a *set of candidate goal latents* against that imagined future and
returns the EFE-minimising choice — closing predict -> imagine -> **choose**.

What it does (and does NOT do)
------------------------------
* It is a pure **inference-time scorer**: given the current hidden state and a
  bank of candidate goal latents, it imagines one trajectory, then computes
  per-goal pragmatic / epistemic value and the EFE, and selects the argmin.
* It adds **zero trainable parameters** and does **not** touch the forward pass.
  It is duck-typed on a :class:`LatentImagination`-like object (anything with an
  ``imagine`` method and a ``head`` exposing ``online_proj``), so it composes
  with the real imagination actuator but imports neither ``model.py`` nor the
  backbone. Default-off: nothing in MT-LNN constructs it.

Pinned in tests/test_active_inference.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

__all__ = ["EFEPlan", "ActiveInferencePlanner"]


@dataclass
class EFEPlan:
    """Outcome of :meth:`ActiveInferencePlanner.score_goals`.

    All per-goal tensors are shaped ``(B, G)`` for ``B`` batch rows and ``G``
    candidate goals.

    Attributes
    ----------
    efe : (B, G)
        Expected Free Energy of each goal — ``-(w_p · pragmatic + w_e ·
        epistemic)``. The agent prefers the **smallest** EFE.
    pragmatic : (B, G)
        Pragmatic / extrinsic value ∈ [0, 1]: confidence-weighted alignment
        between the imagined trajectory and the goal (exploit term).
    epistemic : (B, G)
        Epistemic / intrinsic value ∈ [0, 1]: novelty of the goal relative to
        the present latent (explore / information-gain term).
    selected : (B,)
        Index of the EFE-minimising goal for each batch row (``argmin`` of
        ``efe``).
    goals : (G, P)
        The L2-normalised candidate goal latents that were scored.
    """

    efe: torch.Tensor
    pragmatic: torch.Tensor
    epistemic: torch.Tensor
    selected: torch.Tensor
    goals: torch.Tensor

    def chosen_goals(self) -> torch.Tensor:
        """The selected goal latent per batch row — ``(B, P)``."""
        return self.goals[self.selected]


class ActiveInferencePlanner:
    """Select autonomous goals by minimising Expected Free Energy.

    .. note::
        **研究实验组件 / Research prototype — NOT wired into the main path.**
        Exported for standalone use only; never instantiated by ``MTLNNModel``,
        the training loop, or the serving path (``generate_with_thinking`` /
        ``DualSpeedSentry``). Kept as a code prototype for offline active-inference
        experiments. Do not assume it runs as part of the model.

    Parameters
    ----------
    imagination : object
        A :class:`mt_lnn.imagination.LatentImagination`-like actuator exposing an
        ``imagine(hidden, horizon=None) -> ImaginedTrajectory`` method and a
        ``head`` with ``online_proj`` (to encode the present hidden state into
        the head's latent space). Duck-typed: anything matching works.
    pragmatic_weight : float
        Weight ``w_p >= 0`` on the exploit (goal-reaching) term. Default 1.0.
    info_gain_weight : float
        Weight ``w_e >= 0`` on the explore (information-gain) term. Default 1.0.
        Raising it makes the agent prefer novel, uncertain goals; lowering it
        makes it prefer goals it already expects to reach.

    Notes
    -----
    Adds **zero** trainable parameters and runs under ``no_grad`` (it reuses the
    imagination actuator, which itself reuses the trained head).
    """

    def __init__(
        self,
        imagination,
        *,
        pragmatic_weight: float = 1.0,
        info_gain_weight: float = 1.0,
    ):
        if not hasattr(imagination, "imagine"):
            raise TypeError(
                "imagination must expose an 'imagine' method (a "
                "LatentImagination-like object)."
            )
        if not hasattr(imagination, "head") or not hasattr(imagination.head, "online_proj"):
            raise TypeError(
                "imagination.head must expose 'online_proj' to encode the "
                "present hidden state."
            )
        if pragmatic_weight < 0.0:
            raise ValueError(f"pragmatic_weight must be >= 0, got {pragmatic_weight}")
        if info_gain_weight < 0.0:
            raise ValueError(f"info_gain_weight must be >= 0, got {info_gain_weight}")
        self.imagination = imagination
        self.pragmatic_weight = float(pragmatic_weight)
        self.info_gain_weight = float(info_gain_weight)

    @property
    def n_parameters(self) -> int:
        """Trainable parameters this scorer adds: always 0."""
        return 0

    def _present_latent(self, hidden: torch.Tensor) -> torch.Tensor:
        """Encode the current hidden state into the head's normalised latent."""
        if hidden.dim() == 3:
            hidden = hidden[:, -1, :]                     # (B, d_model)
        elif hidden.dim() != 2:
            raise ValueError(
                f"hidden must be (B, d_model) or (B, T, d_model), got {tuple(hidden.shape)}"
            )
        z0 = self.imagination.head.online_proj(hidden)    # (B, P) unnormalised
        return F.normalize(z0, dim=-1, eps=1e-8)          # (B, P) normalised present

    @torch.no_grad()
    def score_goals(
        self,
        hidden: torch.Tensor,
        goals: torch.Tensor,
        *,
        horizon: Optional[int] = None,
    ) -> EFEPlan:
        """Score candidate goal latents by EFE and select the minimiser.

        Parameters
        ----------
        hidden : (B, d_model) or (B, T, d_model)
            The current hidden state (e.g. a backbone's ``final_norm`` output).
        goals : (G, P) or (P,)
            Candidate goal latents in the head's projection space. They are
            L2-normalised internally; magnitude is ignored (direction is the
            goal).
        horizon : int, optional
            Override the imagination horizon for this call.

        Returns
        -------
        EFEPlan
        """
        if goals.dim() == 1:
            goals = goals.unsqueeze(0)                    # (1, P)
        if goals.dim() != 2:
            raise ValueError(f"goals must be (G, P) or (P,), got {tuple(goals.shape)}")

        traj = self.imagination.imagine(hidden, horizon=horizon)
        z = traj.latents                                  # (B, H, P) normalised
        conf = traj.confidence                            # (B, H)
        P = z.shape[-1]
        if goals.shape[-1] != P:
            raise ValueError(
                f"goal width {goals.shape[-1]} != projection dim {P}"
            )
        goals_n = F.normalize(goals.to(z.dtype), dim=-1, eps=1e-8)   # (G, P)
        z0 = self._present_latent(hidden)                 # (B, P) normalised present

        # Pragmatic (exploit): confidence-weighted alignment of the imagined
        # trajectory with each goal, mapped to [0, 1]. A future the model expects
        # to actually traverse toward the goal scores high.
        cos_tg = torch.einsum("bhp,gp->bhg", z, goals_n)  # (B, H, G) ∈ [-1, 1]
        w = conf.unsqueeze(-1)                            # (B, H, 1)
        denom = w.sum(dim=1).clamp_min(1e-8)              # (B, 1)
        weighted_cos = (w * cos_tg).sum(dim=1) / denom    # (B, G) ∈ [-1, 1]
        pragmatic = ((1.0 + weighted_cos) * 0.5).clamp(0.0, 1.0)     # (B, G)

        # Epistemic (explore): novelty of the goal relative to the present state.
        # Goals far from "now" promise more new information if pursued.
        cos_g0 = torch.einsum("bp,gp->bg", z0, goals_n)   # (B, G) ∈ [-1, 1]
        epistemic = ((1.0 - cos_g0) * 0.5).clamp(0.0, 1.0)          # (B, G)

        # Expected Free Energy: minimise -> maximise (pragmatic + epistemic).
        efe = -(self.pragmatic_weight * pragmatic + self.info_gain_weight * epistemic)
        selected = torch.argmin(efe, dim=-1)              # (B,)

        return EFEPlan(
            efe=efe,
            pragmatic=pragmatic,
            epistemic=epistemic,
            selected=selected,
            goals=goals_n,
        )

    def select_goal(
        self,
        hidden: torch.Tensor,
        goals: torch.Tensor,
        *,
        horizon: Optional[int] = None,
    ) -> torch.Tensor:
        """Convenience: just the EFE-minimising goal index per batch row — ``(B,)``."""
        return self.score_goals(hidden, goals, horizon=horizon).selected

    # An instance is directly callable -> the full plan.
    __call__ = score_goals

    def __repr__(self) -> str:
        return (
            f"ActiveInferencePlanner(pragmatic_weight={self.pragmatic_weight}, "
            f"info_gain_weight={self.info_gain_weight})"
        )
