"""
spatial_reasoning.py — 空间思考: spatial perception + self-thinking deliberation.

This module is the *convergence* of two capabilities that already ship
separately:

  * ``spatial.py``   — grid-cell / coordinate frontends that turn a spatial
                       scene into ``(B, N, d_model)`` tokens (perception);
  * ``thinking.py`` / ``deliberation.py`` — the entropy-based router that
                       decides, per token, whether the model is confident,
                       should self-critique, or needs an external fact
                       (deliberation).

:class:`SpatialReasoner` wires a :class:`SpatialCoordEncoder` to an MT-LNN
backbone and runs the deliberation router over the backbone's *per-spatial-
position* predictions. The output is a :class:`ThinkingTrace` whose steps are
spatial positions rather than decode steps — i.e. **a map of where in the
scene the model is uncertain** and where it had to reconsider. That is the
concrete meaning of "空间思考" here: uncertainty-aware reasoning over space.

Coupling / classification
-------------------------
* Imports only public APIs: ``spatial`` (encoder), ``deliberation`` (router),
  ``thinking`` (trace data structures + self-consistency vote). It touches the
  MT-LNN backbone solely through the documented ``forward(inputs_embeds=...)``
  → ``out["logits"]`` contract — **no new coupling** to model internals.
* It is an ``nn.Module`` so the spatial encoder trains jointly with the model
  when you want it to; ``reason()`` itself runs under ``no_grad``.
* Works alongside text: pass ``text_embeds`` and the spatial tokens are fused
  ahead of them via :func:`multimodal.fuse`, so the same backbone reasons over
  a mixed (spatial + language) sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spatial import SpatialCoordEncoder
from .multimodal import fuse
from .deliberation import DeliberationRouter, Route, RouterThresholds
from .thinking import StepTrace, ThinkingTrace, self_consistency_vote
from .causality import CausalConsistencyChecker
from .causal_steering import CausalActivationSteerer
from .spatial_memory import SpatialMemory

__all__ = ["SpatialThinkingResult", "SpatialReasoner"]


@dataclass
class SpatialThinkingResult:
    """Outcome of :meth:`SpatialReasoner.reason`.

    Attributes
    ----------
    logits        : ``(B, N, V)`` backbone logits over the spatial tokens
                    (and any fused text tokens).
    trace         : :class:`ThinkingTrace` with one step per *spatial* token —
                    its ``route`` flags whether the model was confident
                    (LOCAL), reconsidered (SELF_CRITIQUE) or wanted external
                    grounding (CLOUD) at that position.
    n_spatial     : number of spatial tokens scored (the first ``n_spatial``
                    positions of the sequence).
    memory_familiarity : optional ``[float]`` of length ``n_spatial`` — per
                    position, how much the currently perceived token matches what
                    the (L2) :class:`SpatialMemory` recalls at that location
                    (cosine in ``[-1, 1]``). ``None`` when no memory is attached.
                    High values mark places the reasoner has "seen before".
    """

    logits: torch.Tensor
    trace: ThinkingTrace
    n_spatial: int
    memory_familiarity: Optional[List[float]] = None

    def uncertain_positions(self) -> List[int]:
        """Indices of spatial tokens the router did **not** mark LOCAL.

        These are the positions where the model "stopped to think" — the
        spatial analogue of the thinking trace's self-critique steps.
        """
        return [s.index for s in self.trace.steps if s.route != Route.LOCAL.value]


class SpatialReasoner(nn.Module):
    """Perceive a spatial scene, then deliberate over it with the router.

    Parameters
    ----------
    model : an MT-LNN backbone exposing ``forward(inputs_embeds=...)`` and
        returning a dict with a ``"logits"`` entry, plus ``config.d_model``.
    coord_dim : coordinate dimensionality (2 → hexagonal grid-cell code).
    feat_dim : optional per-point feature width (0 = coordinates only).
    encoder : a pre-built :class:`SpatialCoordEncoder`; one is constructed from
        ``coord_dim`` / ``feat_dim`` when omitted.
    router : a :class:`DeliberationRouter`; a default one is built otherwise.
    thresholds : entropy thresholds for the default router.
    checker : optional :class:`CausalConsistencyChecker`. When supplied,
        :meth:`reason` walks the model's *per-position belief trajectory* through
        it and feeds the resulting consistency score to the router as
        ``consistency_signal`` — so a position where the trajectory jumps abruptly
        is flagged for deliberation even when its token entropy looks confident.
        ``None`` (default) keeps the original entropy-only behaviour.
    steerer : optional :class:`CausalActivationSteerer`. Requires ``checker``.
        When supplied, each position's state is tested against the legal causal
        subspace and the off-subspace drift is surfaced as a per-position
        diagnostic in the trace (see :meth:`reason`). ``None`` (default) is a
        no-op. Both are plain attributes (no params) — fully decoupled and
        backward-compatible.
    memory : optional (L2) :class:`SpatialMemory`. When supplied, call
        :meth:`remember` to write the perceived scene into the place-indexed
        store, and :meth:`reason` additionally reports a per-position *memory
        familiarity* (how well the current perception matches what was recalled
        at that location) on the result. Writing is kept an explicit caller
        action — ``reason`` stays read-only and never mutates the map — so the
        reasoner does not silently own long-lived memory state. ``None`` (default)
        is a no-op. Zero params, no model coupling.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        coord_dim: int = 2,
        feat_dim: int = 0,
        encoder: Optional[SpatialCoordEncoder] = None,
        router: Optional[DeliberationRouter] = None,
        thresholds: Optional[RouterThresholds] = None,
        checker: Optional[CausalConsistencyChecker] = None,
        steerer: Optional[CausalActivationSteerer] = None,
        memory: Optional[SpatialMemory] = None,
    ):
        super().__init__()
        self.model = model
        d_model = int(model.config.d_model)
        self.encoder = encoder or SpatialCoordEncoder(
            d_model, coord_dim=coord_dim, feat_dim=feat_dim,
        )
        # The router is policy-only (no params); kept as a plain attribute.
        self.router = router or DeliberationRouter(thresholds=thresholds)
        # Optional causal-trajectory monitor + actuator (both default off).
        if steerer is not None and checker is None:
            raise ValueError("steerer requires a checker (it reads the checker's subspace)")
        self.checker = checker
        self.steerer = steerer
        # Optional L2 place-indexed memory (default off). Registered as a
        # submodule when present so its buffers move with .to(); it carries zero
        # trainable parameters. Assigning None is a no-op attribute.
        self.memory = memory

    # ------------------------------------------------------------- perception
    def perceive(
        self,
        coords: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``(B, N, coord_dim) -> (B, N, d_model)`` spatial tokens."""
        return self.encoder(coords, feats)

    # ------------------------------------------------------------- memory (L2)
    @torch.no_grad()
    def remember(
        self,
        coords: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Write the perceived scene into the attached :class:`SpatialMemory`.

        Perceives ``coords`` (+ optional ``feats``) into ``(B, N, d_model)``
        tokens and stores them keyed by location, so a later :meth:`reason` (or
        :meth:`recall`) over nearby places completes back to this content. This
        is an *explicit* caller action: it is the only path that mutates the map,
        keeping :meth:`reason` read-only. Returns the perceived tokens written.

        Raises if no ``memory`` was attached at construction.
        """
        if self.memory is None:
            raise ValueError("remember() requires a SpatialMemory (pass memory=... to __init__)")
        tokens = self.perceive(coords, feats)                  # (B, N, d)
        self.memory.write(coords, tokens)
        return tokens

    @torch.no_grad()
    def recall(self, coords: torch.Tensor) -> torch.Tensor:
        """``(B, N, coord_dim) -> (B, N, d_model)`` content recalled at ``coords``.

        Pattern-completion read of the attached memory. Raises if none attached.
        """
        if self.memory is None:
            raise ValueError("recall() requires a SpatialMemory (pass memory=... to __init__)")
        return self.memory.read(coords)

    # ------------------------------------------------------------- reasoning
    @torch.no_grad()
    def reason(
        self,
        coords: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
        *,
        text_embeds: Optional[torch.Tensor] = None,
        n_critique_samples: int = 3,
        query: str = "",
    ) -> SpatialThinkingResult:
        """Run perception → backbone → per-position deliberation.

        The spatial tokens occupy the first ``N`` sequence positions (fused
        ahead of ``text_embeds`` when provided). We score exactly those ``N``
        positions: for each we ask the router whether the prediction is
        confident; uncertain positions trigger a self-consistency vote, and the
        decision is logged as one :class:`StepTrace`.
        """
        self.model.eval()
        spatial_tokens = self.perceive(coords, feats)          # (B, N, d)
        n_spatial = spatial_tokens.shape[1]

        # --- memory familiarity (optional, read-only) ---------------------
        # How well the current perception matches what the place-indexed memory
        # recalls at each location. This never writes the map (remember() does),
        # so reason() stays side-effect free.
        memory_familiarity: Optional[List[float]] = None
        if self.memory is not None:
            recalled = self.memory.read(coords)                # (B, N, d)
            fam = F.cosine_similarity(spatial_tokens, recalled, dim=-1, eps=1e-8)
            memory_familiarity = [float(fam[:, n].mean()) for n in range(n_spatial)]

        # Fuse spatial tokens ahead of any text tokens (multimodal contract).
        embeds = (fuse(spatial_tokens, text_embeds)
                  if text_embeds is not None else spatial_tokens)

        out = self.model(inputs_embeds=embeds, use_lnn_recurrence=True)
        logits = out["logits"]                                 # (B, N(+T), V)

        # Reset the causal monitor so each scene starts with a clean trajectory.
        if self.checker is not None:
            self.checker.reset()

        trace = ThinkingTrace()
        for n in range(n_spatial):
            pos_logits = logits[:, n, :]                       # (B, V)

            # --- causal-trajectory signal (optional) ----------------------
            # Treat the per-position logits as the observable "belief state"
            # and walk it through the checker; an abrupt jump lowers the
            # consistency score, which the router treats as a reason to think.
            consistency_signal: Optional[float] = None
            steer_note = ""
            mem_note = (f" | mem:familiarity {memory_familiarity[n]:.2f}"
                        if memory_familiarity is not None else "")
            if self.checker is not None:
                consistency_signal = self.checker.update(pos_logits)
                if self.steerer is not None:
                    # Diagnostic only: how far this position drifted OUT of the
                    # legal causal subspace (corrective re-feed of the steered
                    # state belongs to a recurrent decode loop that owns the
                    # state — kept out of this one-shot pass to avoid coupling
                    # to model internals).
                    res = self.steerer.steer(pos_logits, self.checker,
                                             score=consistency_signal)
                    if res.applied:
                        steer_note = (f" | steer:off-subspace drift "
                                      f"{res.correction_norm:.3g} "
                                      f"(k={res.subspace_dim})")

            decision = self.router.decide(
                pos_logits, query=query, evidence_log=[],
                consistency_signal=consistency_signal,
            )
            n_resamples = 0
            sem_h: Optional[float] = None
            revised = False
            if decision.route in (Route.SELF_CRITIQUE, Route.CLOUD):
                # "Stop and think" at this spatial position: re-decide the
                # most-likely class via a self-consistency vote.
                base_id = int(pos_logits.reshape(-1, pos_logits.shape[-1])[-1]
                              .argmax().item())
                chosen, sem_h, n_resamples = self_consistency_vote(
                    pos_logits, n_samples=n_critique_samples,
                )
                revised = chosen != base_id
            trace.steps.append(StepTrace(
                index=n,
                token_id=-1,            # spatial position, not a vocab token
                token_text="",
                entropy=decision.entropy,
                route=decision.route.value,
                reason=decision.reason + steer_note + mem_note,
                n_resamples=n_resamples,
                sem_entropy=sem_h,
                revised=revised,
            ))

        return SpatialThinkingResult(
            logits=logits, trace=trace, n_spatial=n_spatial,
            memory_familiarity=memory_familiarity,
        )
