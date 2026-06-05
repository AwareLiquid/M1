"""
deliberation.py — Layer 2 router for AwareLiquid.

Replaces the v1 single-threshold entropy gate with a 3-way decision:

    low entropy                        →  local direct decode
    mid entropy (uncertain reasoning)  →  local self-critique (N-sample)
    high entropy + fact gap            →  cloud oracle inject

The split between "reasoning difficulty" and "missing fact" matters: a
hard reasoning step does not benefit from a Wikipedia round-trip, and a
factual gap is not solved by sampling more from the same model. Gemini's
single thinking-budget knob conflates the two; we don't.

This module is intentionally cheap and dependency-free. Real cloud API
adapters live in ``cloud_client.py`` and are pluggable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Sequence

import torch
import torch.nn.functional as F


class Route(str, Enum):
    """Possible routing decisions for a single decode step."""

    LOCAL = "local"
    SELF_CRITIQUE = "self_critique"
    CLOUD = "cloud"


@dataclass
class RouterThresholds:
    """Tunable decision thresholds. Defaults sit between v1 hard threshold."""

    low: float = 3.0    # below this → trust local decode
    high: float = 5.0   # above this → consider cloud (if fact gap)
    # Phase B (2026-06-06): causal consistency floor.
    # When CausalConsistencyChecker reports score < consistency_floor,
    # the router forces SELF_CRITIQUE regardless of token entropy.
    consistency_floor: float = 0.3


@dataclass
class RouteDecision:
    """Result of a single ``DeliberationRouter.decide()`` call."""

    route: Route
    reason: str
    entropy: float
    semantic_entropy: Optional[float] = None
    fact_gap: Optional[bool] = None
    # Phase B (2026-06-06): causal consistency score from CausalConsistencyChecker.
    # Populated when consistency_signal is passed to decide(). None otherwise.
    causal_consistency: Optional[float] = None


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------

def token_entropy(logits: torch.Tensor) -> float:
    """Shannon entropy of a single next-token logit vector (B=1 friendly)."""
    if logits.dim() > 1:
        logits = logits.reshape(-1, logits.shape[-1])[-1]
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return float(-(probs * log_probs).sum().item())


def semantic_entropy(samples: Sequence[Sequence[int]]) -> float:
    """Cluster-based semantic entropy over N candidate continuations.

    We group identical token-prefix tuples into equivalence classes and
    return Shannon entropy of the class distribution. The intent matches
    Kuhn et al. 2023's semantic-entropy intuition without paraphrase
    grouping (which would need NLI). For short prefixes from a fixed
    sampler this is a reasonable proxy for "do the samples agree?".

    Returns 0.0 for ≤1 sample.
    """
    if len(samples) <= 1:
        return 0.0
    buckets: dict = {}
    for seq in samples:
        key = tuple(int(x) for x in seq)
        buckets[key] = buckets.get(key, 0) + 1
    total = sum(buckets.values())
    h = 0.0
    for count in buckets.values():
        p = count / total
        if p > 0:
            h -= p * math.log(p)
    return h


# ---------------------------------------------------------------------------
# Fact-gap detection
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set:
    out: set = set()
    for raw in text.split():
        tok = raw.strip(".,;:!?\"'()[]{}").lower()
        if len(tok) > 3:
            out.add(tok)
    return out


def lexical_fact_gap(
    query: str,
    evidence_log: Sequence[dict],
    *,
    min_overlap: float = 0.2,
) -> bool:
    """True if no prior evidence row covers the query lexically.

    Cheap stand-in for an embedding retriever. ``evidence_log`` rows are
    expected to carry a ``query`` field (written by
    ``capsule.add_evidence``). Once Layer 4 lands a real embedding
    retriever, swap this function in ``DeliberationRouter``.
    """
    q_tokens = _tokenize(query)
    if not q_tokens:
        return True
    for row in evidence_log:
        prior = _tokenize(str(row.get("query", "")))
        if not prior:
            continue
        overlap = len(q_tokens & prior) / max(1, len(q_tokens))
        if overlap >= min_overlap:
            return False
    return True


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class DeliberationRouter:
    """Three-way router. Stateless apart from the threshold config.

    Parameters
    ----------
    thresholds : entropy bounds for low / high regions.
    sampler : optional callable returning N candidate continuations as
        token-id lists for the same prefix. Used when entropy is in the
        mid range. If omitted, mid-entropy steps fall through to LOCAL.
    fact_gap_fn : function ``(query, evidence_log) -> bool``. Default is
        ``lexical_fact_gap``.
    """

    def __init__(
        self,
        thresholds: Optional[RouterThresholds] = None,
        sampler: Optional[Callable[[], List[List[int]]]] = None,
        fact_gap_fn: Callable[[str, Sequence[dict]], bool] = lexical_fact_gap,
    ):
        self.thresholds = thresholds or RouterThresholds()
        self.sampler = sampler
        self.fact_gap_fn = fact_gap_fn

    def decide(
        self,
        logits: torch.Tensor,
        *,
        query: str,
        evidence_log: Sequence[dict],
        consistency_signal: Optional[float] = None,
    ) -> RouteDecision:
        """Decide a route given current logits and session state.

        Parameters
        ----------
        logits : next-token logit vector (or batch of logits — last row used).
        query : the current user query (for fact-gap detection).
        evidence_log : prior evidence rows from capsule.
        consistency_signal : optional CausalConsistencyChecker score ∈ [0, 1].
            When provided and below RouterThresholds.consistency_floor, the
            router forces SELF_CRITIQUE regardless of token entropy.
            Pass None (default) to skip causal consistency checking entirely.
        """
        h = token_entropy(logits)
        t = self.thresholds

        # Phase B: causal consistency pre-check.
        # A broken trajectory (sudden h_prev jump) overrides entropy routing —
        # the model should re-examine its reasoning before emitting the next
        # token, even if entropy looks low (confidently wrong).
        if consistency_signal is not None and consistency_signal < t.consistency_floor:
            return RouteDecision(
                route=Route.SELF_CRITIQUE,
                reason="causal_break",
                entropy=h,
                causal_consistency=consistency_signal,
            )

        if h < t.low:
            return RouteDecision(route=Route.LOCAL, reason="low_entropy", entropy=h)

        if h < t.high:
            sem_h: Optional[float] = None
            if self.sampler is not None:
                samples = self.sampler()
                sem_h = semantic_entropy(samples)
                # Tight cluster → still trust local (decode resolves).
                # Diverging samples → kick to cloud if we also lack evidence.
                if sem_h < math.log(2):
                    return RouteDecision(
                        route=Route.SELF_CRITIQUE,
                        reason="mid_entropy_convergent_samples",
                        entropy=h,
                        semantic_entropy=sem_h,
                    )
                gap = self.fact_gap_fn(query, evidence_log)
                if gap:
                    return RouteDecision(
                        route=Route.CLOUD,
                        reason="mid_entropy_divergent_samples_with_gap",
                        entropy=h,
                        semantic_entropy=sem_h,
                        fact_gap=True,
                    )
                return RouteDecision(
                    route=Route.SELF_CRITIQUE,
                    reason="mid_entropy_divergent_samples_no_gap",
                    entropy=h,
                    semantic_entropy=sem_h,
                    fact_gap=False,
                )
            return RouteDecision(
                route=Route.SELF_CRITIQUE,
                reason="mid_entropy_no_sampler",
                entropy=h,
            )

        # High entropy.
        gap = self.fact_gap_fn(query, evidence_log)
        if gap:
            return RouteDecision(
                route=Route.CLOUD,
                reason="high_entropy_with_fact_gap",
                entropy=h,
                fact_gap=True,
            )
        return RouteDecision(
            route=Route.SELF_CRITIQUE,
            reason="high_entropy_evidence_already_present",
            entropy=h,
            fact_gap=False,
        )
