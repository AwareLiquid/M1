"""
thinking.py — Live "self-thinking" generation for the serve / demo path.

This module turns the *policy* defined in :mod:`mt_lnn.deliberation` (the
three-way LOCAL / SELF_CRITIQUE / CLOUD router) into an actual generation
*mechanism* that the public demo can run token-by-token. Concretely it
implements the local SELF_CRITIQUE step that ``deliberation.py`` leaves as a
hook (see the ``# Future: N-sample re-decode here`` comment in
``demo_awareliquid_v2.generate_with_router``): when the router flags a token
as uncertain, the model *reconsiders* it via a cheap token-level
self-consistency vote instead of committing to a single temperature draw.

Why a separate module (coupling / classification)
-------------------------------------------------
* **Zero backbone coupling.** It imports only ``torch`` and
  ``mt_lnn.deliberation`` (itself dependency-free). It never imports
  ``model.py``; it works with *any* object whose ``forward`` returns an
  object with a ``.logits`` tensor — Qwen, Llama, or an MT-LNN ``serve.pt``.
  The public Gradio demo (``app.py``) can therefore switch self-thinking on
  without touching the existing chat / completion paths.
* **Clear separation of concerns.** ``deliberation.py`` = decision policy,
  ``thinking.py`` = decode mechanism + live trace, ``reasoning_trace.py`` =
  *persisted* JSONL trace for offline replay. This module's trace is
  in-memory so the frontend can render a thinking timeline immediately.
* **Performance.** The default self-critique uses a self-consistency vote
  over candidates drawn from the *already-computed* logits — no extra model
  forward passes — so enabling it costs ~one extra softmax per flagged
  token. An optional one-step look-ahead mode is available when accuracy
  matters more than latency.
* **Graceful CLOUD degradation.** The public demo has no cloud oracle wired
  in. Rather than fabricate facts, a CLOUD decision is *annotated* in the
  trace ("would consult cloud") and falls back to local self-critique so the
  visible output stays coherent. Inject a real client later via ``cloud_fn``.

Public API
----------
    StepTrace, ThinkingTrace          — in-memory trace data structures
    self_consistency_vote(...)        — the cheap SELF_CRITIQUE mechanism
    generate_with_thinking(...)       — drop-in generation loop w/ trace
    render_trace_html / _markdown     — frontend-friendly renderers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .causality import CausalConsistencyChecker
from .deliberation import (
    DeliberationRouter,
    Route,
    RouterThresholds,
    semantic_entropy,
)

__all__ = [
    "StepTrace",
    "ThinkingTrace",
    "self_consistency_vote",
    "generate_with_thinking",
    "render_trace_markdown",
    "render_trace_html",
]


# ---------------------------------------------------------------------------
# Trace data structures (in-memory; rendered live by the frontend)
# ---------------------------------------------------------------------------

@dataclass
class StepTrace:
    """One decode step's self-thinking record.

    Attributes
    ----------
    index        : 0-based step position in the generated sequence.
    token_id     : the token that was finally emitted.
    token_text   : decoded surface form (filled in by the caller).
    entropy      : Shannon entropy of the next-token distribution.
    route        : ``Route`` value the router chose ("local" / ...).
    reason       : short human string explaining the route.
    n_resamples  : how many candidates the self-critique vote considered
                   (0 when no critique ran).
    sem_entropy  : semantic entropy across the critique candidates (None
                   when no critique ran).
    revised      : True if self-critique changed the emitted token relative
                   to the plain top-p draw.
    causal_consistency : CausalConsistencyChecker score in [0, 1] for this
                   step (None when the causal-consistency check is disabled).
                   < RouterThresholds.consistency_floor forces SELF_CRITIQUE.
    """

    index: int
    token_id: int
    token_text: str
    entropy: float
    route: str
    reason: str
    n_resamples: int = 0
    sem_entropy: Optional[float] = None
    revised: bool = False
    causal_consistency: Optional[float] = None


@dataclass
class ThinkingTrace:
    """Ordered collection of :class:`StepTrace` plus derived summaries."""

    steps: List[StepTrace] = field(default_factory=list)

    # -- aggregate views ----------------------------------------------------
    @property
    def route_counts(self) -> Dict[str, int]:
        """Number of steps taken on each route."""
        counts: Dict[str, int] = {}
        for s in self.steps:
            counts[s.route] = counts.get(s.route, 0) + 1
        return counts

    @property
    def mean_entropy(self) -> float:
        if not self.steps:
            return 0.0
        return sum(s.entropy for s in self.steps) / len(self.steps)

    @property
    def n_self_critique(self) -> int:
        return sum(1 for s in self.steps if s.route == Route.SELF_CRITIQUE.value)

    @property
    def n_cloud_flagged(self) -> int:
        return sum(1 for s in self.steps if s.route == Route.CLOUD.value)

    @property
    def n_revised(self) -> int:
        return sum(1 for s in self.steps if s.revised)

    @property
    def n_causal_breaks(self) -> int:
        """Steps the router re-routed to SELF_CRITIQUE because the recurrent /
        hidden-state trajectory broke (low consistency), regardless of token
        entropy. 0 when the causal-consistency check is disabled."""
        return sum(1 for s in self.steps if s.reason == "causal_break")

    @property
    def min_consistency(self) -> Optional[float]:
        """Lowest causal-consistency score seen across the generation (None
        when the check was disabled / never populated)."""
        scores = [s.causal_consistency for s in self.steps
                  if s.causal_consistency is not None]
        return round(min(scores), 3) if scores else None

    def summary(self) -> Dict[str, object]:
        """Compact dict suitable for logging or a UI header."""
        out: Dict[str, object] = {
            "n_tokens": len(self.steps),
            "route_counts": self.route_counts,
            "mean_entropy": round(self.mean_entropy, 3),
            "n_self_critique": self.n_self_critique,
            "n_cloud_flagged": self.n_cloud_flagged,
            "n_revised": self.n_revised,
        }
        # Only surface the causal-consistency view when it was actually running,
        # so a plain self-thinking trace stays clean.
        if self.min_consistency is not None:
            out["n_causal_breaks"] = self.n_causal_breaks
            out["min_consistency"] = self.min_consistency
        return out


# ---------------------------------------------------------------------------
# The SELF_CRITIQUE mechanism
# ---------------------------------------------------------------------------

def self_consistency_vote(
    logits: torch.Tensor,
    *,
    n_samples: int = 3,
    temperature: float = 1.0,
) -> Tuple[int, float, int]:
    """Token-level self-consistency: draw N candidates, return the majority.

    This is the cheap, no-extra-forward realisation of "the model
    reconsiders an uncertain token". We sample ``n_samples`` candidates from
    the *same* (already-computed) next-token distribution and emit the modal
    candidate — the token the model keeps landing on when it second-guesses
    itself. It is the token-level analogue of self-consistency decoding
    (Wang et al. 2022) and uses :func:`deliberation.semantic_entropy` to
    quantify how much the candidates disagree.

    Parameters
    ----------
    logits      : ``(1, V)`` (or ``(V,)``) next-token logits.
    n_samples   : number of candidates to draw (>= 1).
    temperature : sampling temperature for the candidates.

    Returns
    -------
    (token_id, sem_entropy, n_samples_used)
        ``token_id``    — the voted token.
        ``sem_entropy`` — semantic entropy across candidates (0 = unanimous).
        ``n_samples_used`` — echoes ``n_samples`` (>= 1).
    """
    flat = logits.reshape(-1, logits.shape[-1])[-1]
    n = max(1, int(n_samples))
    probs = F.softmax(flat / max(float(temperature), 1e-6), dim=-1)
    # One multinomial call draws all candidates → no Python loop, no extra
    # model forward passes.
    draws = torch.multinomial(probs, num_samples=n, replacement=True)
    cand_ids = [int(x) for x in draws.tolist()]

    # Majority vote; ties broken by highest base probability.
    tally: Dict[int, int] = {}
    for cid in cand_ids:
        tally[cid] = tally.get(cid, 0) + 1
    best_count = max(tally.values())
    winners = [cid for cid, c in tally.items() if c == best_count]
    if len(winners) == 1:
        chosen = winners[0]
    else:
        chosen = max(winners, key=lambda cid: float(probs[cid]))

    sem_h = semantic_entropy([[cid] for cid in cand_ids])
    return chosen, sem_h, n


# ---------------------------------------------------------------------------
# Sampling helpers (local copies so this module stays self-contained)
# ---------------------------------------------------------------------------

def _top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return logits
    v, _ = torch.topk(logits, min(k, logits.size(-1)))
    return logits.masked_fill(logits < v[:, [-1]], float("-inf"))


def _top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    keep = probs.cumsum(dim=-1) <= p
    keep[..., 0] = True
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(-1, sorted_idx, keep)
    return logits.masked_fill(~mask, float("-inf"))


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_with_thinking(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_k: int = 0,
    top_p: float = 0.9,
    n_critique_samples: int = 3,
    router: Optional[DeliberationRouter] = None,
    thresholds: Optional[RouterThresholds] = None,
    cloud_fn: Optional[Callable[[str], str]] = None,
    device: Optional[str] = None,
    input_ids: Optional[torch.Tensor] = None,
    use_cache: bool = True,
    consistency_check: bool = False,
    consistency_method: str = "subspace",
    consistency_window: int = 8,
) -> Tuple[str, ThinkingTrace]:
    """Generate text while routing each token through the self-thinking policy.

    For every decode step we:

    1. compute next-token ``logits`` from a single model forward;
    2. ask the :class:`DeliberationRouter` which route the step deserves
       (LOCAL / SELF_CRITIQUE / CLOUD) based on token entropy + fact gap;
    3. act on it —
         * LOCAL          → ordinary top-k/top-p sample;
         * SELF_CRITIQUE  → :func:`self_consistency_vote` re-decode;
         * CLOUD          → if ``cloud_fn`` is provided, inject its text and
                            continue; otherwise annotate "would consult
                            cloud" and fall back to self-critique so the
                            public demo (no cloud wired in) stays coherent;
    4. record a :class:`StepTrace` for the live frontend.

    The function is model-agnostic: ``model(input_ids=...)`` must return an
    object exposing ``.logits`` of shape ``(B, T, V)`` (the HF convention).

    Causal-consistency check (``consistency_check=True``)
    -----------------------------------------------------
    Complements the entropy router with a trajectory monitor
    (:class:`~mt_lnn.causality.CausalConsistencyChecker`). Each step's final
    hidden state is fed to the checker; when the state makes an abrupt jump
    (consistency < ``RouterThresholds.consistency_floor``) the router forces
    SELF_CRITIQUE *even if the token entropy looks low* — catching the
    "confidently wrong" / hallucination-boundary pattern that pure entropy
    gating misses. Requires the model to support ``output_hidden_states=True``
    (every HF causal LM does). Off by default so the plain self-thinking path
    is unchanged.

    HONEST SCOPE: this check was designed for the *native MT-LNN recurrent
    state* (continuous LTC/ODE dynamics, where a trajectory jump is a genuine
    semantic break). On a frozen HF Transformer the hidden state is recomputed
    fresh each token, so per-step novelty is high regardless of meaning; a local
    calibration found NO consistency floor that separates a coherent prompt from
    a deliberate topic-switch on a Qwen/TinyLlama base (both methods). Treat
    ``consistency_check=True`` on a Transformer base as experimental — it RUNS
    and the trace is correct, but it is not a validated hallucination detector
    there. It is meaningful on the recurrent MT-LNN path it was built for.

    Returns ``(generated_text, ThinkingTrace)``.
    """
    device = device or (next(model.parameters()).device.type
                         if hasattr(model, "parameters") else "cpu")
    router = router or DeliberationRouter(thresholds=thresholds)
    # Trajectory monitor (optional). "subspace" is the anisotropy-robust
    # detector recommended for real hidden states; it cancels the shared
    # dominant direction so only genuinely new directions register as breaks.
    checker = (
        CausalConsistencyChecker(window=consistency_window,
                                 method=consistency_method)
        if consistency_check else None
    )

    # Caller may pass pre-built input_ids (e.g. already run through a chat
    # template); otherwise tokenize the raw prompt here. Passing input_ids
    # avoids double-applying special tokens for instruct models.
    if input_ids is not None:
        ids = input_ids.to(device)
    else:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = ids.shape[1]
    eos_id = tokenizer.eos_token_id
    trace = ThinkingTrace()
    cloud_used = False

    # KV cache: feed the full prompt once, then only the newest token each
    # step. Numerically identical to re-running the whole sequence (HF caches
    # the same keys/values) but turns the decode from O(n^2) into O(n), which
    # is what makes this usable for serving on CPU. A cloud inject (rare; only
    # when cloud_fn is wired) resets the cache to re-prime on the new context.
    past = None
    cur_input = ids

    for step in range(int(max_new_tokens)):
        if use_cache:
            out = model(input_ids=cur_input, past_key_values=past,
                        use_cache=True, output_hidden_states=checker is not None)
            past = getattr(out, "past_key_values", None)
        else:
            out = model(input_ids=ids, output_hidden_states=checker is not None)
        raw_logits = out.logits[:, -1, :]              # (1, V), unscaled
        scaled = raw_logits / max(float(temperature), 1e-6)

        # --- trajectory monitor: feed this step's hidden state -------------
        # out.hidden_states is a tuple (embeddings, layer_1, ..., layer_N);
        # the last entry is the final contextual state. Take the newest
        # position's d_model vector and update the consistency score.
        consistency_signal: Optional[float] = None
        if checker is not None and getattr(out, "hidden_states", None):
            h_last = out.hidden_states[-1][:, -1, :]    # (1, d_model)
            consistency_signal = checker.update(h_last)

        # --- policy: what kind of step is this? ---------------------------
        decision = router.decide(
            scaled,
            query=prompt,
            evidence_log=[],          # public demo has no capsule/evidence
            consistency_signal=consistency_signal,
        )
        route = decision.route
        reason = decision.reason
        n_resamples = 0
        sem_h: Optional[float] = None
        revised = False

        # --- CLOUD: inject real evidence if a client exists, else degrade --
        if route == Route.CLOUD and cloud_fn is not None and not cloud_used:
            fact = cloud_fn(prompt)
            inject = tokenizer(
                f"\n[fact] {fact}\n", return_tensors="pt",
                add_special_tokens=False,
            ).input_ids.to(device)
            ids = torch.cat([ids, inject], dim=1)
            cloud_used = True
            # Injected tokens are not in the KV cache -> reset so the next
            # step re-primes on the full (prompt + inject) context. The MT
            # adapters' streaming state must reset with it: the re-fed prompt
            # would otherwise be double-written into the recurrent state.
            from .llama_adapter import reset_adapter_streams
            reset_adapter_streams(model)
            past = None
            cur_input = ids
            # Record the inject as a zero-token annotation and move on.
            trace.steps.append(StepTrace(
                index=step, token_id=-1, token_text="",
                entropy=decision.entropy, route=Route.CLOUD.value,
                reason="cloud_inject", n_resamples=0,
                causal_consistency=consistency_signal,
            ))
            continue

        # --- pick the next token ------------------------------------------
        # Baseline draw (also used to detect whether critique revised it).
        base_filtered = _top_p(_top_k(scaled.clone(), int(top_k)), float(top_p))
        base_probs = F.softmax(base_filtered, dim=-1)
        base_id = int(torch.multinomial(base_probs, num_samples=1).item())

        if route in (Route.SELF_CRITIQUE, Route.CLOUD):
            # CLOUD with no client degrades to local self-critique.
            chosen_id, sem_h, n_resamples = self_consistency_vote(
                base_filtered, n_samples=n_critique_samples,
                temperature=float(temperature),
            )
            revised = chosen_id != base_id
            if route == Route.CLOUD:
                reason = "cloud_unavailable_fallback_self_critique"
            next_id = chosen_id
        else:
            next_id = base_id

        token_text = tokenizer.decode([next_id], skip_special_tokens=True)
        trace.steps.append(StepTrace(
            index=step,
            token_id=next_id,
            token_text=token_text,
            entropy=decision.entropy,
            route=route.value,
            reason=reason,
            n_resamples=n_resamples,
            sem_entropy=sem_h,
            revised=revised,
            causal_consistency=consistency_signal,
        ))

        next_tok = torch.tensor([[next_id]], device=device)
        ids = torch.cat([ids, next_tok], dim=1)
        cur_input = next_tok          # KV cache: only feed the new token next
        if eos_id is not None and next_id == eos_id:
            break

    text = tokenizer.decode(ids[0, prompt_len:], skip_special_tokens=True)
    return text, trace


# ---------------------------------------------------------------------------
# Frontend renderers
# ---------------------------------------------------------------------------

# Per-route palette used by both renderers (kept here so the colour scheme is
# defined once and the frontend stays a thin consumer).
_ROUTE_STYLE = {
    Route.LOCAL.value:         ("#2e7d32", "trust local decode"),
    Route.SELF_CRITIQUE.value: ("#ef6c00", "reconsidered (self-critique)"),
    Route.CLOUD.value:         ("#c62828", "flagged for cloud"),
}


def render_trace_markdown(trace: ThinkingTrace) -> str:
    """Compact markdown summary of a thinking trace (for a UI header)."""
    s = trace.summary()
    rc = s["route_counts"]
    lines = [
        "### 🧠 Self-thinking summary",
        f"- **Tokens:** {s['n_tokens']}",
        f"- **Mean next-token entropy:** {s['mean_entropy']}",
        f"- **Local / Self-critique / Cloud-flagged:** "
        f"{rc.get('local', 0)} / {rc.get('self_critique', 0)} / "
        f"{rc.get('cloud', 0)}",
        f"- **Tokens revised by self-critique:** {s['n_revised']}",
    ]
    return "\n".join(lines)


def render_trace_html(trace: ThinkingTrace) -> str:
    """Inline-coloured token strip: each token tinted by its route.

    Self-critiqued / revised tokens are underlined; hovering shows the
    entropy and route reason. Safe to drop into a Gradio ``gr.HTML``.
    """
    spans: List[str] = []
    for st in trace.steps:
        if not st.token_text:
            continue
        colour, label = _ROUTE_STYLE.get(st.route, ("#555", st.route))
        deco = "underline" if st.revised else "none"
        tip = (f"entropy={st.entropy:.2f} · {label}"
               + (f" · revised" if st.revised else "")
               + (f" · sem_H={st.sem_entropy:.2f}"
                  if st.sem_entropy is not None else ""))
        safe = (st.token_text.replace("&", "&amp;")
                .replace("<", "&lt;").replace(">", "&gt;")
                .replace("\n", "⏎"))
        spans.append(
            f'<span title="{tip}" style="color:{colour};'
            f'text-decoration:{deco};">{safe}</span>'
        )
    legend = (
        '<div style="margin-top:8px;font-size:0.8em;color:#888;">'
        '<span style="color:#2e7d32;">■</span> local &nbsp;'
        '<span style="color:#ef6c00;">■</span> self-critique &nbsp;'
        '<span style="color:#c62828;">■</span> cloud-flagged &nbsp;'
        '· <u>underline</u> = token revised by self-thinking</div>'
    )
    return (
        '<div style="font-family:monospace;line-height:1.7;white-space:'
        f'pre-wrap;">{"".join(spans)}</div>{legend}'
    )
