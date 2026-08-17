"""
reasoning_trace.py — Per-step reasoning trace for AwareLiquid.

Layer 3 of the AwareLiquid architecture: emit a structured, auditable
record of *how* a response was produced. Each generated token (or token
range) gets one event with:

    step          — monotonically increasing step index
    token_id      — sampled token id (or None for non-token events)
    entropy       — Shannon entropy of next-token logits
    route         — "local" | "self_critique" | "cloud" | "inject"
    phi           — optional IIT-Φ sample (may be None; expensive)
    extras        — free-form dict (e.g. cloud source, query)

Events are written via the existing ``JsonlMetricWriter`` so they can be
replayed offline or rendered as a UI timeline (see ``ui.html``).

Φ sampling is intentionally sparse: PyPhi costs O(2^n). Default behaviour
is to call ``compute_iit_phi_from_model`` every ``phi_every`` steps with a
small node budget. If PyPhi is unavailable the trace still records
entropy + route and silently skips Φ.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from .observability import JsonlMetricWriter

# Phi support is optional. Importing should never raise at module load time.
try:
    from .phi_iit import PYPHI_AVAILABLE, compute_iit_phi_from_model
except Exception:  # pragma: no cover — defensive
    PYPHI_AVAILABLE = False
    compute_iit_phi_from_model = None  # type: ignore[assignment]


class ReasoningTrace:
    """Lightweight per-step reasoning recorder.

    Parameters
    ----------
    path : where to append JSONL events.
    session_id : identifier propagated into every event.
    phi_every : sample Φ every N token steps (set to 0 to disable).
    phi_nodes : number of protofilament nodes for Φ (small for speed).
    """

    def __init__(
        self,
        path: Union[str, Path],
        *,
        session_id: str = "default",
        phi_every: int = 0,
        phi_nodes: int = 4,
    ):
        self.writer = JsonlMetricWriter(
            path,
            static_fields={"session_id": session_id, "channel": "reasoning_trace"},
        )
        self.session_id = session_id
        self.phi_every = phi_every
        self.phi_nodes = phi_nodes
        self._step = 0

    # ------------------------------------------------------------------ events

    def record_token(
        self,
        *,
        token_id: Optional[int],
        entropy: float,
        route: str = "local",
        phi: Optional[float] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a single decoding step."""
        self._step += 1
        fields: Dict[str, Any] = {
            "step": self._step,
            "token_id": token_id,
            "entropy": float(entropy),
            "route": route,
        }
        if phi is not None:
            fields["phi"] = float(phi)
        if extras:
            fields.update(extras)
        self.writer.write("token", fields)

    def record_route(
        self,
        *,
        route: str,
        reason: str,
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a routing decision (without consuming a step counter)."""
        fields: Dict[str, Any] = {"step": self._step, "route": route, "reason": reason}
        if extras:
            fields.update(extras)
        self.writer.write("route", fields)

    def record_cloud_inject(
        self,
        *,
        source: str,
        query: str,
        fact_len: int,
    ) -> None:
        """Record an evidence injection from the cloud oracle."""
        self.writer.write(
            "cloud_inject",
            {
                "step": self._step,
                "source": source,
                "query": query,
                "fact_len": fact_len,
            },
        )

    # ------------------------------------------------------------------- phi

    def should_sample_phi(self) -> bool:
        if self.phi_every <= 0:
            return False
        if not PYPHI_AVAILABLE:
            return False
        return self._step > 0 and (self._step % self.phi_every == 0)

    def sample_phi(
        self,
        model,
        probe_ids: torch.Tensor,
    ) -> Optional[float]:
        """Run a sparse Φ measurement; returns None if unavailable.

        ``probe_ids`` is a small ``(B, T)`` token batch used to drive the
        protofilament TPM estimate. Callers typically pass the last few
        generated tokens or a fixed probe sequence.
        """
        if not PYPHI_AVAILABLE or compute_iit_phi_from_model is None:
            return None
        try:
            phi = compute_iit_phi_from_model(
                model,
                probe_ids,
                n_nodes=self.phi_nodes,
                n_samples=64,
            )
            return float(phi)
        except Exception as exc:  # pragma: no cover — Φ failures are non-fatal
            self.writer.write("phi_error", {"step": self._step, "error": str(exc)})
            return None

    # ----------------------------------------------------------------- close

    def close(self) -> None:
        self.writer.close()

    def __enter__(self) -> "ReasoningTrace":
        return self

    def __exit__(self, *_) -> None:
        self.close()
