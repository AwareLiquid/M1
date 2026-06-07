"""Lightweight observability helpers for MT-LNN scripts and benchmarks.

The core model stays framework-agnostic. This module provides small utilities
for structured logs/metrics so demos and benchmarks can expose cache growth,
latency, and compression diagnostics without depending on a production stack.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union


def setup_logging(level: Union[int, str] = "INFO") -> logging.Logger:
    """Configure a simple stderr logger and return the project logger."""
    logger = logging.getLogger("mt_lnn")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level if isinstance(level, int) else level.upper())
    return logger


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


class JsonlMetricWriter:
    """Append one JSON object per metric event.

    Parameters
    ----------
    path:
        Destination JSONL file. Parent directories are created automatically.
    static_fields:
        Fields included in every event, for example benchmark name or device.
    """

    def __init__(
        self,
        path: Union[str, Path],
        static_fields: Optional[Mapping[str, Any]] = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.static_fields = dict(static_fields or {})
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, event: str, fields: Optional[Mapping[str, Any]] = None) -> None:
        row: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **self.static_fields,
        }
        if fields:
            row.update(dict(fields))
        self._fh.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JsonlMetricWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ---------------------------------------------------------------------------
# v2.0 module observability — the "eyes" for long pre-training runs.
# ---------------------------------------------------------------------------

# Canonical scalar keys pulled from MTLNNModel.get_mt_diagnostics(). Only the
# v2.0 module metrics are selected here so the JSONL stays small and focused.
# All values are plain Python floats (JSON-serialisable, not tensors).
_V2_DIAG_KEYS = (
    # Phase A — competitive workspace
    "gwtb_competition_entropy",   # 0 = monopoly, log(K) = uniform
    "gwtb_bid0_weight",
    "gwtb_bid1_weight",
    "gwtb_bid2_weight",
    "gwtb_broadcast_gate",
    "gwtb_external_bid_weight",    # P3.1 multi-source: mass on external module bids
    "gwtb_world_bid_gate",         # P3.1: learned trust in the world-model bid
    # Phase C — predictive world model
    "world_model_pred_error",     # normalised surprise ∈ [0, 1]
    "world_model_pred_error_raw", # raw latent MSE magnitude
    # Phase D — Hebbian plasticity
    "hebbian_signal_mean",
    "hebbian_lavi_temperature",
    # LAVI rhythm
    "lavi_mean",
    "rhythm_scale_mean",
    "global_rhythm_scale",
)


def v2_module_metrics(
    model: Any,
    step: Optional[int] = None,
    checker: Any = None,
) -> Dict[str, Any]:
    """Collect a flat, scalar metric dict for the four v2.0 modules.

    The single source of truth is ``model.get_mt_diagnostics()`` (which already
    returns Python scalars). Only the v2.0 module keys are kept. An optional
    ``CausalConsistencyChecker`` (Phase B) is a stateless inference-time object,
    not part of the model graph, so it is passed in separately when available.

    Parameters
    ----------
    model : MTLNNModel
        A model exposing ``get_mt_diagnostics()``.
    step : int, optional
        Training step, copied into the row when provided.
    checker : CausalConsistencyChecker, optional
        When given, ``causal_consistency`` and ``causal_effective_rank`` are
        added from its current state.

    Returns
    -------
    dict
        ``{step?, <v2 diag keys present>, causal_*?}`` — all values are floats.
        Keys absent from diagnostics (module disabled) are simply omitted.
    """
    metrics: Dict[str, Any] = {}
    if step is not None:
        metrics["step"] = int(step)

    diag = model.get_mt_diagnostics() if hasattr(model, "get_mt_diagnostics") else {}
    for key in _V2_DIAG_KEYS:
        if key in diag and diag[key] is not None:
            metrics[key] = float(diag[key])

    if checker is not None:
        # consistency_score() and effective_rank are both plain floats.
        score = getattr(checker, "consistency_score", None)
        if callable(score):
            metrics["causal_consistency"] = float(checker.consistency_score())
        eff = getattr(checker, "effective_rank", None)
        if eff is not None:
            metrics["causal_effective_rank"] = float(eff)

    return metrics


def record_v2_metrics(
    writer: "JsonlMetricWriter",
    model: Any,
    step: int,
    checker: Any = None,
) -> Dict[str, Any]:
    """Convenience: collect v2.0 module metrics and append them to a JSONL writer.

    Intended to be called every N steps during pre-training (e.g. every 100)::

        writer = JsonlMetricWriter("metrics.jsonl", static_fields={"run": "125m"})
        if step % 100 == 0:
            record_v2_metrics(writer, model, step, checker)

    Returns the metric dict that was written (handy for stdout logging too).
    """
    metrics = v2_module_metrics(model, step=step, checker=checker)
    writer.write("v2_modules", metrics)
    return metrics


def cache_summary(cache: Any) -> Dict[str, Any]:
    """Return a small, serializable cache summary.

    Works with ``ModelCacheStruct`` without importing it at module load time.
    """
    if cache is None:
        return {
            "token_count": 0,
            "cache_bytes": 0,
            "layers": 0,
            "has_attention_kv": False,
            "has_recurrent_state": False,
            "has_gwtb_kv": False,
            "has_coherence_kv": False,
        }

    layers = getattr(cache, "layers", []) or []
    has_attention_kv = any(layer is not None and layer[0] is not None for layer in layers)
    has_recurrent_state = any(
        layer is not None and len(layer) > 1 and layer[1] is not None for layer in layers
    )
    has_block_gwtb_kv = any(
        layer is not None and len(layer) > 2 and layer[2] is not None for layer in layers
    )
    cache_bytes = cache.tensor_bytes() if hasattr(cache, "tensor_bytes") else None

    return {
        "token_count": getattr(cache, "token_count", 0),
        "cache_bytes": cache_bytes,
        "layers": len(layers),
        "has_attention_kv": has_attention_kv,
        "has_recurrent_state": has_recurrent_state,
        "has_gwtb_kv": has_block_gwtb_kv or getattr(cache, "gwtb_kv", None) is not None,
        "has_coherence_kv": getattr(cache, "coherence_kv", None) is not None,
    }
