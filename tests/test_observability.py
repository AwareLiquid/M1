import json
import warnings

import torch

warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel, ModelCacheStruct
from mt_lnn.observability import (
    JsonlMetricWriter,
    cache_summary,
    record_v2_metrics,
    v2_module_metrics,
)


def _v2_model():
    """Tiny model with all four v2.0 modules switched ON."""
    cfg = MTLNNConfig(
        vocab_size=200,
        max_seq_len=64,
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_head=32,
        dropout=0.0,
        attention_dropout=0.0,
        use_world_model=True,
        use_competitive_gwtb=True,
        use_hebbian=True,
        global_rhythm=True,
    )
    return MTLNNModel(cfg)


def test_cache_summary_distinguishes_recurrent_only_cache():
    cache = ModelCacheStruct(token_count=7)
    h_prev = torch.zeros(1, 13, 5, 8)
    cache.layers.append((None, h_prev, None))

    summary = cache_summary(cache)

    assert summary["token_count"] == 7
    assert summary["cache_bytes"] == h_prev.numel() * h_prev.element_size()
    assert summary["layers"] == 1
    assert summary["has_attention_kv"] is False
    assert summary["has_recurrent_state"] is True
    assert summary["has_gwtb_kv"] is False
    assert summary["has_coherence_kv"] is False


def test_jsonl_metric_writer_appends_structured_events(tmp_path):
    path = tmp_path / "metrics.jsonl"

    with JsonlMetricWriter(path, static_fields={"benchmark": "unit"}) as writer:
        writer.write("cache", {"cache_bytes": 123, "ok": True})

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["event"] == "cache"
    assert rows[0]["benchmark"] == "unit"
    assert rows[0]["cache_bytes"] == 123
    assert rows[0]["ok"] is True
    assert "ts" in rows[0]


# ---------------------------------------------------------------------------
# v2.0 module metrics
# ---------------------------------------------------------------------------

def test_v2_module_metrics_are_flat_json_scalars():
    model = _v2_model().eval()
    ids = torch.randint(0, 200, (2, 16))
    with torch.no_grad():
        model(ids, labels=ids)  # populate diagnostics buffers

    metrics = v2_module_metrics(model, step=300)

    assert metrics["step"] == 300
    # every value must be a JSON-serialisable scalar (float / int / bool)
    for key, value in metrics.items():
        assert isinstance(value, (int, float, bool)), f"{key}={value!r} not scalar"
    # the world-model surprise signal is the normalised [0,1] one
    assert "world_model_pred_error" in metrics
    assert 0.0 <= metrics["world_model_pred_error"] <= 1.0


def test_v2_module_metrics_includes_checker_state_when_supplied():
    model = _v2_model().eval()
    ids = torch.randint(0, 200, (1, 16))
    with torch.no_grad():
        model(ids, labels=ids)

    from mt_lnn.causality import CausalConsistencyChecker

    checker = CausalConsistencyChecker(window=4, ema_alpha=0.5, threshold=0.3)
    for _ in range(4):
        checker.update(torch.randn(8))

    metrics = v2_module_metrics(model, step=1, checker=checker)
    assert "causal_consistency" in metrics
    assert isinstance(metrics["causal_consistency"], float)


def test_record_v2_metrics_writes_jsonl_row(tmp_path):
    model = _v2_model().eval()
    ids = torch.randint(0, 200, (1, 16))
    with torch.no_grad():
        model(ids, labels=ids)

    path = tmp_path / "v2.jsonl"
    with JsonlMetricWriter(path, static_fields={"run": "unit"}) as writer:
        returned = record_v2_metrics(writer, model, step=100)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["event"] == "v2_modules"
    assert rows[0]["run"] == "unit"
    assert rows[0]["step"] == 100
    # round-trips: the written row matches the returned dict (plus envelope keys)
    for key, value in returned.items():
        assert rows[0][key] == value
