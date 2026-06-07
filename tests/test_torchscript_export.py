"""
tests/test_torchscript_export.py — TorchScript export / on-device perf path.

Pins the deployment contract for mt_lnn/export.py:
  • the logits path traces and matches eager bit-for-bit (within atol),
  • the artifact saves and reloads and reruns at its traced shape,
  • optimize_for_inference still produces a numerically faithful module,
  • latency benchmarking returns sane positive timings + a per-token figure,
  • export is a pure deployment utility — it never perturbs the eager model.

Tiny dims; CPU; fast under pytest. The < 50 ms/token target is a *deployment*
figure (depends on the serving device), so here we only assert timings are
finite/positive rather than pinning an absolute ceiling that would flake on CI.
"""
import math

import torch

from mt_lnn.config import MTLNNConfig
from mt_lnn.export import (
    LogitsOnlyWrapper,
    benchmark_latency,
    export_and_benchmark,
    trace_logits_module,
)
from mt_lnn.model import MTLNNModel


def _cfg():
    return MTLNNConfig(
        vocab_size=256, d_model=104, n_layers=2, n_heads=13, n_kv_heads=1,
        d_head=8, max_seq_len=64, gwtb_n_heads=1, dropout=0.0,
        attention_dropout=0.0,
        use_competitive_gwtb=True, use_world_model=True, use_hebbian=True,
    )


def _model():
    torch.manual_seed(0)
    return MTLNNModel(_cfg()).eval()


# ---------------------------------------------------------------------------
# Trace parity
# ---------------------------------------------------------------------------

def test_trace_matches_eager_within_atol():
    model = _model()
    traced, info = trace_logits_module(model, batch=1, seq_len=16)
    assert info["parity_max_abs"] is not None
    assert info["parity_max_abs"] <= 1e-3, info["parity_max_abs"]
    assert info["seq_len"] == 16 and info["batch"] == 1


def test_traced_logits_equal_wrapper_logits():
    model = _model()
    wrapper = LogitsOnlyWrapper(model).eval()
    ids = torch.randint(0, 256, (1, 16))
    traced, _ = trace_logits_module(model, batch=1, seq_len=16)
    with torch.no_grad():
        a = wrapper(ids)
        b = traced(ids)
    assert a.shape == b.shape == (1, 16, 256)
    assert torch.allclose(a, b, atol=1e-3)


# ---------------------------------------------------------------------------
# Save / load roundtrip + rerun at traced shape
# ---------------------------------------------------------------------------

def test_save_load_roundtrip_reruns(tmp_path):
    model = _model()
    traced, _ = trace_logits_module(model, batch=1, seq_len=16)
    path = tmp_path / "mt_lnn_ts.pt"
    torch.jit.save(traced, str(path))
    reloaded = torch.jit.load(str(path))
    ids = torch.randint(0, 256, (1, 16))
    with torch.no_grad():
        out = reloaded(ids)
    assert out.shape == (1, 16, 256)
    assert torch.isfinite(out).all()


def test_optimize_for_inference_preserves_outputs():
    model = _model()
    plain, _ = trace_logits_module(model, batch=1, seq_len=16, check_parity=False)
    opt, info = trace_logits_module(model, batch=1, seq_len=16, optimize=True)
    ids = torch.randint(0, 256, (1, 16))
    with torch.no_grad():
        assert torch.allclose(plain(ids), opt(ids), atol=1e-3)


# ---------------------------------------------------------------------------
# Latency benchmarking
# ---------------------------------------------------------------------------

def test_benchmark_latency_returns_positive_timings():
    model = _model()
    traced, _ = trace_logits_module(model, batch=1, seq_len=16)
    ids = torch.randint(0, 256, (1, 16))
    stats = benchmark_latency(lambda: traced(ids), iters=8, warmup=2)
    assert stats["iters"] == 8
    for k in ("median_ms", "mean_ms", "p90_ms", "min_ms"):
        assert math.isfinite(stats[k]) and stats[k] >= 0.0
    assert stats["min_ms"] <= stats["median_ms"] <= stats["p90_ms"] + 1e-6


def test_export_and_benchmark_report_shape():
    model = _model()
    report = export_and_benchmark(
        model, batch=1, seq_len=16, iters=8, warmup=2, compare_eager=True,
    )
    assert report["tokens_per_call"] == 16
    assert math.isfinite(report["scripted_ms_per_token"])
    assert report["scripted_ms_per_token"] > 0.0
    # eager comparison present and speedup finite
    assert "eager" in report
    assert math.isfinite(report["speedup_vs_eager"])


# ---------------------------------------------------------------------------
# Zero side effects
# ---------------------------------------------------------------------------

def test_export_does_not_mutate_eager_model():
    model = _model()
    ids = torch.randint(0, 256, (1, 16))
    with torch.no_grad():
        before = model(ids, use_lnn_recurrence=False)["logits"].clone()
    _ = export_and_benchmark(model, batch=1, seq_len=16, iters=4, warmup=1,
                             compare_eager=True)
    with torch.no_grad():
        after = model(ids, use_lnn_recurrence=False)["logits"]
    assert torch.allclose(before, after, atol=1e-5)


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        try:
            try:
                fn()
            except TypeError:
                import tempfile, pathlib
                fn(pathlib.Path(tempfile.mkdtemp()))
            print(f"[ok] {fn.__name__}")
        except Exception:
            print(f"[FAIL] {fn.__name__}")
            traceback.print_exc()
            raise
