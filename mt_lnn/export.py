"""
mt_lnn/export.py — TorchScript export + on-device latency benchmarking.

The full :meth:`MTLNNModel.forward` returns a ``dict`` and threads an optional
dataclass KV/coherence cache with Python-level control flow, so it is not
``torch.jit.script``-able. For deployment we export the **logits path** only: a
thin wrapper that runs the model in eval with ``use_lnn_recurrence=False`` (the
training-parallel mode, which makes a traced fixed-shape graph well defined) and
returns just the logits tensor.

``torch.jit.trace`` bakes in the sequence length (global coherence performs a
seq-length-dependent reduction), so a traced artifact is valid only for the
``(batch, seq_len)`` it was traced at — export one artifact per target shape.

Nothing here changes training or eager inference; it is an opt-in deployment
utility. Importing this module has no side effects on the model.
"""
from __future__ import annotations

import statistics
import time
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

__all__ = [
    "LogitsOnlyWrapper",
    "trace_logits_module",
    "benchmark_latency",
    "export_and_benchmark",
]


class LogitsOnlyWrapper(nn.Module):
    """Wrap an :class:`MTLNNModel` so ``forward(input_ids) -> logits`` (a single
    tensor), which is what ``torch.jit.trace`` and most edge runtimes expect.

    Runs the inner model with ``use_lnn_recurrence=False`` so the exported graph
    matches the training-time parallel path (no cross-step recurrent threading),
    keeping the traced computation pure w.r.t. the inputs.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, use_lnn_recurrence=False)["logits"]


def trace_logits_module(
    model: nn.Module,
    *,
    batch: int = 1,
    seq_len: int = 64,
    optimize: bool = False,
    check_parity: bool = True,
    atol: float = 1e-3,
    device: str = "cpu",
) -> tuple[torch.jit.ScriptModule, Dict]:
    """Trace ``model``'s logits path at a fixed ``(batch, seq_len)``.

    Returns ``(traced_module, info)``. ``info`` records the trace shape, the
    eager-vs-traced max-abs parity, and the device. Raises if parity exceeds
    ``atol`` (a silent numeric divergence in an exported artifact is a bug).
    """
    model = model.eval().to(device)
    wrapper = LogitsOnlyWrapper(model).eval()
    vocab = int(model.config.vocab_size)
    ids = torch.randint(0, vocab, (batch, seq_len), device=device)

    with torch.no_grad():
        eager = wrapper(ids)
        traced = torch.jit.trace(wrapper, ids, check_trace=False)

    parity: Optional[float] = None
    if check_parity:
        with torch.no_grad():
            parity = (traced(ids) - eager).abs().max().item()
        if not (parity <= atol):
            raise RuntimeError(
                f"traced/eager logits diverge: max|Δ|={parity:.3e} > atol={atol:.0e}"
            )

    if optimize:
        # Fold/freeze for inference. Best-effort: some op sets aren't supported
        # by optimize_for_inference on every torch build, so degrade gracefully.
        try:
            traced = torch.jit.optimize_for_inference(traced)
        except Exception as exc:  # pragma: no cover - build-dependent
            print(f"[export] optimize_for_inference skipped: {exc}")
            optimize = False

    info = {
        "batch": batch,
        "seq_len": seq_len,
        "parity_max_abs": parity,
        "optimized": optimize,
        "device": device,
        "vocab_size": vocab,
    }
    return traced, info


def benchmark_latency(
    fn: Callable[[], object],
    *,
    iters: int = 50,
    warmup: int = 5,
    sync_cuda: bool = False,
) -> Dict[str, float]:
    """Time a zero-arg ``fn`` (typically ``lambda: module(ids)``).

    Returns median/mean/p90/min wall-clock milliseconds. ``sync_cuda`` inserts a
    device sync after each call so CUDA timings aren't measuring async launch.
    """
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            fn()
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            if sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    n = len(samples)
    return {
        "iters": float(n),
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "p90_ms": samples[min(n - 1, int(0.9 * (n - 1)))],
        "min_ms": samples[0],
    }


def export_and_benchmark(
    model: nn.Module,
    *,
    batch: int = 1,
    seq_len: int = 64,
    optimize: bool = False,
    iters: int = 50,
    warmup: int = 5,
    device: str = "cpu",
    save_path: Optional[str] = None,
    compare_eager: bool = True,
) -> Dict:
    """End-to-end: trace the logits path, (optionally) save it, and benchmark
    scripted vs eager latency. Returns a JSON-friendly report dict including
    per-token milliseconds (the on-device deployment metric)."""
    traced, info = trace_logits_module(
        model, batch=batch, seq_len=seq_len, optimize=optimize, device=device,
    )

    sync = device.startswith("cuda")
    vocab = info["vocab_size"]
    ids = torch.randint(0, vocab, (batch, seq_len), device=device)

    scripted_lat = benchmark_latency(
        lambda: traced(ids), iters=iters, warmup=warmup, sync_cuda=sync,
    )
    tokens = batch * seq_len
    report = {
        **info,
        "scripted": scripted_lat,
        "scripted_ms_per_token": scripted_lat["median_ms"] / tokens,
        "tokens_per_call": tokens,
    }

    if compare_eager:
        wrapper = LogitsOnlyWrapper(model.eval().to(device)).eval()
        eager_lat = benchmark_latency(
            lambda: wrapper(ids), iters=iters, warmup=warmup, sync_cuda=sync,
        )
        report["eager"] = eager_lat
        report["eager_ms_per_token"] = eager_lat["median_ms"] / tokens
        report["speedup_vs_eager"] = (
            eager_lat["median_ms"] / scripted_lat["median_ms"]
            if scripted_lat["median_ms"] > 0 else float("nan")
        )

    if save_path is not None:
        torch.jit.save(traced, save_path)
        report["saved_to"] = save_path

    return report
