"""
tests/test_demo_causal_decoding.py -- L3 steering-in-real-decoding demo.

Pins the behavioural contract of ``examples/demo_causal_decoding.py`` (the
production-path L3 validation -- steering applied inside ``MTLNNModel.generate``
on the live recurrent cache) so it cannot silently regress:

  * steering fires on the injected in-decode belief jump (>= 1 correction);
  * it pulls the corrupted recurrent trajectory back toward baseline (steered
    drift < injection-only drift, by a clear margin);
  * SAFETY: on a clean (un-injected) run the gate never trips -- zero
    corrections and decoding is bit-for-bit identical to vanilla;
  * the demo adds zero trainable parameters;
  * the verdict is deterministic for a fixed seed and robust across seeds.

Assertions are loose (mechanism-level), pinning *that steering acts in the real
decode loop and helps on a break while never harming a healthy run* rather than
exact floats.
"""
import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.demo_causal_decoding import (  # noqa: E402
    build_model,
    decode_with_probe,
    mean_drift,
    run_demo,
)


def _args(**kw):
    base = dict(
        seed=0, steps=18, inject_step=8, noise=5.0, prompt_len=6,
        d_model=104, n_layers=2, plot=False, plot_path="unused.png",
    )
    base.update(kw)
    return type("A", (), base)()


def test_run_demo_verdict_and_report(capsys):
    summary = run_demo(_args())
    assert summary["verdict"] is True
    assert summary["corrections"] >= 1
    assert summary["drift_steered"] < summary["drift_injection"]
    assert summary["drift_reduction_pct"] > 5.0
    out = capsys.readouterr().out
    assert "VERDICT" in out and "[OK]" in out


def test_safety_clean_run_is_noop():
    summary = run_demo(_args())
    # Steering a healthy (un-injected) trajectory must never fire or change output.
    assert summary["clean_run_corrections"] == 0
    assert summary["clean_run_unchanged"] is True


def test_steering_reduces_drift_directly():
    # Reproduce the three-way comparison at the function level.
    model = build_model(seed=1)
    g = torch.Generator().manual_seed(101)
    prompt = torch.randint(0, 64, (1, 6), generator=g)
    base, _, _ = decode_with_probe(model, prompt, inject_step=None, steer_on=False,
                                   max_new_tokens=18, noise=5.0, seed=8)
    inj, _, _ = decode_with_probe(model, prompt, inject_step=8, steer_on=False,
                                  max_new_tokens=18, noise=5.0, seed=8)
    both, steer, _ = decode_with_probe(model, prompt, inject_step=8, steer_on=True,
                                       max_new_tokens=18, noise=5.0, seed=8)
    assert steer.n_corrections >= 1
    assert mean_drift(both, base, frm=8) < mean_drift(inj, base, frm=8)


def test_demo_is_zero_parameter():
    # The whole steering path adds no trainable parameters to the model.
    model = build_model(seed=0)
    n_before = sum(p.numel() for p in model.parameters())
    g = torch.Generator().manual_seed(100)
    prompt = torch.randint(0, 64, (1, 6), generator=g)
    _, steer, _ = decode_with_probe(model, prompt, inject_step=8, steer_on=True,
                                    max_new_tokens=10, noise=5.0, seed=7)
    n_after = sum(p.numel() for p in model.parameters())
    assert n_after == n_before
    assert not hasattr(steer, "parameters")          # driver is not an nn.Module


def test_deterministic_for_fixed_seed():
    a = run_demo(_args(seed=3))
    b = run_demo(_args(seed=3))
    assert a["drift_injection"] == pytest.approx(b["drift_injection"])
    assert a["drift_steered"] == pytest.approx(b["drift_steered"])
    assert a["corrections"] == b["corrections"]


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 5])
def test_verdict_robust_across_seeds(seed):
    assert run_demo(_args(seed=seed))["verdict"] is True
