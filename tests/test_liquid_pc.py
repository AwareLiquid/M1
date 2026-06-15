"""
tests/test_liquid_pc.py — contract for the INTEGRATED predictive-coding liquid
core (experiments/liquid_pc/model.py, 2026-06-15).

This is the architecture-integration deliverable, so the tests pin that the
predictive-coding loop really IS the continuous-time recurrence, not a bolt-on:
  * next-step output shape; causal (xhat[:, t] depends only on x[:, <=t]);
  * the exp(-dt/tau) liquid kernel is present and multi-timescale (per-level taus
    span fast->slow); decay in (0, 1);
  * predictive-coding errors drive the dynamics: with the generative weights
    forced to predict perfectly (zero error), the hidden state stops evolving;
  * the model trains (overfits a tiny batch -> loss drops a lot);
  * deterministic for fixed input/seed; O(1) state (no per-step parameter growth);
  * baselines build and run with comparable parameter budgets.

Run:  python -m pytest tests/test_liquid_pc.py -v
"""
import os
import sys

import pytest
import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.liquid_pc.model import (                              # noqa: E402
    PCLiquidCore,
    build_models,
    count_params,
)
from experiments.liquid_pc.data import make_signal, two_regimes        # noqa: E402


# --------------------------------------------------------------------------- #
# shape / causality                                                           #
# --------------------------------------------------------------------------- #
def test_next_step_output_shape():
    m = PCLiquidCore(d_in=2, d=16, n_levels=3)
    x = torch.randn(4, 20, 2)
    y = m(x)
    assert y.shape == (4, 20, 2)


def test_prediction_is_causal():
    # Changing x at time t must not change predictions at times < t.
    torch.manual_seed(0)
    m = PCLiquidCore(d_in=1, d=16, n_levels=2).eval()
    x = torch.randn(1, 12, 1)
    y0 = m(x)
    x2 = x.clone()
    x2[:, 7, :] += 5.0                                   # perturb t=7
    y1 = m(x2)
    # predictions for t < 7 are unchanged; from t=7 on they may change.
    assert torch.allclose(y0[:, :7], y1[:, :7], atol=1e-6)
    assert not torch.allclose(y0[:, 7:], y1[:, 7:], atol=1e-6)


# --------------------------------------------------------------------------- #
# the liquid kernel is real & multi-timescale                                 #
# --------------------------------------------------------------------------- #
def test_decay_is_multi_timescale_and_bounded():
    m = PCLiquidCore(d_in=1, d=8, n_levels=4, tau_min=1.0, tau_max=40.0)
    decay = m._decay()
    assert decay.shape == (4,)
    assert (decay > 0.0).all() and (decay < 1.0).all()
    # initialised fast->slow: higher levels have larger decay (slower leak).
    assert torch.all(decay[1:] >= decay[:-1] - 1e-4)
    assert decay[-1] > decay[0]                          # top level genuinely slower


def test_precision_is_positive_and_per_level():
    m = PCLiquidCore(d_in=1, d=8, n_levels=3)
    prec = m._precision()
    assert prec.shape == (3,)
    assert (prec > 0.0).all()
    # default init: softplus(log_precision) == 1.0 -> precision-free at start, so
    # a freshly built precision core is identical to the original (clean ablation).
    assert torch.allclose(prec, torch.ones(3), atol=1e-5)


def test_zero_precision_freezes_state():
    # Precision gates how strongly errors drive the state. Drive precision to ~0
    # (very negative log-precision) and every error is down-weighted to nothing,
    # so the liquid state cannot move from its zero init -> readout constant.
    m = PCLiquidCore(d_in=1, d=8, n_levels=2).eval()
    with torch.no_grad():
        m.log_precision.fill_(-30.0)                     # softplus(-30) ~ 0
        for rec in m.recognize:                          # null bias so drive->tanh(0)
            rec.bias.zero_()
    x = torch.randn(1, 6, 1)
    y = m(x)
    assert torch.allclose(y[:, 0], y[:, -1], atol=1e-6)


def test_precision_is_trainable_and_changes_dynamics():
    torch.manual_seed(0)
    m = PCLiquidCore(d_in=1, d=8, n_levels=3)
    assert m.log_precision.requires_grad
    x = torch.randn(1, 10, 1)
    y0 = m(x)
    with torch.no_grad():
        m.log_precision.add_(torch.randn(3))             # perturb precisions
    y1 = m(x)
    assert not torch.allclose(y0, y1, atol=1e-6)


def test_precision_can_be_fixed():
    m = PCLiquidCore(d_in=1, d=8, n_levels=2, learn_precision=False)
    assert not isinstance(m.log_precision, nn.Parameter)
    names = {n for n, _ in m.named_parameters()}
    assert "log_precision" not in names                  # a buffer, not trained


def test_dynamic_precision_builds_and_runs():
    m = PCLiquidCore(d_in=2, d=8, n_levels=3, dynamic_precision=True)
    assert hasattr(m, "prec_gate") and len(m.prec_gate) == 3
    y = m(torch.randn(4, 12, 2))
    assert y.shape == (4, 12, 2)
    # the gate is a real, trained pathway.
    gate_names = [n for n, _ in m.named_parameters() if "prec_gate" in n]
    assert gate_names and all(
        p.requires_grad for n, p in m.named_parameters() if "prec_gate" in n
    )


def test_dynamic_precision_zero_gate_matches_static():
    # Clean-ablation property: the precision gate is zero-initialised, so on the
    # SAME weights the dynamic path equals the static path (none->static->dynamic
    # is a strict refinement chain, each step a no-op at init).
    torch.manual_seed(0)
    m = PCLiquidCore(d_in=1, d=8, n_levels=3, dynamic_precision=True).eval()
    x = torch.randn(2, 14, 1)
    y_dyn = m(x)
    m.dynamic_precision = False                          # same weights, static path
    y_stat = m(x)
    assert torch.allclose(y_dyn, y_stat, atol=1e-6)


def test_dynamic_precision_is_input_dependent():
    # Once the gate is non-zero, precision is modulated by the state/context, so
    # the dynamics genuinely depend on the input through the precision channel.
    torch.manual_seed(0)
    m = PCLiquidCore(d_in=1, d=8, n_levels=2, dynamic_precision=True).eval()
    x = torch.randn(1, 12, 1)
    y0 = m(x)                                            # zero gate -> static-equiv
    with torch.no_grad():
        m.prec_gate[0].weight.normal_()
        m.prec_gate[0].bias.normal_()
    y1 = m(x)
    assert not torch.allclose(y0, y1, atol=1e-6)


def test_zero_prediction_error_freezes_state():
    # Predictive-coding tenet: the state is driven by errors. If we null the
    # recognition + generative pathways so every error maps to zero drive, the
    # liquid state must stay at its initial (zero) value across the whole sequence.
    m = PCLiquidCore(d_in=1, d=8, n_levels=2).eval()
    with torch.no_grad():
        for rec in m.recognize:
            rec.weight.zero_(); rec.bias.zero_()
    # With recognize == 0 and top-level td error, the drive is tanh(0 - eps_top).
    # Force the top error to zero too by zeroing the top generative map's effect:
    # easier check -- readout of a frozen-zero state stays constant in time.
    x = torch.zeros(1, 6, 1)                             # zero input -> r0 = tanh(0)=0
    with torch.no_grad():
        for gen in m.generate:
            gen.weight.zero_(); gen.bias.zero_()
    y = m(x)
    # all errors are zero (input 0, predictions 0) -> drive 0 -> state stays 0 ->
    # readout constant across time.
    assert torch.allclose(y[:, 0], y[:, -1], atol=1e-6)


# --------------------------------------------------------------------------- #
# it learns                                                                    #
# --------------------------------------------------------------------------- #
def test_overfits_a_tiny_batch():
    torch.manual_seed(0)
    x = make_signal(8, 40, d_in=1, seed=0)
    m = PCLiquidCore(d_in=1, d=48, n_levels=3)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    inp, tgt = x[:, :-1], x[:, 1:]
    first = None
    for step in range(150):
        opt.zero_grad()
        loss = nn.functional.mse_loss(m(inp), tgt)
        if first is None:
            first = loss.item()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first                     # learned the signal


def test_free_energy_aux_is_exposed():
    m = PCLiquidCore(d_in=1, d=8, n_levels=2, free_energy_weight=1.0)
    _, aux = m(torch.randn(2, 10, 1), return_aux=True)
    assert "free_energy" in aux
    assert aux["free_energy"].ndim == 0
    assert aux["free_energy"].item() >= 0.0


# --------------------------------------------------------------------------- #
# determinism / O(1) state / validation                                       #
# --------------------------------------------------------------------------- #
def test_deterministic_for_fixed_input():
    torch.manual_seed(1)
    m = PCLiquidCore(d_in=1, d=16, n_levels=3).eval()
    x = torch.randn(2, 15, 1)
    assert torch.equal(m(x), m(x))


def test_param_count_independent_of_sequence_length():
    # O(1) memory: the parameter set does not grow with the sequence length.
    m = PCLiquidCore(d_in=1, d=16, n_levels=3)
    n = count_params(m)
    m(torch.randn(1, 8, 1))
    m(torch.randn(1, 256, 1))
    assert count_params(m) == n


def test_construction_validates():
    with pytest.raises(ValueError):
        PCLiquidCore(d_in=1, n_levels=0)


# --------------------------------------------------------------------------- #
# baselines                                                                   #
# --------------------------------------------------------------------------- #
def test_baselines_build_and_run_with_comparable_params():
    models = build_models(d_in=1)
    counts = {k: count_params(m) for k, m in models.items()}
    x = torch.randn(2, 30, 1)
    for k, m in models.items():
        assert m(x).shape == (2, 30, 1)
    # all four within ~3x of each other -> a fair-ish comparison.
    lo, hi = min(counts.values()), max(counts.values())
    assert hi <= 3 * lo, f"param budgets too unequal: {counts}"


def test_two_regimes_shapes_differ_in_content():
    a, b = two_regimes(4, 50, d_in=1, seed=0)
    assert a.shape == b.shape == (4, 50, 1)
    assert not torch.allclose(a, b)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
