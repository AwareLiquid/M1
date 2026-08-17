"""
tests/test_astrocyte.py — behavioural contract for the slow astrocytic gate
(mt_lnn/astrocyte.py, 2026-06-15).

AstrocyteGate is a tiny stateful slow integrator that models the tripartite
synapse (Araque et al. 1999): it integrates per-step neuronal *activity* through
a slow leaky calcium variable (time constant tau >> 1 step) and reads out a
bounded, band-shaped (inverted-U) consolidation multiplier (De Pittà et al.
2016). You multiply the Hebbian alpha by that gate.

We pin:
  * the calcium variable is SLOW: one high step barely moves the gate, but
    sustained activity over many steps drives it across the band;
  * the gate is bounded in [gate_min, gate_max] and is an inverted-U over
    calcium (quiescent and saturated -> ~gate_min; productive mid -> ~gate_max);
  * duck-typed modulate(reg) scales HebbianRegularizer.base_lr by the gate and
    does NOT compound across repeated calls (captures the original base);
  * modulate on an object without base_lr is a safe no-op (returns None);
  * activity_of() is a duck-typed mean-abs proxy over tensors;
  * reset() restores calcium/steps; construction validates; not an nn.Module.

Run:  python -m pytest tests/test_astrocyte.py -v
"""
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, ".")

from mt_lnn import AstrocyteGate, AstrocyteState, HebbianRegularizer   # noqa: E402


# --------------------------------------------------------------------------- #
# slow timescale                                                              #
# --------------------------------------------------------------------------- #
def test_calcium_is_slow_single_step_barely_moves():
    # tau=50 steps: one unit-activity step moves calcium by only ~1/50.
    g = AstrocyteGate(tau=50.0, ca_peak=1.0, width=0.5)
    assert g.calcium == 0.0
    g.step(1.0)
    assert g.calcium == pytest.approx(1.0 / 50.0, rel=1e-6)
    assert g.n_steps == 1


def test_sustained_activity_drives_calcium_across_band():
    # Sustained unit activity should pull calcium up toward 1.0 (the input level)
    # over many steps -- the slow integrator approaches its input asymptotically.
    g = AstrocyteGate(tau=10.0, ca_peak=1.0, width=0.5)
    for _ in range(200):
        g.step(1.0)
    assert g.calcium == pytest.approx(1.0, abs=1e-3)
    # And at ca == ca_peak the gate is essentially gate_max.
    assert g.gate == pytest.approx(g.gate_max, rel=1e-2)


def test_one_step_vs_sustained_gate_difference():
    # The whole point: a brief spike and a sustained drive of equal amplitude
    # produce very different gates because the gate integrates slowly.
    brief = AstrocyteGate(tau=30.0, ca_peak=1.0, width=0.4)
    brief.step(1.0)

    sustained = AstrocyteGate(tau=30.0, ca_peak=1.0, width=0.4)
    for _ in range(150):
        sustained.step(1.0)

    # sustained calcium is far closer to the productive peak -> higher gate.
    assert sustained.gate > brief.gate
    assert sustained.calcium > 10.0 * brief.calcium


# --------------------------------------------------------------------------- #
# inverted-U band                                                             #
# --------------------------------------------------------------------------- #
def test_gate_bounded_and_inverted_u():
    g = AstrocyteGate(ca_peak=1.0, width=0.5, gate_min=0.25, gate_max=1.5)
    # At the peak -> gate_max; far below and far above -> ~gate_min.
    at_peak = g._gate_of(1.0)
    quiescent = g._gate_of(0.0)
    saturated = g._gate_of(5.0)
    assert at_peak == pytest.approx(1.5, rel=1e-6)
    assert quiescent < at_peak
    assert saturated < at_peak
    # Bounded for any calcium.
    for ca in [-3.0, 0.0, 0.5, 1.0, 2.0, 10.0]:
        gate = g._gate_of(ca)
        assert 0.25 - 1e-9 <= gate <= 1.5 + 1e-9


def test_saturation_depresses_like_quiescence():
    # Homeostatic: an over-driven (saturated) network gets ~as little glial
    # potentiation as a silent one -- both fall to the floor.
    g = AstrocyteGate(ca_peak=1.0, width=0.4, gate_min=0.25, gate_max=1.5)
    assert g._gate_of(10.0) == pytest.approx(g.gate_min, abs=1e-3)


# --------------------------------------------------------------------------- #
# duck-typed modulate over HebbianRegularizer                                 #
# --------------------------------------------------------------------------- #
def test_modulate_scales_base_lr_by_gate():
    reg = HebbianRegularizer(base_lr=1e-3)
    g = AstrocyteGate(tau=5.0, ca_peak=1.0, width=0.5)
    for _ in range(100):
        g.step(1.0)                    # drive calcium to the productive peak
    new_lr = g.modulate(reg)
    assert new_lr == pytest.approx(1e-3 * g.gate, rel=1e-6)
    assert reg.base_lr == pytest.approx(new_lr, rel=1e-6)
    assert g.gate > 1.0                # peak band potentiates


def test_modulate_does_not_compound():
    # Repeated modulate() calls multiply the ORIGINAL base by the gate, never the
    # previously-gated value -- otherwise base_lr would decay/explode each call.
    reg = HebbianRegularizer(base_lr=2e-3)
    g = AstrocyteGate(ca_peak=1.0, width=0.5)
    g.step(0.5)
    lr1 = g.modulate(reg)
    lr2 = g.modulate(reg)              # same gate, same calcium -> same lr
    assert lr1 == pytest.approx(lr2, rel=1e-9)
    assert reg.base_lr == pytest.approx(2e-3 * g.gate, rel=1e-6)


def test_modulate_without_base_lr_is_safe_noop():
    class Bare:
        pass

    g = AstrocyteGate()
    assert g.modulate(Bare()) is None


def test_modulate_explicit_base_overrides_capture():
    reg = HebbianRegularizer(base_lr=1e-3)
    g = AstrocyteGate(ca_peak=1.0, width=0.5)
    g.step(1.0)
    new_lr = g.modulate(reg, base_lr=4e-3)
    assert new_lr == pytest.approx(4e-3 * g.gate, rel=1e-6)


# --------------------------------------------------------------------------- #
# activity helper / housekeeping                                              #
# --------------------------------------------------------------------------- #
def test_activity_of_is_mean_abs_over_tensors():
    a = torch.full((4,), -2.0)
    b = torch.full((4,), 4.0)
    # mean|a| = 2, mean|b| = 4 -> mean of those = 3
    assert AstrocyteGate.activity_of(a, b) == pytest.approx(3.0, rel=1e-6)
    # plain scalars work too.
    assert AstrocyteGate.activity_of(-5.0) == pytest.approx(5.0, rel=1e-6)
    with pytest.raises(ValueError):
        AstrocyteGate.activity_of()


def test_run_and_state_snapshot():
    g = AstrocyteGate(tau=8.0, ca_peak=1.0, width=0.5)
    gates = g.run([1.0, 1.0, 1.0])
    assert len(gates) == 3
    st = g.state
    assert isinstance(st, AstrocyteState)
    assert st.n_steps == 3
    assert st.gate == pytest.approx(gates[-1], rel=1e-9)
    assert g.consolidation_scale() == pytest.approx(g.gate, rel=1e-9)


def test_reset_restores_initial():
    g = AstrocyteGate(tau=5.0)
    for _ in range(20):
        g.step(1.0)
    assert g.calcium > 0.0
    g.reset()
    assert g.calcium == 0.0
    assert g.n_steps == 0


def test_rejects_non_finite_activity():
    g = AstrocyteGate()
    with pytest.raises(ValueError):
        g.step(float("nan"))


def test_construction_validates_and_not_nn_module():
    g = AstrocyteGate()
    assert not isinstance(g, nn.Module)
    with pytest.raises(ValueError):
        AstrocyteGate(tau=0.0)
    with pytest.raises(ValueError):
        AstrocyteGate(width=0.0)
    with pytest.raises(ValueError):
        AstrocyteGate(gate_min=-0.1)
    with pytest.raises(ValueError):
        AstrocyteGate(gate_min=2.0, gate_max=1.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
