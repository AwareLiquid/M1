"""
tests/test_neuromodulation.py — behavioural contract for the multi-neuromodulator
global regulator (mt_lnn/neuromodulation.py, 2026-06-15).

NeuromodulationController is the "联调中枢" (joint-tuning hub): a zero-parameter,
model-decoupled controller that turns four observable signals
(reward / surprise / risk / arousal) into four global neuromodulatory scalars
(dopamine / acetylcholine / serotonin / norepinephrine) and orchestrates the
EXISTING MT-LNN modules those scalars should retune:

    DA  → HebbianRegularizer.base_lr        (plasticity rate)
    ACh → GWTBLayer dynamic-bandwidth gate   (workspace ignition)
    5HT → liquid time-constant τ             (integration patience)
    NE  → response gain                      (a scalar multiplier)

We pin the properties that matter for a default-off, zero-regression feature:
  • neutral baseline: with no/neutral signals every channel sits at 0.5 → all
    modulation read-outs are identity (×1.0, +0.0) → bit-identical to un-modulated;
  • phasic response: a positive surprise/reward departure pushes its channel > 0.5;
  • the read-outs map into the documented target-space ranges and directions;
  • the ACh→GWT actuator is REAL: it changes the gate bias and thus the output,
    and it is a strict no-op on a layer without the hook (duck-typed decoupling);
  • the DA→Hebbian actuator scales base_lr in the right direction;
  • defensive construction validation;
  • zero trainable parameters / zero model coupling (it is not an nn.Module).

Run:  python -m pytest tests/test_neuromodulation.py -v
"""
import sys

import pytest
import torch

sys.path.insert(0, ".")

from mt_lnn import MTLNNConfig, NeuromodulationController, NeuromodulatorState  # noqa: E402
from mt_lnn import HebbianRegularizer                                            # noqa: E402
from mt_lnn.gwtb import GWTBLayer                                                # noqa: E402


def _config(**overrides):
    base = dict(
        d_model=128,
        n_heads=4,
        d_head=32,
        gwtb_compression_ratio=8,   # d_gw = 16
        gwtb_n_heads=4,
        max_seq_len=64,
        dropout=0.0,
    )
    base.update(overrides)
    return MTLNNConfig(**base)


def _input(B=2, T=8, d_model=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, d_model, generator=g)


# --------------------------------------------------------------------------- #
# neutral baseline → every read-out is identity (zero regression)              #
# --------------------------------------------------------------------------- #
def test_neutral_state_is_identity():
    nm = NeuromodulationController()
    # No update yet → all channels at the 0.5 neutral baseline.
    st = nm.state
    assert (st.dopamine, st.acetylcholine, st.serotonin, st.norepinephrine) == (0.5, 0.5, 0.5, 0.5)
    # Read-outs at neutral are exact identities.
    assert nm.bandwidth_bias_offset() == pytest.approx(0.0, abs=1e-12)
    assert nm.plasticity_scale() == pytest.approx(1.0, abs=1e-12)
    assert nm.time_constant_scale() == pytest.approx(1.0, abs=1e-12)
    assert nm.gain_scale() == pytest.approx(1.0, abs=1e-12)


def test_first_update_holds_baseline():
    # The first sample of each channel only seeds the EMA (z-score 0) → stays 0.5.
    nm = NeuromodulationController()
    st = nm.update(reward=1.0, surprise=1.0, risk=1.0, arousal=1.0)
    assert st.dopamine == pytest.approx(0.5, abs=1e-9)
    assert st.acetylcholine == pytest.approx(0.5, abs=1e-9)
    assert st.serotonin == pytest.approx(0.5, abs=1e-9)
    assert st.norepinephrine == pytest.approx(0.5, abs=1e-9)


# --------------------------------------------------------------------------- #
# phasic response: a rise above baseline pushes the channel above 0.5          #
# --------------------------------------------------------------------------- #
def test_surprise_spike_raises_acetylcholine():
    nm = NeuromodulationController()
    nm.update(surprise=0.1)          # seed baseline low
    nm.update(surprise=0.1)
    st = nm.update(surprise=5.0)     # sharp upward departure
    assert st.acetylcholine > 0.5
    # And it widens the workspace (positive bandwidth offset).
    assert nm.bandwidth_bias_offset() > 0.0


def test_reward_prediction_error_raises_dopamine():
    nm = NeuromodulationController()
    nm.update(reward=0.0)
    nm.update(reward=0.0)
    st = nm.update(reward=3.0)       # better than expected
    assert st.dopamine > 0.5
    assert nm.plasticity_scale() > 1.0


def test_channels_are_independent():
    # Driving only surprise must not move the other three channels off neutral.
    nm = NeuromodulationController()
    nm.update(surprise=0.1)
    st = nm.update(surprise=5.0)
    assert st.acetylcholine > 0.5
    assert st.dopamine == 0.5
    assert st.serotonin == 0.5
    assert st.norepinephrine == 0.5


# --------------------------------------------------------------------------- #
# read-outs map into the documented target ranges / directions                #
# --------------------------------------------------------------------------- #
def test_readout_ranges_and_directions():
    nm = NeuromodulationController(bandwidth_span=4.0, plasticity_span=1.0,
                                   tau_span=1.0, gain_span=0.5)
    # Force extreme states directly to check the linear maps at the rails.
    nm._state = NeuromodulatorState(1.0, 1.0, 1.0, 1.0)
    assert nm.bandwidth_bias_offset() == pytest.approx(4.0)     # +span
    assert nm.plasticity_scale() == pytest.approx(2.0)          # 1+span
    assert nm.time_constant_scale() == pytest.approx(2.0)       # 1+span
    assert nm.gain_scale() == pytest.approx(1.5)                # 1+span

    nm._state = NeuromodulatorState(0.0, 0.0, 0.0, 0.0)
    assert nm.bandwidth_bias_offset() == pytest.approx(-4.0)    # -span
    assert nm.plasticity_scale() == pytest.approx(0.0)          # 1-span
    # 5-HT only lengthens τ: below-baseline serotonin never shortens it.
    assert nm.time_constant_scale() == pytest.approx(1.0)
    assert nm.gain_scale() == pytest.approx(0.5)                # 1-span


# --------------------------------------------------------------------------- #
# ACh → GWT actuator is REAL and changes the output                            #
# --------------------------------------------------------------------------- #
def test_modulate_gwtb_changes_output():
    torch.manual_seed(0)
    layer = GWTBLayer(_config(gwtb_dynamic_bandwidth=True, gwtb_bandwidth_gate_bias=0.0))
    # Give the gate a non-trivial weight so channels differ; eval for determinism.
    with torch.no_grad():
        layer.bandwidth_gate_weight.add_(torch.randn_like(layer.bandwidth_gate_weight))
    layer.eval()
    x = _input()

    with torch.no_grad():
        out_base, _ = layer(x)

    # Drive the controller to a high-ACh state, apply it, and re-run.
    nm = NeuromodulationController()
    nm._state = NeuromodulatorState(0.5, 1.0, 0.5, 0.5)   # ACh = 1 → wide
    offset = nm.modulate_gwtb(layer)
    assert offset == pytest.approx(4.0)
    assert layer._bandwidth_bias_offset.item() == pytest.approx(4.0)

    with torch.no_grad():
        out_wide, _ = layer(x)
    # A non-zero bias offset shifts the gate → the broadcast changes.
    assert not torch.allclose(out_base, out_wide)
    # Wider workspace → higher mean gate value than the un-modulated bias=0 case.
    assert layer.last_bandwidth_gate_mean.item() > 0.5


def test_modulate_gwtb_neutral_is_noop():
    torch.manual_seed(0)
    layer = GWTBLayer(_config(gwtb_dynamic_bandwidth=True, gwtb_bandwidth_gate_bias=0.0))
    with torch.no_grad():
        layer.bandwidth_gate_weight.add_(torch.randn_like(layer.bandwidth_gate_weight))
    layer.eval()
    x = _input()
    with torch.no_grad():
        out_ref, _ = layer(x)

    nm = NeuromodulationController()          # neutral ACh = 0.5 → offset 0
    nm.modulate_gwtb(layer)
    assert layer._bandwidth_bias_offset.item() == pytest.approx(0.0)
    with torch.no_grad():
        out_same, _ = layer(x)
    assert torch.equal(out_ref, out_same)     # bit-identical


def test_modulate_gwtb_on_layer_without_hook_is_safe():
    # Duck-typed decoupling: a target lacking the hook is a clean no-op.
    nm = NeuromodulationController()
    nm._state = NeuromodulatorState(0.5, 1.0, 0.5, 0.5)

    class _Bare:
        pass

    assert nm.modulate_gwtb(_Bare()) == pytest.approx(4.0)   # returns offset, no crash

    # And a dynamic-bandwidth-DISABLED layer ignores the push (hook is a no-op).
    off = GWTBLayer(_config(gwtb_dynamic_bandwidth=False))
    nm.modulate_gwtb(off)
    assert not hasattr(off, "_bandwidth_bias_offset")


# --------------------------------------------------------------------------- #
# DA → Hebbian actuator scales base_lr in the right direction                  #
# --------------------------------------------------------------------------- #
def test_modulate_hebbian_scales_lr():
    nm = NeuromodulationController(plasticity_span=1.0)
    reg = HebbianRegularizer(base_lr=1e-4)

    nm._state = NeuromodulatorState(1.0, 0.5, 0.5, 0.5)   # max DA → ×2
    new_lr = nm.modulate_hebbian(reg)
    assert new_lr == pytest.approx(2e-4)
    assert reg.base_lr == pytest.approx(2e-4)


def test_modulate_hebbian_without_attr_returns_none():
    nm = NeuromodulationController()

    class _Bare:
        pass

    assert nm.modulate_hebbian(_Bare()) is None


# --------------------------------------------------------------------------- #
# housekeeping: reset, run-stream, zero coupling                               #
# --------------------------------------------------------------------------- #
def test_reset_returns_to_neutral():
    nm = NeuromodulationController()
    nm.update(surprise=0.1)
    nm.update(surprise=5.0)
    nm.reset()
    st = nm.state
    assert (st.dopamine, st.acetylcholine, st.serotonin, st.norepinephrine) == (0.5, 0.5, 0.5, 0.5)
    assert nm.n_updates == 0


def test_run_stream_is_deterministic():
    stream = [{"surprise": 0.1}, {"surprise": 0.1}, {"surprise": 5.0}]
    a = NeuromodulationController().run(stream)
    b = NeuromodulationController().run(stream)
    assert [s.acetylcholine for s in a] == [s.acetylcholine for s in b]
    assert a[-1].acetylcholine > 0.5


def test_controller_is_not_an_nn_module():
    # Zero trainable parameters / zero torch coupling: a slow neuromodulatory bus.
    import torch.nn as nn
    nm = NeuromodulationController()
    assert not isinstance(nm, nn.Module)


# --------------------------------------------------------------------------- #
# defensive construction validation                                           #
# --------------------------------------------------------------------------- #
def test_invalid_construction_raises():
    with pytest.raises(ValueError):
        NeuromodulationController(ema_decay=0.0)
    with pytest.raises(ValueError):
        NeuromodulationController(ema_decay=1.0)
    with pytest.raises(ValueError):
        NeuromodulationController(sensitivity=0.0)
    with pytest.raises(ValueError):
        NeuromodulationController(bandwidth_span=-1.0)
    with pytest.raises(ValueError):
        NeuromodulationController(plasticity_span=2.0)   # must be in [0,1]
    with pytest.raises(ValueError):
        NeuromodulationController(gain_span=1.5)          # must be in [0,1]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
