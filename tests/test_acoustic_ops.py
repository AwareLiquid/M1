"""
tests/test_acoustic_ops.py -- composable acoustic / binaural-hearing operators.

Every operator is pinned against a closed-form acoustic ground truth: time of
flight = distance / c, 1/r spherical spreading, ITD/ILD sign and magnitude,
the standard Doppler formula, phasor interference (constructive / destructive),
and the far-field round-trip ITD -> azimuth -> ITD. The flagship
``binaural_scene`` composition is checked as a fly-by: the localized azimuth
sweeps left->right through zero while the Doppler falls through the rest
frequency at closest approach -- "where is it, and is it coming or going?",
computed from raw geometry.
"""
import math
import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mt_lnn.acoustic_ops import (  # noqa: E402
    SPEED_OF_SOUND,
    BinauralScene,
    propagation_delay,
    spherical_spreading_gain,
    interaural_time_difference,
    interaural_level_difference,
    doppler_shift,
    superpose_arrivals,
    localize_azimuth,
    binaural_scene,
)

C = SPEED_OF_SOUND


def _t(*xs):
    return torch.tensor(xs, dtype=torch.float32)


# --- propagation primitives ------------------------------------------------


def test_propagation_delay_is_distance_over_c():
    src = _t(C, 0.0)                                      # exactly one second away
    rcv = _t(0.0, 0.0)
    assert propagation_delay(src, rcv) == pytest.approx(1.0, abs=1e-5)
    assert propagation_delay(src, rcv, speed=C / 2) == pytest.approx(2.0, abs=1e-5)


def test_spreading_gain_is_inverse_distance():
    src = _t(2.0, 0.0)
    rcv = _t(0.0, 0.0)
    assert spherical_spreading_gain(src, rcv) == pytest.approx(0.5, abs=1e-6)       # 1/2
    assert spherical_spreading_gain(src, rcv, ref_dist=2.0) == pytest.approx(1.0, abs=1e-6)


def test_delay_broadcasts_over_a_trajectory():
    traj = torch.stack([_t(x, 0.0) for x in (1.0, 2.0, 3.0)])     # (3, 2)
    rcv = _t(0.0, 0.0)
    d = propagation_delay(traj, rcv)
    assert d.shape == (3,)
    assert torch.allclose(d, torch.tensor([1.0, 2.0, 3.0]) / C, atol=1e-6)


# --- binaural cues: ITD / ILD ----------------------------------------------


LEFT = _t(-0.1, 0.0)
RIGHT = _t(0.1, 0.0)


def test_itd_zero_straight_ahead():
    ahead = _t(0.0, 5.0)                                  # equidistant from both ears
    assert interaural_time_difference(ahead, LEFT, RIGHT).abs().item() < 1e-7


def test_itd_positive_to_the_right():
    src = _t(5.0, 0.0)                                    # on the right ear's side
    itd = interaural_time_difference(src, LEFT, RIGHT)
    assert itd.item() > 0.0
    # exact: (|src-L| - |src-R|)/c
    dl = (src - LEFT).norm().item()
    dr = (src - RIGHT).norm().item()
    assert itd.item() == pytest.approx((dl - dr) / C, abs=1e-9)


def test_itd_antisymmetric_left_right():
    rsrc = _t(5.0, 1.0)
    lsrc = _t(-5.0, 1.0)
    assert interaural_time_difference(rsrc, LEFT, RIGHT).item() == \
        pytest.approx(-interaural_time_difference(lsrc, LEFT, RIGHT).item(), abs=1e-9)


def test_ild_positive_to_the_right_and_matches_gains():
    src = _t(5.0, 0.0)
    ild = interaural_level_difference(src, LEFT, RIGHT)
    assert ild.item() > 0.0                              # right ear louder
    dl = (src - LEFT).norm().item()
    dr = (src - RIGHT).norm().item()
    assert ild.item() == pytest.approx(20.0 * math.log10(dl / dr), abs=1e-4)


def test_ild_zero_straight_ahead():
    ahead = _t(0.0, 5.0)
    assert interaural_level_difference(ahead, LEFT, RIGHT).abs().item() < 1e-5


# --- Doppler ---------------------------------------------------------------


def test_doppler_approaching_raises_pitch():
    rcv = _t(0.0, 0.0)
    src = _t(10.0, 0.0)
    vel = _t(-5.0, 0.0)                                   # moving toward the receiver
    f = doppler_shift(src, vel, rcv, torch.zeros(2), freq=100.0)
    assert f.item() == pytest.approx(100.0 * C / (C - 5.0), abs=1e-3)
    assert f.item() > 100.0


def test_doppler_receding_lowers_pitch():
    rcv = _t(0.0, 0.0)
    src = _t(10.0, 0.0)
    vel = _t(5.0, 0.0)                                    # moving away
    f = doppler_shift(src, vel, rcv, torch.zeros(2), freq=100.0)
    assert f.item() == pytest.approx(100.0 * C / (C + 5.0), abs=1e-3)
    assert f.item() < 100.0


def test_doppler_tangential_motion_no_shift():
    rcv = _t(0.0, 0.0)
    src = _t(0.0, 10.0)
    vel = _t(3.0, 0.0)                                    # perpendicular to line of sight
    f = doppler_shift(src, vel, rcv, torch.zeros(2), freq=100.0)
    assert f.item() == pytest.approx(100.0, abs=1e-3)


def test_doppler_moving_receiver_toward_source_raises_pitch():
    src = _t(10.0, 0.0)
    rcv = _t(0.0, 0.0)
    rvel = _t(4.0, 0.0)                                   # receiver chases the source
    f = doppler_shift(src, torch.zeros(2), rcv, rvel, freq=100.0)
    # rhat points src->rcv = (-1,0); v_recv.rhat = -4 -> numerator c+4
    assert f.item() == pytest.approx(100.0 * (C + 4.0) / C, abs=1e-3)
    assert f.item() > 100.0


# --- superposition / interference ------------------------------------------


def test_two_equal_inphase_arrivals_double_amplitude():
    rcv = _t(0.0, 0.0)
    sources = torch.stack([_t(1.0, 0.0), _t(-1.0, 0.0)])  # both distance 1
    p = superpose_arrivals(sources, rcv, freq=440.0)
    assert p.abs().item() == pytest.approx(2.0, abs=1e-4)  # gain 1 each, in phase


def test_half_wavelength_path_difference_cancels():
    # f = C makes wavelength exactly 1 m; a 0.5 m extra path is half a wavelength.
    rcv = _t(0.0, 0.0)
    sources = torch.stack([_t(1.0, 0.0), _t(1.5, 0.0)])   # path diff 0.5 m = lambda/2
    p = superpose_arrivals(sources, rcv, freq=C)
    # opposite phase -> magnitude is the gain *difference* |1 - 1/1.5|
    assert p.abs().item() == pytest.approx(abs(1.0 - 1.0 / 1.5), abs=1e-4)


def test_full_wavelength_path_difference_adds():
    rcv = _t(0.0, 0.0)
    sources = torch.stack([_t(1.0, 0.0), _t(2.0, 0.0)])   # path diff 1.0 m = lambda
    p = superpose_arrivals(sources, rcv, freq=C)
    assert p.abs().item() == pytest.approx(1.0 + 1.0 / 2.0, abs=1e-4)  # in phase, gains add


def test_superpose_amplitude_weights():
    rcv = _t(0.0, 0.0)
    sources = torch.stack([_t(1.0, 0.0), _t(-1.0, 0.0)])
    amps = _t(2.0, 0.0)                                   # silence the second source
    p = superpose_arrivals(sources, rcv, freq=440.0, amplitudes=amps)
    assert p.abs().item() == pytest.approx(2.0, abs=1e-4)


# --- localization (inverse readout) ----------------------------------------


@pytest.mark.parametrize("deg", [-60.0, -30.0, -5.0, 0.0, 5.0, 30.0, 60.0])
def test_localize_round_trip_far_field(deg):
    theta = math.radians(deg)
    R = 5000.0                                            # far field
    src = _t(R * math.sin(theta), R * math.cos(theta))
    head_width = float((RIGHT - LEFT).norm().item())     # 0.2
    itd = interaural_time_difference(src, LEFT, RIGHT)
    est = localize_azimuth(itd, head_width=head_width)
    assert est.item() == pytest.approx(theta, abs=2e-3)


def test_localize_zero_itd_is_straight_ahead():
    assert localize_azimuth(torch.tensor(0.0), head_width=0.2).item() == pytest.approx(0.0)


def test_localize_saturates_at_poles():
    huge = torch.tensor(10.0)                             # physically impossible ITD
    assert localize_azimuth(huge, head_width=0.2).item() == pytest.approx(math.pi / 2, abs=1e-6)
    assert localize_azimuth(-huge, head_width=0.2).item() == pytest.approx(-math.pi / 2, abs=1e-6)


# --- flagship: binaural_scene fly-by ---------------------------------------


def _flyby(T=21):
    xs = torch.linspace(-10.0, 10.0, T)
    source = torch.stack([xs, torch.full_like(xs, 2.0)], dim=1)   # cross in front, y=2
    vel = torch.zeros_like(source)
    vel[:, 0] = 1.0                                       # constant +x motion
    return source, vel


def test_scene_azimuth_sweeps_left_to_right_through_zero():
    source, _ = _flyby()
    sc = binaural_scene(source, LEFT, RIGHT)
    assert isinstance(sc, BinauralScene)
    az = sc.azimuth
    assert az[0].item() < 0.0                             # starts on the left
    assert az[-1].item() > 0.0                            # ends on the right
    # monotonic non-decreasing as it crosses in front
    assert torch.all(az[1:] - az[:-1] >= -1e-6)
    mid = az[az.shape[0] // 2].item()
    assert abs(mid) < 1e-3                                # ~0 at closest approach


def test_scene_doppler_falls_through_rest_freq():
    source, vel = _flyby()
    sc = binaural_scene(source, LEFT, RIGHT, source_vel=vel, freq=1000.0)
    assert sc.doppler is not None
    assert sc.doppler[0].item() > 1000.0                 # approaching -> higher
    assert sc.doppler[-1].item() < 1000.0                # receding -> lower
    # monotonically decreasing pitch through the fly-by
    assert torch.all(sc.doppler[1:] - sc.doppler[:-1] <= 1e-4)


def test_scene_without_velocity_has_no_doppler():
    source, _ = _flyby()
    sc = binaural_scene(source, LEFT, RIGHT)
    assert sc.doppler is None
    assert sc.head_width == pytest.approx(0.2, abs=1e-6)


def test_scene_ild_sign_tracks_side():
    source, _ = _flyby()
    sc = binaural_scene(source, LEFT, RIGHT)
    assert sc.ild[0].item() < 0.0                         # left -> right ear quieter
    assert sc.ild[-1].item() > 0.0


# --- differentiability + validation ----------------------------------------


def test_operators_are_differentiable():
    src = torch.tensor([3.0, 4.0], requires_grad=True)
    rcv = _t(0.0, 0.0)
    d = propagation_delay(src, rcv)
    d.backward()
    assert src.grad is not None and torch.isfinite(src.grad).all()


def test_no_trainable_parameters_module_is_functional():
    import mt_lnn.acoustic_ops as ao
    import torch.nn as nn
    # the module exposes plain functions, not nn.Modules with parameters
    for name in ao.__all__:
        obj = getattr(ao, name)
        assert not isinstance(obj, nn.Module)


@pytest.mark.parametrize("call", [
    lambda: propagation_delay(_t(1.0, 0.0), _t(0.0, 0.0), speed=0.0),
    lambda: spherical_spreading_gain(_t(1.0, 0.0), _t(0.0, 0.0), ref_dist=0.0),
    lambda: doppler_shift(_t(1.0, 0.0), torch.zeros(2), _t(0.0, 0.0), torch.zeros(2), freq=0.0),
    lambda: localize_azimuth(torch.tensor(0.0), head_width=0.0),
    lambda: superpose_arrivals(torch.zeros(1, 2), _t(0.0, 0.0), freq=-1.0),
])
def test_validates_config(call):
    with pytest.raises(ValueError):
        call()
