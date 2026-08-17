import pytest
pytest.importorskip("hypothesis")  # optional dev dep — see requirements-dev.txt

"""
tests/test_acoustic_ops_properties.py -- property-based acoustic invariants for
the composable binaural-hearing operators.

``tests/test_acoustic_ops.py`` pins these against chosen scenarios; here we pin
the *physics laws themselves* with Hypothesis, over the whole input space:

- propagation delay is non-negative, symmetric in source/receiver, recovers the
  distance (``delay * speed == |source - receiver|``), and scales inversely with
  the speed of sound;
- spherical spreading gain is positive, recovers ``ref_dist`` (``gain * r ==
  ref_dist``), and falls monotonically with distance;
- the ITD is antisymmetric under swapping the ears, is bounded by
  ``head_width / speed`` (reverse triangle inequality), and vanishes on the
  perpendicular bisector;
- the ILD is antisymmetric under swapping the ears, sign-aligned with the ITD,
  and vanishes on the bisector;
- the Doppler shift is identity for a still scene and raises the pitch of an
  approaching source / lowers a receding one;
- wavefront superposition is linear in the source amplitudes, doubles two
  co-located in-phase arrivals, and obeys the triangle inequality on magnitude;
- localization inverts the far-field plane-wave model exactly (round-trip), maps
  into ``[-pi/2, pi/2]``, is monotone in the ITD, and saturates at the poles;
- the composed ``binaural_scene`` agrees field-by-field with the cue operators.

All tensors are explicit float64; the global default dtype is left untouched so
the float32 model tests sharing this process are unaffected.

Run:  python -m pytest tests/test_acoustic_ops_properties.py -v
"""
import math
import sys

import pytest
import torch
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

sys.path.insert(0, ".")

from mt_lnn.acoustic_ops import (                                  # noqa: E402
    SPEED_OF_SOUND,
    binaural_scene,
    doppler_shift,
    interaural_level_difference,
    interaural_time_difference,
    localize_azimuth,
    propagation_delay,
    spherical_spreading_gain,
    superpose_arrivals,
)

PROP = settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

_F = dict(allow_nan=False, allow_infinity=False, width=64)
_f64 = torch.float64


def _t(rows):
    return torch.tensor(rows, dtype=_f64)


# --------------------------------------------------------------------------- #
# strategies                                                                  #
# --------------------------------------------------------------------------- #
@st.composite
def point(draw, d, lo=-10.0, hi=10.0):
    return _t(draw(st.lists(st.floats(lo, hi, **_F), min_size=d, max_size=d)))


@st.composite
def scene(draw, dims=(2, 3)):
    """A source and two distinct ears in the same dimension."""
    d = draw(st.sampled_from(list(dims)))
    src = draw(point(d))
    left = draw(point(d))
    right = draw(point(d))
    head = float((left - right).norm())
    assume(head > 1e-2)                     # ears must be distinct
    # source must be off-ear so distances are well behaved
    assume(float((src - left).norm()) > 1e-2)
    assume(float((src - right).norm()) > 1e-2)
    return src, left, right, d


_speed = st.floats(50.0, 500.0, **_F)


# --------------------------------------------------------------------------- #
# propagation primitives                                                      #
# --------------------------------------------------------------------------- #
@PROP
@given(s=scene(), speed=_speed)
def test_propagation_delay_recovers_distance_and_is_symmetric(s, speed):
    src, _, recv, _ = s
    delay = propagation_delay(src, recv, speed=speed)
    assert float(delay) >= -1e-12                                   # non-negative
    dist = (src - recv).norm()
    assert torch.allclose(delay * speed, dist, atol=1e-6, rtol=1e-6)
    # symmetric in its two arguments (distance is symmetric)
    assert torch.allclose(delay, propagation_delay(recv, src, speed=speed), atol=1e-12)


@PROP
@given(s=scene())
def test_propagation_delay_scales_inversely_with_speed(s):
    src, _, recv, _ = s
    d1 = propagation_delay(src, recv, speed=100.0)
    d2 = propagation_delay(src, recv, speed=200.0)
    assert torch.allclose(d1, 2.0 * d2, atol=1e-9, rtol=1e-6)


@PROP
@given(s=scene(), ref=st.floats(0.5, 5.0, **_F))
def test_spreading_gain_recovers_ref_dist_and_is_positive(s, ref):
    src, _, recv, _ = s
    gain = spherical_spreading_gain(src, recv, ref_dist=ref)
    assert float(gain) > 0.0
    r = (src - recv).norm()
    assert torch.allclose(gain * r, _t(ref), atol=1e-6, rtol=1e-6)


@PROP
@given(s=scene())
def test_spreading_gain_falls_monotonically_with_distance(s):
    src, _, recv, _ = s
    # push the source twice as far along its own ray from the receiver
    far = recv + 2.0 * (src - recv)
    g_near = spherical_spreading_gain(src, recv)
    g_far = spherical_spreading_gain(far, recv)
    assert float(g_far) <= float(g_near) + 1e-9


# --------------------------------------------------------------------------- #
# binaural cues                                                               #
# --------------------------------------------------------------------------- #
@PROP
@given(s=scene(), speed=_speed)
def test_itd_is_antisymmetric_and_bounded_by_head_width(s, speed):
    src, left, right, _ = s
    itd = interaural_time_difference(src, left, right, speed=speed)
    # swapping the ears negates the ITD
    swapped = interaural_time_difference(src, right, left, speed=speed)
    assert torch.allclose(itd, -swapped, atol=1e-12)
    # |dl - dr| <= |left - right|  (reverse triangle inequality)
    head_width = float((left - right).norm())
    assert abs(float(itd)) <= head_width / speed + 1e-9


@PROP
@given(d=st.sampled_from([2, 3]), a=point(3), off=st.floats(0.5, 8.0, **_F),
       depth=st.floats(-8.0, 8.0, **_F))
def test_itd_and_ild_vanish_on_the_perpendicular_bisector(d, a, off, depth):
    # ears symmetric about the origin on axis 0; a source anywhere on axis 1
    # (and 2) is equidistant -> both cues are exactly zero.
    left = torch.zeros(d, dtype=_f64); left[0] = -off
    right = torch.zeros(d, dtype=_f64); right[0] = off
    src = torch.zeros(d, dtype=_f64); src[1] = depth
    if d == 3:
        src[2] = float(a[2])
    itd = interaural_time_difference(src, left, right)
    ild = interaural_level_difference(src, left, right)
    assert torch.allclose(itd, _t(0.0), atol=1e-9)
    assert torch.allclose(ild, _t(0.0), atol=1e-6)


@PROP
@given(s=scene())
def test_ild_is_antisymmetric_and_sign_aligned_with_itd(s):
    src, left, right, _ = s
    ild = interaural_level_difference(src, left, right)
    swapped = interaural_level_difference(src, right, left)
    assert torch.allclose(ild, -swapped, atol=1e-6)
    itd = interaural_time_difference(src, left, right)
    # both are positive exactly when the source is nearer the right ear
    assert float(itd) * float(ild) >= -1e-9


# --------------------------------------------------------------------------- #
# doppler                                                                     #
# --------------------------------------------------------------------------- #
@PROP
@given(s=scene(), freq=st.floats(100.0, 2000.0, **_F))
def test_doppler_is_identity_for_a_still_scene(s, freq):
    src, _, recv, d = s
    zero = torch.zeros(d, dtype=_f64)
    f_obs = doppler_shift(src, zero, recv, zero, freq=freq)
    assert torch.allclose(f_obs, _t(freq), atol=1e-6, rtol=1e-6)


@PROP
@given(s=scene(), freq=st.floats(100.0, 2000.0, **_F), spd=st.floats(1.0, 30.0, **_F))
def test_doppler_raises_pitch_for_an_approaching_source(s, freq, spd):
    src, _, recv, _ = s
    rhat = (recv - src)
    rhat = rhat / rhat.norm()
    zero = torch.zeros_like(rhat)
    # source moving straight at the receiver -> observed frequency above rest
    src_vel = spd * rhat
    f_in = doppler_shift(src, src_vel, recv, zero, freq=freq)
    assert float(f_in) > freq
    # source moving straight away -> observed frequency below rest
    f_out = doppler_shift(src, -src_vel, recv, zero, freq=freq)
    assert float(f_out) < freq


# --------------------------------------------------------------------------- #
# wavefront superposition                                                     #
# --------------------------------------------------------------------------- #
@PROP
@given(d=st.sampled_from([2, 3]), freq=st.floats(50.0, 400.0, **_F))
def test_two_colocated_in_phase_arrivals_double_the_amplitude(d, freq):
    src = torch.zeros(d, dtype=_f64); src[0] = 3.0
    recv = torch.zeros(d, dtype=_f64)
    one = src.unsqueeze(0)                                          # (1, D)
    two = torch.stack([src, src], dim=0)                            # (2, D)
    p1 = superpose_arrivals(one, recv, freq=freq)
    p2 = superpose_arrivals(two, recv, freq=freq)
    assert abs(float(p2.abs()) - 2.0 * float(p1.abs())) < 1e-3


@PROP
@given(d=st.sampled_from([2, 3]), freq=st.floats(50.0, 400.0, **_F),
       k=st.floats(0.25, 4.0, **_F))
def test_superposition_is_linear_in_amplitudes(d, freq, k):
    srcs = torch.stack([
        _t([3.0, 1.0, -2.0][:d]),
        _t([-4.0, 2.0, 1.0][:d]),
    ], dim=0)
    recv = torch.zeros(d, dtype=_f64)
    amp = _t([1.0, 0.5])
    p1 = superpose_arrivals(srcs, recv, freq=freq, amplitudes=amp)
    pk = superpose_arrivals(srcs, recv, freq=freq, amplitudes=k * amp)
    assert torch.allclose(pk, k * p1, atol=1e-3, rtol=1e-4)


@PROP
@given(d=st.sampled_from([2, 3]), freq=st.floats(50.0, 400.0, **_F))
def test_superposition_magnitude_obeys_triangle_inequality(d, freq):
    srcs = torch.stack([
        _t([3.0, 1.0, -2.0][:d]),
        _t([-4.0, 2.0, 1.0][:d]),
        _t([0.5, -3.0, 2.0][:d]),
    ], dim=0)
    recv = torch.zeros(d, dtype=_f64)
    summed = superpose_arrivals(srcs, recv, freq=freq).abs()
    individual = sum(
        float(superpose_arrivals(srcs[i:i + 1], recv, freq=freq).abs())
        for i in range(srcs.shape[0])
    )
    assert float(summed) <= individual + 1e-3


# --------------------------------------------------------------------------- #
# localization (inverse readout)                                              #
# --------------------------------------------------------------------------- #
@PROP
@given(theta=st.floats(-math.pi / 2 + 1e-3, math.pi / 2 - 1e-3, **_F),
       head=st.floats(0.1, 0.4, **_F), speed=_speed)
def test_localize_inverts_the_far_field_model_exactly(theta, head, speed):
    itd = _t((head / speed) * math.sin(theta))
    recovered = localize_azimuth(itd, head_width=head, speed=speed)
    assert torch.allclose(recovered, _t(theta), atol=1e-6)
    assert -math.pi / 2 - 1e-6 <= float(recovered) <= math.pi / 2 + 1e-6


@PROP
@given(i1=st.floats(-2e-3, 2e-3, **_F), i2=st.floats(-2e-3, 2e-3, **_F),
       head=st.floats(0.1, 0.4, **_F))
def test_localize_is_monotone_in_the_itd_and_saturates(i1, i2, head):
    assume(i1 <= i2)
    a1 = localize_azimuth(_t(i1), head_width=head)
    a2 = localize_azimuth(_t(i2), head_width=head)
    assert float(a1) <= float(a2) + 1e-9
    # an ITD far past the physical bound saturates at the right pole
    big = localize_azimuth(_t(10.0), head_width=head)
    assert torch.allclose(big, _t(math.pi / 2), atol=1e-6)


# --------------------------------------------------------------------------- #
# composed flagship                                                           #
# --------------------------------------------------------------------------- #
@PROP
@given(off=st.floats(0.1, 0.3, **_F),
       xs=st.lists(st.floats(-8.0, 8.0, **_F), min_size=2, max_size=5),
       freq=st.floats(200.0, 1000.0, **_F))
def test_binaural_scene_agrees_field_by_field_with_the_cue_operators(off, xs, freq):
    left = _t([-off, 0.0]); right = _t([off, 0.0])
    src = torch.stack([_t([x, 4.0]) for x in xs], dim=0)           # (T, 2)
    vel = torch.zeros_like(src); vel[:, 1] = -1.0                  # closing in
    bs = binaural_scene(src, left, right, source_vel=vel, freq=freq)
    head_width = float((left - right).norm())
    assert abs(bs.head_width - head_width) < 1e-9
    assert torch.allclose(
        bs.itd, interaural_time_difference(src, left, right), atol=1e-9)
    assert torch.allclose(
        bs.ild, interaural_level_difference(src, left, right), atol=1e-6)
    assert torch.allclose(
        bs.azimuth, localize_azimuth(bs.itd, head_width=head_width), atol=1e-9)
    assert bs.doppler is not None and bs.doppler.shape == (len(xs),)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
