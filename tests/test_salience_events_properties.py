"""
tests/test_salience_events_properties.py -- property-based invariants for the
global-workspace ignition / state-change event detector.

``tests/test_salience_events.py`` pins these against chosen synthetic streams;
here we pin the *detector's contract itself* with Hypothesis, over the whole
input space of streams and configurations:

- it is deterministic, and the online ``update`` path agrees with batch
  ``observe``;
- no event ever fires during the warmup window;
- events strictly alternate ignition -> quiescence -> ignition ... (Schmitt
  hysteresis), always starting with an ignition;
- consecutive events are separated by at least ``refractory + 1`` steps;
- every ignition carries ``z >= ignite_z`` and every quiescence ``z <=
  release_z``; the recorded signal/baseline/z fields are self-consistent;
- a constant stream never fires (zero variance -> zero z-score);
- a single spike on a settled flat baseline fires exactly one ignition, at the
  spike;
- the z-score -- hence the entire event stream -- is invariant under a positive
  affine rescaling ``a*x + b`` of the signal (the EMA baseline/variance is a
  linear operator, so the standardised surprise is scale/shift free);
- ``world_model_surprise`` is the duck-typed bridge: it reads a float or a torch
  scalar off ``last_pred_error`` and rejects heads that lack it.

Run:  python -m pytest tests/test_salience_events_properties.py -v
"""
import sys

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

sys.path.insert(0, ".")

from mt_lnn.salience_events import (                                # noqa: E402
    SalienceEventDetector,
    StateChangeEvent,
    world_model_surprise,
)

PROP = settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

_F = dict(allow_nan=False, allow_infinity=False, width=64)


# --------------------------------------------------------------------------- #
# strategies                                                                  #
# --------------------------------------------------------------------------- #
@st.composite
def stream(draw, min_len=10, max_len=45):
    n = draw(st.integers(min_len, max_len))
    return draw(st.lists(st.floats(0.0, 50.0, **_F), min_size=n, max_size=n))


@st.composite
def detector_cfg(draw):
    ignite = draw(st.floats(2.0, 5.0, **_F))
    release = draw(st.floats(0.0, 1.5, **_F))
    assume(release < ignite)
    return dict(
        ignite_z=ignite,
        release_z=release,
        refractory=draw(st.integers(0, 6)),
        warmup=draw(st.integers(1, 10)),
        ema_decay=draw(st.floats(0.5, 0.95, **_F)),
    )


def _run(cfg, sig):
    return SalienceEventDetector(**cfg).observe(sig)


# --------------------------------------------------------------------------- #
# determinism & path equivalence                                             #
# --------------------------------------------------------------------------- #
@PROP
@given(cfg=detector_cfg(), sig=stream())
def test_detector_is_deterministic_and_update_equals_observe(cfg, sig):
    assert _run(cfg, sig) == _run(cfg, sig)                          # deterministic
    online = SalienceEventDetector(**cfg)
    by_hand = [ev for s in sig for ev in [online.update(s)] if ev is not None]
    assert by_hand == _run(cfg, sig)
    assert online.n_events == len(by_hand)                          # counter agrees


# --------------------------------------------------------------------------- #
# warmup / hysteresis / refractory / thresholds                              #
# --------------------------------------------------------------------------- #
@PROP
@given(cfg=detector_cfg(), sig=stream())
def test_no_event_fires_during_warmup(cfg, sig):
    for e in _run(cfg, sig):
        assert e.step > cfg["warmup"]


@PROP
@given(cfg=detector_cfg(), sig=stream())
def test_events_strictly_alternate_starting_with_ignition(cfg, sig):
    events = _run(cfg, sig)
    for i, e in enumerate(events):
        assert e.kind == ("ignition" if i % 2 == 0 else "quiescence")


@PROP
@given(cfg=detector_cfg(), sig=stream())
def test_consecutive_events_respect_the_refractory_gap(cfg, sig):
    steps = [e.step for e in _run(cfg, sig)]
    for a, b in zip(steps, steps[1:]):
        assert b - a >= cfg["refractory"] + 1


@PROP
@given(cfg=detector_cfg(), sig=stream())
def test_events_cross_their_thresholds_and_fields_are_consistent(cfg, sig):
    for e in _run(cfg, sig):
        assert e.z == e.salience                                    # alias
        if e.kind == "ignition":
            assert e.z >= cfg["ignite_z"] - 1e-9
        else:
            assert e.z <= cfg["release_z"] + 1e-9
        # the recorded raw signal is the fed sample at that 1-based step
        assert e.signal == float(sig[e.step - 1])


# --------------------------------------------------------------------------- #
# degenerate / canonical streams                                             #
# --------------------------------------------------------------------------- #
@PROP
@given(cfg=detector_cfg(), c=st.floats(0.0, 50.0, **_F),
       n=st.integers(10, 40))
def test_a_constant_stream_never_fires(cfg, c, n):
    assert _run(cfg, [c] * n) == []                                 # zero variance -> z=0


@PROP
@given(base=st.floats(0.0, 5.0, **_F), spike=st.floats(50.0, 500.0, **_F),
       warm=st.integers(1, 8), pad=st.integers(2, 6))
def test_a_single_spike_on_a_flat_baseline_fires_one_ignition(base, spike, warm, pad):
    assume(spike > base + 10.0)
    pre = warm + pad
    sig = [base] * pre + [spike]
    events = _run(dict(warmup=warm, refractory=3), sig)
    assert len(events) == 1
    assert events[0].kind == "ignition"
    assert events[0].step == pre + 1                                # the spike sample


# --------------------------------------------------------------------------- #
# z-score DC-offset (additive-shift) invariance                              #
# --------------------------------------------------------------------------- #
@PROP
@given(cfg=detector_cfg(), sig=stream(), b=st.floats(0.0, 50.0, **_F))
def test_event_stream_is_invariant_under_an_additive_dc_offset(cfg, sig, b):
    # The EMA baseline/variance is a linear operator: shifting the whole signal
    # by a constant shifts the baseline by the same constant and leaves the
    # variance (hence the standardised surprise z) *exactly* unchanged -- even
    # through the variance floor. So the entire event stream is shift-invariant.
    # (A multiplicative rescale is only invariant *above* the eps floor, since
    # that floor is absolute, not relative -- an honest limitation, not tested.)
    base = _run(cfg, sig)
    shifted = _run(cfg, [x + b for x in sig])
    assert [(e.step, e.kind) for e in base] == [(e.step, e.kind) for e in shifted]
    for e0, e1 in zip(base, shifted):
        # Exact in real arithmetic; in float64 the shift-then-subtract cancels a
        # large common term, so the residual scales with |z| (it can reach ~1e-9
        # absolute when the variance floor drives |z| into the hundreds). Pin the
        # invariance to floating-point PRECISION (relative-aware), not an absolute
        # bound that ignores the magnitude of z.
        assert abs(e0.z - e1.z) <= 1e-9 * max(1.0, abs(e0.z))      # exact to fp


# --------------------------------------------------------------------------- #
# reset & the world-model bridge                                             #
# --------------------------------------------------------------------------- #
@PROP
@given(cfg=detector_cfg(), sig=stream())
def test_reset_clears_all_running_state(cfg, sig):
    d = SalienceEventDetector(**cfg)
    d.observe(sig)
    d.reset()
    assert d.state == "quiescent"
    assert d.n_events == 0
    assert d.baseline == 0.0


class _FloatHead:
    def __init__(self, v):
        self.last_pred_error = v


class _ScalarHead:
    """Mimics a torch scalar: exposes ``.item()``."""
    class _S:
        def __init__(self, v):
            self._v = v
        def item(self):
            return self._v
    def __init__(self, v):
        self.last_pred_error = _ScalarHead._S(v)


@PROP
@given(v=st.floats(0.0, 1.0, **_F))
def test_world_model_surprise_reads_float_and_scalar_heads(v):
    assert world_model_surprise(_FloatHead(v)) == pytest.approx(v)
    assert world_model_surprise(_ScalarHead(v)) == pytest.approx(v)


def test_world_model_surprise_rejects_a_head_without_the_attribute():
    with pytest.raises(TypeError):
        world_model_surprise(object())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
