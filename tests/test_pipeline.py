"""
tests/test_pipeline.py -- the dual-speed sentry integration contract.

Pins how the layers compose in ``mt_lnn.pipeline.DualSpeedSentry``: a steady
approach wakes nobody while a manoeuvre ignites exactly one salient event; a
sensor dropout is coasted on the world model and a long dropout goes dark; the
protected-zone containment is recomputed every tick; and the actuator command is
*always* within the mechanical limit and slew-rate -- through dropouts and the
near-field bearing spike alike.
"""
import math
import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mt_lnn.pipeline import DualSpeedSentry, SentryTick, PerceptionEvent  # noqa: E402
from mt_lnn.slow_layer import ThreatAssessment  # noqa: E402


def _sentry(**kw):
    base = dict(left_ear=[-0.1, 0.0], right_ear=[0.1, 0.0],
                zone_center=[0.0, 0.0], danger_radius=3.0, freq=1000.0, dt=1.0)
    base.update(kw)
    return DualSpeedSentry(**base)


def _steady_then_manoeuvre(drop_ticks=()):
    """Constant-velocity approach (14 ticks) then a sharp turn (6 ticks).

    ``drop_ticks`` is a set of 0-based indices *within the manoeuvre phase* whose
    sensor frame is dropped.
    """
    steps = []
    pos = torch.tensor([-14.0, 8.0]); vel = torch.tensor([1.0, -0.4])
    for _ in range(14):
        pos = pos + vel
        steps.append((pos.clone(), vel.clone(), True))
    vel = torch.tensor([0.2, -1.2])
    for t in range(6):
        pos = pos + vel
        steps.append((pos.clone(), vel.clone(), t not in drop_ticks))
    return steps


def _run(sentry, steps):
    return [sentry.step(p, v, sensor_ok=ok) for (p, v, ok) in steps]


# --- salience: steady approach silent, manoeuvre ignites once --------------


def test_steady_approach_ignites_nothing_before_manoeuvre():
    s = _sentry()
    ticks = _run(s, _steady_then_manoeuvre())
    # no ignition in the steady-approach phase (first 14 ticks)
    assert all(t.event is None for t in ticks[:14])


def test_manoeuvre_ignites_exactly_once():
    s = _sentry()
    ticks = _run(s, _steady_then_manoeuvre())
    igs = [t for t in ticks if t.event is not None]
    assert len(igs) == 1
    assert isinstance(igs[0].event, PerceptionEvent)
    assert igs[0].event.salience >= 3.0                 # crossed ignite_z


def test_event_carries_perception_snapshot():
    s = _sentry()
    ticks = _run(s, _steady_then_manoeuvre())
    ev = next(t.event for t in ticks if t.event is not None)
    assert ev.tick >= 14                                 # fires at the turn, not before
    assert math.isfinite(ev.azimuth) and math.isfinite(ev.distance)


# --- slow layer: the ignition actually wakes a multi-step assessment -------


def test_ignition_wakes_the_slow_layer_with_an_assessment():
    s = _sentry()
    ticks = _run(s, _steady_then_manoeuvre())
    ev = next(t.event for t in ticks if t.event is not None)
    assert isinstance(ev.assessment, ThreatAssessment)   # the slow layer actually ran
    assert ev.assessment.woken_tick == ev.tick           # woken on the ignition tick
    assert ev.assessment.level in ("CLEAR", "WATCH", "ENGAGE")
    assert ev.assessment.horizon >= 1


def test_custom_slow_layer_is_invoked_on_ignition():
    calls = []

    class _SpyLayer:
        def assess(self, pos, vel, *, tick):
            calls.append(tick)
            return ThreatAssessment(
                woken_tick=tick, horizon=1, breaches=True, eta_breach=0,
                min_range=0.0, closest_offset=0, level="ENGAGE", recommendation="spy",
            )

    s = _sentry(slow_layer=_SpyLayer())
    ticks = _run(s, _steady_then_manoeuvre())
    ev = next(t.event for t in ticks if t.event is not None)
    assert calls == [ev.tick]                            # called exactly once, on ignition
    assert ev.assessment.recommendation == "spy"


def test_slow_layer_silent_without_ignition():
    # a single steady tick never ignites -> the slow layer is never woken.
    calls = []

    class _SpyLayer:
        def assess(self, pos, vel, *, tick):
            calls.append(tick)
            return None

    s = _sentry(slow_layer=_SpyLayer())
    s.step([-10.0, 5.0], [1.0, 0.0])
    assert calls == []                                   # hot loop never paid for the slow layer


# --- dropout: coast then recover; long dropout goes dark -------------------


def test_dropout_coasts_on_world_model_then_recovers():
    s = _sentry()
    ticks = _run(s, _steady_then_manoeuvre(drop_ticks=(2, 3)))
    sources = [t.source for t in ticks]
    assert "imagined" in sources                         # coasted on a dropped frame
    assert sources[-1] == "live"                         # recovered when the feed returned


def test_long_dropout_goes_dark():
    s = _sentry(max_blind_steps=3)
    steps = _steady_then_manoeuvre()
    # drop everything after the approach: 6 consecutive dropped frames > budget 3
    steps = steps[:14] + [(p, v, False) for (p, v, _) in steps[14:]]
    ticks = _run(s, steps)
    assert any(t.source == "dark" for t in ticks)
    dark = next(t for t in ticks if t.source == "dark")
    assert dark.confidence == pytest.approx(0.0) or dark.blind_steps > 3


def test_dropout_before_any_good_frame_is_dark():
    s = _sentry()
    t = s.step([5.0, 5.0], [0.0, 0.0], sensor_ok=False)
    assert t.source == "dark"


def test_blind_steps_reset_on_live():
    s = _sentry()
    ticks = _run(s, _steady_then_manoeuvre(drop_ticks=(2, 3)))
    assert ticks[-1].blind_steps == 0                    # live again at the end


# --- output safety: always bounded + slew-limited --------------------------


def test_command_always_within_mechanical_limit():
    s = _sentry(aim_limit=math.pi / 2)
    ticks = _run(s, _steady_then_manoeuvre(drop_ticks=(2, 3)))
    for t in ticks:
        assert math.isfinite(t.command)
        assert -math.pi / 2 - 1e-9 <= t.command <= math.pi / 2 + 1e-9


def test_command_respects_slew_rate_even_on_nearfield_spike():
    slew = math.radians(20.0)
    s = _sentry(max_slew=slew)
    ticks = _run(s, _steady_then_manoeuvre())            # contains a +90deg near-field jump
    prev = 0.0
    for t in ticks:
        assert abs(t.command - prev) <= slew + 1e-6
        prev = t.command


def test_dark_holds_the_command():
    s = _sentry(max_blind_steps=3)
    steps = _steady_then_manoeuvre()
    steps = steps[:14] + [(p, v, False) for (p, v, _) in steps[14:]]
    ticks = _run(s, steps)
    # once dark, the emitted command stops moving (held, not driven by stale data)
    darks = [i for i, t in enumerate(ticks) if t.source == "dark"]
    i = darks[0]
    if i + 1 < len(ticks) and ticks[i + 1].source == "dark":
        assert ticks[i + 1].command == pytest.approx(ticks[i].command, abs=1e-6)


# --- spatial reasoning: zone containment -----------------------------------


def test_zone_entry_is_detected():
    s = _sentry(danger_radius=3.0)
    ticks = _run(s, _steady_then_manoeuvre())
    assert any(not t.inside_zone for t in ticks)         # starts outside
    assert any(t.inside_zone for t in ticks)             # crosses in


def test_azimuth_sign_tracks_side():
    s = _sentry()
    # a target straight out on the right ear's side reads positive azimuth
    t = s.step([10.0, 0.0], [0.0, 0.0])
    assert t.azimuth > 0.0
    s.reset()
    t = s.step([-10.0, 0.0], [0.0, 0.0])
    assert t.azimuth < 0.0


# --- determinism + bookkeeping ---------------------------------------------


def test_sentry_is_deterministic():
    a = _run(_sentry(), _steady_then_manoeuvre(drop_ticks=(2, 3)))
    b = _run(_sentry(), _steady_then_manoeuvre(drop_ticks=(2, 3)))
    assert [(t.source, round(t.command, 6), t.event is not None) for t in a] == \
           [(t.source, round(t.command, 6), t.event is not None) for t in b]


def test_orchestrator_adds_no_parameters():
    assert _sentry().n_parameters == 0


def test_reset_clears_state():
    s = _sentry()
    _run(s, _steady_then_manoeuvre())
    s.reset()
    assert s._tick == 0
    assert s._last_pos is None
    t = s.step([-10.0, 5.0], [1.0, 0.0])
    assert t.tick == 1 and t.surprise is None            # no prev after reset


def test_validates_config():
    with pytest.raises(ValueError):
        _sentry(danger_radius=0.0)
    with pytest.raises(ValueError):
        _sentry(left_ear=[0.0, 0.0], right_ear=[0.0, 0.0])   # coincident ears
