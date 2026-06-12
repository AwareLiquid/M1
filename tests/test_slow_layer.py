"""
tests/test_slow_layer.py -- the slow half of the dual-speed engine.

Pins ``mt_lnn.slow_layer.SlowThreatAssessor``: a ballistic forecast over a
horizon that grades the threat by time-to-breach, names the closest approach,
escalates CLEAR -> WATCH -> ENGAGE on the ETA, adds zero parameters, and is
deterministic. The whole point of the slow layer is that it *reasons* over a
horizon -- so the contract is on the forecast, not a single step.
"""
import math
import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mt_lnn.slow_layer import SlowThreatAssessor, ThreatAssessment  # noqa: E402


def _assessor(**kw):
    base = dict(zone_center=[0.0, 0.0], danger_radius=3.0, dt=1.0, horizon=12, engage_eta=3)
    base.update(kw)
    return SlowThreatAssessor(**base)


# --- adds no parameters ----------------------------------------------------


def test_slow_layer_adds_no_parameters():
    assert _assessor().n_parameters == 0


# --- a target heading straight in breaches; ETA is the distance in steps ----


def test_incoming_target_breaches_with_correct_eta():
    s = _assessor(danger_radius=3.0)
    # at (10, 0) closing at (-1, 0): enters the radius-3 ball after 7 steps.
    a = s.assess([10.0, 0.0], [-1.0, 0.0])
    assert a.breaches
    assert a.eta_breach == 7
    assert a.min_range <= 3.0


def test_already_inside_has_zero_eta():
    s = _assessor(danger_radius=3.0)
    a = s.assess([1.0, 0.0], [0.0, 0.0])         # sitting inside the zone
    assert a.breaches and a.eta_breach == 0
    assert a.level == "ENGAGE"


# --- a target moving away never breaches -> CLEAR ---------------------------


def test_receding_target_is_clear():
    s = _assessor()
    a = s.assess([10.0, 0.0], [1.0, 0.0])        # heading outward
    assert not a.breaches
    assert a.eta_breach is None
    assert a.level == "CLEAR"
    assert a.min_range >= 3.0


# --- a target that passes wide of the zone never breaches -------------------


def test_tangential_miss_is_clear():
    s = _assessor(danger_radius=1.0)
    # passes the y-axis at x=0 with y=8 -- never within radius 1 of the origin.
    a = s.assess([10.0, 8.0], [-1.0, 0.0], )
    assert not a.breaches
    assert a.min_range == pytest.approx(8.0, abs=1e-4)
    assert a.closest_offset == 10                 # closest when x crosses 0


# --- ETA drives the CLEAR -> WATCH -> ENGAGE escalation ---------------------


def test_eta_escalates_watch_versus_engage():
    far = _assessor(danger_radius=3.0, engage_eta=3)
    # breach at step 7 > engage_eta 3 -> WATCH
    assert far.assess([10.0, 0.0], [-1.0, 0.0]).level == "WATCH"
    # breach at step 1 <= engage_eta 3 -> ENGAGE
    assert far.assess([4.0, 0.0], [-1.0, 0.0]).level == "ENGAGE"


def test_engage_eta_threshold_is_inclusive():
    s = _assessor(danger_radius=3.0, engage_eta=4)
    a = s.assess([7.0, 0.0], [-1.0, 0.0])         # enters radius 3 after exactly 4 steps
    assert a.eta_breach == 4
    assert a.level == "ENGAGE"                     # 4 <= engage_eta 4


# --- horizon bounds the lookahead ------------------------------------------


def test_short_horizon_misses_a_distant_breach():
    short = _assessor(danger_radius=3.0, horizon=3)
    a = short.assess([10.0, 0.0], [-1.0, 0.0])    # would breach at step 7, beyond horizon 3
    assert not a.breaches
    assert a.level == "CLEAR"
    assert a.horizon == 3


# --- recommendation is populated + ASCII -----------------------------------


def test_recommendation_is_ascii_and_nonempty():
    s = _assessor()
    for state in (s.assess([10.0, 0.0], [-1.0, 0.0]),   # WATCH
                  s.assess([4.0, 0.0], [-1.0, 0.0]),     # ENGAGE
                  s.assess([10.0, 0.0], [1.0, 0.0])):    # CLEAR
        assert state.recommendation
        assert state.recommendation.isascii()


# --- determinism + bookkeeping ---------------------------------------------


def test_assessment_is_deterministic():
    s = _assessor()
    a = s.assess([10.0, 0.0], [-1.0, 0.4], tick=15)
    b = s.assess([10.0, 0.0], [-1.0, 0.4], tick=15)
    assert a == b


def test_woken_tick_is_recorded():
    s = _assessor()
    a = s.assess([10.0, 0.0], [-1.0, 0.0], tick=42)
    assert a.woken_tick == 42
    assert isinstance(a, ThreatAssessment)


# --- config validation -----------------------------------------------------


def test_validates_config():
    with pytest.raises(ValueError):
        _assessor(danger_radius=0.0)
    with pytest.raises(ValueError):
        _assessor(horizon=0)
    with pytest.raises(ValueError):
        _assessor(engage_eta=-1)
