"""
tests/test_demo_failsafe.py -- failsafe demo contract.

Pins the behaviour of ``examples/demo_failsafe.py``: the blind-rollout guard
coasts on the world model through the dropout window and then goes dark, the
output breaker keeps every emitted value finite and inside the red-lines while
tripping on sustained violations, and the whole report is deterministic.
"""
import math
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.demo_failsafe import (  # noqa: E402
    run_guard, run_breaker, print_report, command_stream,
)


def _args(d_model=32, horizon=6, trust_decay=0.8, floor=0.35):
    return type("A", (), {"d_model": d_model, "horizon": horizon,
                          "trust_decay": trust_decay, "floor": floor})()


# --- Scene 1: blind rollout guard -----------------------------------------


def test_guard_serves_live_then_coasts_then_goes_dark():
    d = run_guard(_args())
    rows = d["rows"]
    # first three ticks are live (the feed is up)
    assert [r.source for r in rows[:3]] == ["live", "live", "live"]
    # the dropout window forces at least one imagined coast and then darkness
    assert sum(r.source == "imagined" for r in rows) >= 1
    assert sum(r.source == "dark" for r in rows) >= 1


def test_guard_re_coasts_after_feed_returns():
    d = run_guard(_args())
    sources = [r.source for r in d["rows"]]
    # after the long dropout there is a single live frame, then dropouts again;
    # the guard must re-coast (imagined) rather than stay dark.
    assert sources[-1] == "imagined"


def test_guard_dark_only_after_confidence_or_budget():
    d = run_guard(_args())
    # a higher floor goes dark sooner (fewer imagined ticks)
    strict = run_guard(_args(floor=0.6))
    loose = run_guard(_args(floor=0.0))
    assert sum(r.source == "imagined" for r in loose["rows"]) >= \
           sum(r.source == "imagined" for r in strict["rows"])


# --- Scene 2: circuit breaker ----------------------------------------------


def test_breaker_keeps_every_output_safe():
    d = run_breaker(_args())
    for r in d["rows"]:
        assert math.isfinite(r.value)
        assert -1.0 <= r.value <= 1.0


def test_breaker_scrubs_nan_and_trips_on_sustained_violation():
    d = run_breaker(_args())
    reasons = [r.reason for r in d["rows"]]
    assert "nonfinite" in reasons                        # the NaN was caught
    assert d["n_tripped"] >= 1                            # sustained bounds tripped it


def test_breaker_recovers_and_settles():
    d = run_breaker(_args())
    # the final settle ticks (clean 0.3) end un-tripped at the commanded value
    assert not d["rows"][-1].tripped
    assert d["rows"][-1].value == pytest.approx(0.3, abs=1e-6)


# --- determinism + report --------------------------------------------------


def test_demo_is_deterministic():
    a = run_guard(_args())
    b = run_guard(_args())
    assert [r.source for r in a["rows"]] == [r.source for r in b["rows"]]
    assert [r.confidence for r in a["rows"]] == [r.confidence for r in b["rows"]]


def test_report_prints_verdict(capsys):
    print_report(run_guard(_args()), run_breaker(_args()))
    out = capsys.readouterr().out
    assert "VERDICT [OK]" in out
    assert "SCENE 1" in out and "SCENE 2" in out
    # ASCII-only (Windows/GBK console safe)
    assert out.isascii()
