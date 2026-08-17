"""
tests/test_demo_salience_events.py -- salience-events demo contract.

Pins the behaviour of ``examples/demo_salience_events.py``: the slow adapted
drift fires nothing, the regime change ignites exactly once at its onset, and the
report is deterministic.
"""
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.demo_salience_events import run_demo, print_report  # noqa: E402


def _args(ignite_z=3.0, release_z=1.0, refractory=3, warmup=10):
    return type("A", (), {"ignite_z": ignite_z, "release_z": release_z,
                          "refractory": refractory, "warmup": warmup})()


def test_regime_change_ignites_once_not_the_drift(capsys):
    s = run_demo(_args())
    assert s["n_ignition"] == 1
    # the regime change starts at tick 41 in the stream; the slow drift (21..40)
    # must NOT have fired
    assert s["first_ignition_step"] == 41
    print_report(s)
    out = capsys.readouterr().out
    assert "VERDICT [OK]" in out and "ignition" in out


def test_settles_and_quiesces():
    s = run_demo(_args())
    assert s["n_quiescence"] >= 1                     # re-arms after the new normal


def test_demo_is_deterministic():
    a = run_demo(_args())
    b = run_demo(_args())
    assert [(e.step, e.kind) for e in a["events"]] == [(e.step, e.kind) for e in b["events"]]


def test_lower_threshold_is_more_sensitive():
    strict = run_demo(_args(ignite_z=3.0))
    loose = run_demo(_args(ignite_z=2.0))
    assert loose["n_ignition"] >= strict["n_ignition"]
