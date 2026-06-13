"""
tests/test_demo_topology_failsafe.py -- topological failsafe demo contract.

Pins ``examples/demo_topology_failsafe.py``: the TopologyBreaker ignores benign
jitter, trips on a mode collapse only after the debounce confirms it, and closes
again once the cluster structure recovers -- with zero trainable parameters.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.demo_topology_failsafe import run_demo, print_report     # noqa: E402


def _args(threshold=1.0):
    return type("A", (), {"threshold": threshold})()


def test_benign_jitter_never_trips():
    s = run_demo(_args())
    assert not s["benign_tripped"]


def test_collapse_trips_after_debounce():
    s = run_demo(_args())
    assert s["collapse_trip_tick"] == 1          # 2nd consecutive bad tick (trip_after=2)


def test_recovery_closes_after_clean_run():
    s = run_demo(_args())
    assert s["recovery_close_tick"] == 2         # 3rd consecutive clean tick (reset_after=3)


def test_zero_parameters():
    s = run_demo(_args())
    assert s["n_parameters"] == 0


def test_report_prints_ok_verdict(capsys):
    print_report(run_demo(_args()))
    out = capsys.readouterr().out
    assert "VERDICT [OK]" in out
    assert "TOPOLOGY FAILSAFE" in out


def test_collapse_phase_flags_srtd_reason():
    s = run_demo(_args())
    collapse = [row for row in s["log"] if row[0] == "collapse"]
    assert all(row[4] == "srtd" for row in collapse)   # divergence is the cause
    assert all(row[2] == 1 for row in collapse)         # fused to a single component
