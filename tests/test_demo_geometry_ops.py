"""
tests/test_demo_geometry_ops.py -- Fisher-Rao geometry demo contract.

Pins the behaviour of ``examples/demo_geometry_ops.py``: the geodesic obeys the
constant-speed law and its midpoint is the Karcher barycenter, while the
Euclidean chord midpoint genuinely disagrees (the metric-inconsistency point).
"""
import math
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.demo_geometry_ops import run_demo, print_report   # noqa: E402


def _args(steps):
    return type("A", (), {"steps": steps})()


def test_geodesic_midpoint_is_exact_half_and_constant_speed():
    s = run_demo(_args(10))
    assert s["max_speed_err"] < 1e-6                      # constant speed
    assert abs(s["d_geo"] - s["d_pq"] / 2) < 1e-6         # midpoint is the half
    assert 0.0 <= s["d_pq"] <= math.pi + 1e-6             # distance in range
    assert all(on for _, _, _, on in s["speed_rows"])     # geodesic stays on simplex


def test_euclidean_chord_disagrees_with_the_geodesic():
    s = run_demo(_args(10))
    assert s["chord_gap"] > 1e-3                           # the chord is a different path
    assert s["d_euc"] > s["d_geo"]                         # and the wrong ruler reads longer


def test_karcher_barycenter_matches_geodesic_midpoint():
    s = run_demo(_args(10))
    assert s["bary_gap"] < 1e-5


def test_report_prints_ok_verdict(capsys):
    print_report(run_demo(_args(6)))
    out = capsys.readouterr().out
    assert "VERDICT [OK]" in out
    assert "Fisher-Rao" in out


def test_steps_argument_controls_table_length():
    assert len(run_demo(_args(4))["speed_rows"]) == 5
    assert len(run_demo(_args(12))["speed_rows"]) == 13
