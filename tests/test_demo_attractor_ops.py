"""
tests/test_demo_attractor_ops.py -- attractor self-stabilization demo contract.

Pins ``examples/demo_attractor_ops.py``: a stable map's measured convergence rate
matches the closed form and its energy descends, a divergent map is flagged with
a negative rate and growing energy, and the nonlinear basin probe recovers the
stability boundary at 1.0.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.demo_attractor_ops import run_demo, print_report     # noqa: E402


def _args(rho=0.9):
    return type("A", (), {"rho": rho})()


def test_stable_map_rate_matches_closed_form_and_energy_descends():
    s = run_demo(_args())
    assert s["is_contraction"]
    assert abs(s["empirical_rate"] - s["asymptotic_rate"]) < 1e-2
    assert s["max_lyap_ratio"] < 1.0
    assert s["settling_time"] > 0


def test_linear_basin_is_unbounded_hits_cap():
    s = run_demo(_args())
    assert s["lin_basin"] == s["basin_cap"]


def test_divergent_map_is_flagged():
    s = run_demo(_args())
    assert not s["div_contracts"]
    assert s["div_rate"] < 0.0
    assert s["div_min_lyap"] > 1.0


def test_nonlinear_basin_recovers_the_boundary():
    s = run_demo(_args())
    assert abs(s["nl_basin"] - 1.0) < 1e-2


def test_report_prints_ok_verdict(capsys):
    print_report(run_demo(_args()))
    out = capsys.readouterr().out
    assert "VERDICT [OK]" in out
    assert "spectral radius" in out


def test_faster_contraction_settles_sooner():
    fast = run_demo(_args(0.5))["settling_time"]
    slow = run_demo(_args(0.95))["settling_time"]
    assert fast < slow                          # smaller rho => quicker settling
