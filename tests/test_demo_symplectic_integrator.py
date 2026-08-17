"""
tests/test_demo_symplectic_integrator.py -- symplectic integrator demo contract.

Pins ``examples/demo_symplectic_integrator.py``: over a long circular Kepler
orbit, velocity Verlet conserves orbital energy and radius orders of magnitude
better than semi-implicit Euler, and a fixed-time dt-halving shows Euler's
position error dropping ~2x (1st order) vs Verlet's ~4x (2nd order).
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.demo_symplectic_integrator import run_demo, print_report     # noqa: E402


def _args(orbits=16.0, dt=0.05):
    return type("A", (), {"orbits": orbits, "dt": dt})()


def test_verlet_conserves_energy_far_better_than_euler():
    s = run_demo(_args())
    assert s["verlet_energy"] < s["euler_energy"] / 10.0
    assert s["verlet_radius"] < s["euler_radius"] / 10.0


def test_orbit_truth_is_constant_energy_and_radius():
    # both integrators leak SOME (they are not exact) -- but stay bounded
    s = run_demo(_args())
    assert s["euler_energy"] > 0.0 and s["verlet_energy"] > 0.0
    assert s["verlet_radius"] < 1e-2          # Verlet holds the unit orbit tight


def test_euler_is_first_order_verlet_is_second_order():
    s = run_demo(_args())
    assert 0.6 < s["euler_order"] < 1.5       # ~2x drop per dt-halving
    assert s["verlet_order"] > 1.6            # ~4x drop per dt-halving


def test_report_prints_ok_verdict(capsys):
    print_report(run_demo(_args()))
    out = capsys.readouterr().out
    assert "VERDICT [OK]" in out
    assert "velocity Verlet" in out


def test_longer_flight_grows_euler_drift_more_than_verlet():
    short = run_demo(_args(orbits=8.0))
    long = run_demo(_args(orbits=48.0))
    # Euler's energy leak accumulates with horizon; Verlet stays ~bounded
    euler_growth = long["euler_energy"] / short["euler_energy"]
    verlet_growth = long["verlet_energy"] / short["verlet_energy"]
    assert euler_growth > verlet_growth
