"""
tests/test_demo_physics_ops.py -- physics-operators demo contract.

Pins the behaviour of ``examples/demo_physics_ops.py``: an elastic floor lets the
ball clear the wall, a lossy floor does not, and the verdict is deterministic.
"""
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.demo_physics_ops import run_demo, print_report  # noqa: E402


def _args(restitution, vx=3.0, vy=4.0, steps=800, dt=0.005):
    return type("A", (), {"restitution": restitution, "vx": vx, "vy": vy,
                          "steps": steps, "dt": dt})()


def test_elastic_floor_clears_the_wall(capsys):
    s = run_demo(_args(1.0))
    assert s["clears"] is True
    assert s["cross_h"] is not None and s["cross_h"] >= 0.45
    assert s["n_bounces"] >= 1                        # it actually bounced
    print_report(s)
    out = capsys.readouterr().out
    assert "VERDICT [OK]" in out and "CLEARS the wall" in out


def test_lossy_floor_hits_the_wall():
    s = run_demo(_args(0.4))
    assert s["clears"] is False


def test_verdict_is_deterministic():
    a = run_demo(_args(1.0))
    b = run_demo(_args(1.0))
    assert a["clears"] == b["clears"]
    assert a["cross_h"] == pytest.approx(b["cross_h"])
    assert a["flip_restitution"] == b["flip_restitution"]
