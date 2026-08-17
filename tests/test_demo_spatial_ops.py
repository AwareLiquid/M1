"""
tests/test_demo_spatial_ops.py -- spatial-operators demo contract.

Pins the behaviour of ``examples/demo_spatial_ops.py``: a short stride cannot
cross the gap, a long stride can, and the unlock threshold is deterministic.
"""
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.demo_spatial_ops import run_demo, print_report  # noqa: E402


def _args(reach):
    return type("A", (), {"reach": reach})()


def test_short_stride_cannot_cross_gap(capsys):
    s = run_demo(_args(1.1))
    assert s["goal_reached"] is False
    assert s["n_reachable"] == 6                    # only the near (left) cluster
    print_report(s)
    out = capsys.readouterr().out
    assert "VERDICT [OK]" in out and "NOT reachable" in out


def test_long_stride_crosses_gap():
    s = run_demo(_args(2.1))
    assert s["goal_reached"] is True
    assert s["n_reachable"] == s["n_stones"] == 12
    assert s["goal_hops"] == 3


def test_unlock_threshold_is_deterministic():
    a = run_demo(_args(1.1))
    b = run_demo(_args(1.1))
    assert a["unlock_stride"] == b["unlock_stride"] == 2.0
