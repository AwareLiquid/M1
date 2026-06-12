"""
tests/test_demo_acoustic_ops.py -- acoustic-operators demo contract.

Pins the behaviour of ``examples/demo_acoustic_ops.py``: the fly-by localizes a
bearing that sweeps left->right through zero while the Doppler falls through the
rest frequency, the two-speaker scene produces a non-trivial interference
pattern, and the whole ASCII report is deterministic and GBK-safe.
"""
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.demo_acoustic_ops import (  # noqa: E402
    run_flyby, run_interference, print_report,
)


def _args(ticks=25, standoff=2.0, freq=1000.0):
    return type("A", (), {"ticks": ticks, "standoff": standoff, "freq": freq})()


# --- Scene 1: fly-by localization + Doppler --------------------------------


def test_flyby_bearing_sweeps_left_to_right():
    d = run_flyby(_args())
    az = d["scene"].azimuth
    assert az[0].item() < 0.0                             # starts left
    assert az[-1].item() > 0.0                            # ends right
    mid = az[az.shape[0] // 2].item()
    assert abs(mid) < 1e-3                                # ~ahead at closest approach


def test_flyby_doppler_falls_through_rest_freq():
    d = run_flyby(_args(freq=1000.0))
    dop = d["scene"].doppler
    assert dop[0].item() > 1000.0                         # approaching
    assert dop[-1].item() < 1000.0                        # receding


def test_flyby_doppler_monotonic_decreasing():
    d = run_flyby(_args())
    dop = d["scene"].doppler
    assert all(dop[i + 1].item() <= dop[i].item() + 1e-4 for i in range(dop.shape[0] - 1))


# --- Scene 2: interference -------------------------------------------------


def test_interference_has_loud_and_quiet_bands():
    d = run_interference(_args(freq=1000.0))
    amp = d["amp"]
    # a real standing pattern: the quietest point is well below the loudest
    assert min(amp) < 0.5 * max(amp)


def test_interference_center_is_constructive():
    d = run_interference(_args(freq=1000.0))
    amp = d["amp"]
    mid = amp[len(amp) // 2]
    # the midpoint is equidistant from both speakers -> in phase -> a loud band
    assert mid == pytest.approx(max(amp), rel=0.05)


# --- determinism + report --------------------------------------------------


def test_demo_is_deterministic():
    a = run_flyby(_args())
    b = run_flyby(_args())
    assert a["scene"].azimuth.tolist() == b["scene"].azimuth.tolist()
    assert a["scene"].doppler.tolist() == b["scene"].doppler.tolist()


def test_report_prints_verdict_and_is_ascii(capsys):
    print_report(run_flyby(_args()), run_interference(_args()))
    out = capsys.readouterr().out
    assert "VERDICT [OK]" in out
    assert "SCENE 1" in out and "SCENE 2" in out
    assert out.isascii()                                 # Windows/GBK console safe
