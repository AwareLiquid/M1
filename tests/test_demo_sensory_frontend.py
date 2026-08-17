"""
tests/test_demo_sensory_frontend.py -- sensory frontend demo contract.

Pins ``examples/demo_sensory_frontend.py``: a raw jittered 12-channel stream with
a burst dropout is aligned onto the core's fixed-dt grid, the dropout steps are
flagged in the coverage mask, and the projected tokens drive the backbone to
finite logits -- the perception loop closes with zero model.py coupling.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.demo_sensory_frontend import run_demo, print_report     # noqa: E402


def _args(dt=0.1):
    return type("A", (), {"dt": dt})()


def test_resampled_onto_denser_grid():
    s = run_demo(_args())
    assert s["n_steps"] > s["n_raw"]             # jittered samples -> denser uniform grid


def test_burst_dropout_is_flagged():
    s = run_demo(_args())
    assert s["n_gap_steps"] > 0
    assert not s["fully_covered"]
    assert s["coverage_ratio"] < 1.0


def test_tokens_have_backbone_width():
    s = run_demo(_args())
    assert s["embeds_shape"][-1] == s["d_model"]


def test_backbone_logits_finite_and_shaped():
    s = run_demo(_args())
    assert s["logits_shape"][:2] == (s["B"], s["n_steps"])
    assert s["logits_finite"]


def test_report_prints_ok_verdict(capsys):
    print_report(run_demo(_args()))
    out = capsys.readouterr().out
    assert "VERDICT [OK]" in out
    assert "SENSORY FRONTEND" in out


def test_finer_dt_yields_more_steps():
    coarse = run_demo(_args(dt=0.1))
    fine = run_demo(_args(dt=0.05))
    assert fine["n_steps"] > coarse["n_steps"]   # halving dt roughly doubles the grid
