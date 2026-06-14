"""
tests/test_demo_top_down_modulation.py -- top-down modulation demo contract.

Pins ``examples/demo_top_down_modulation.py``: with the per-block gate closed a
top-down goal is a STRICT no-op (bit-identical to no goal), and with the gate open
two distinct goals steer the same input to different next-token distributions --
the pathway is live, default-off, zero regression.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.demo_top_down_modulation import run_demo, print_report   # noqa: E402


def _args(gate=1.0):
    return type("A", (), {"gate": gate})()


def test_closed_gate_is_strict_no_op():
    s = run_demo(_args())
    assert s["closed_diff"] == 0.0          # bit-exact: goal == no goal at init


def test_open_gate_steers_the_logits():
    s = run_demo(_args())
    assert s["open_diff"] > 1e-3
    assert s["kl_ab"] > 1e-4                 # the two goals diverge the distribution


def test_logits_stay_finite():
    s = run_demo(_args())
    assert s["logits_finite"]


def test_larger_gate_steers_at_least_as_much():
    weak = run_demo(_args(gate=0.5))
    strong = run_demo(_args(gate=2.0))
    assert strong["open_diff"] >= weak["open_diff"]


def test_report_prints_ok_verdict(capsys):
    print_report(run_demo(_args()))
    out = capsys.readouterr().out
    assert "VERDICT [OK]" in out
    assert "TOP-DOWN MODULATION" in out
