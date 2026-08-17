"""
tests/test_demo_stdp_ops.py -- STDP demo contract.

Pins the behaviour of ``examples/demo_stdp_ops.py``: a causal spike stream
potentiates (with zero depression), the role-swapped stream depresses (with zero
potentiation), a coincident pair does nothing, and the O(T) trace form agrees
with the all-to-all pairwise definition to machine precision.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.demo_stdp_ops import run_demo, print_report          # noqa: E402


def _args(tau=15.0):
    return type("A", (), {"tau": tau})()


def test_causal_stream_potentiates_with_no_depression():
    s = run_demo(_args())
    assert s["dw_causal"] > 0.0
    assert s["dep_causal"] < 1e-9


def test_acausal_stream_depresses_with_no_potentiation():
    s = run_demo(_args())
    assert s["dw_acausal"] < 0.0
    assert s["pot_acausal"] < 1e-9
    # role-swap is an exact sign flip for a balanced window
    assert abs(s["dw_causal"] + s["dw_acausal"]) < 1e-9


def test_coincident_pair_does_nothing():
    s = run_demo(_args())
    assert abs(s["dw_coincident"]) < 1e-9


def test_trace_form_equals_pairwise_definition():
    s = run_demo(_args())
    assert s["trace_vs_pairwise_gap"] < 1e-9


def test_report_prints_ok_verdict(capsys):
    print_report(run_demo(_args()))
    out = capsys.readouterr().out
    assert "VERDICT [OK]" in out
    assert "POTENTIATION" in out
