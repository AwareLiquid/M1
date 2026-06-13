"""
tests/test_demo_topology_ops.py -- TDA demo contract.

Pins the behaviour of ``examples/demo_topology_ops.py``: Betti-0 reads three
clusters, a small jitter preserves the topology while a collapse breaks it, and
SRTD separates the two cases by a wide margin.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.demo_topology_ops import run_demo, print_report     # noqa: E402


def _args(spread):
    return type("A", (), {"spread": spread})()


def test_betti0_reads_three_clusters_and_collapse_breaks_them():
    s = run_demo(_args(5.0))
    assert s["b0_ref"] == 3
    assert s["b0_jittered"] == 3            # benign jitter preserves topology
    assert s["b0_collapsed"] < 3           # collapse fuses clusters


def test_srtd_separates_jitter_from_collapse():
    s = run_demo(_args(5.0))
    assert s["srtd_collapse"] > 5.0 * max(s["srtd_jitter"], 1e-9)
    assert s["srtd_jitter"] >= 0.0


def test_betti0_curve_runs_from_15_down_to_1():
    s = run_demo(_args(5.0))
    curve = s["curve"]
    assert int(curve[0]) == 15             # 3 x 5 points, nothing merged
    assert int(curve[-1]) == 1             # everything merged
    diffs = curve[1:] - curve[:-1]
    assert bool((diffs <= 0).all())        # monotone non-increasing


def test_report_prints_ok_verdict(capsys):
    print_report(run_demo(_args(5.0)))
    out = capsys.readouterr().out
    assert "VERDICT [OK]" in out
    assert "Betti-0" in out


def test_wider_spread_increases_total_persistence():
    near = run_demo(_args(4.0))["total_persistence"]
    far = run_demo(_args(8.0))["total_persistence"]
    assert far > near                       # farther clusters => longer-lived bars
