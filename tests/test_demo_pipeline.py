"""
tests/test_demo_pipeline.py -- dual-speed sentry demo contract.

Pins ``examples/demo_pipeline.py``: the steady approach wakes nobody, the
manoeuvre ignites exactly one salient event, the sensor dropout is coasted, the
drone breaches the zone, the aim command stays bounded and slew-limited, and the
ASCII report is deterministic and GBK-safe.
"""
import math
import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.demo_pipeline import run, print_report, scenario, align_feed  # noqa: E402


def _args(radius=3.0, ignite_z=3.0):
    return type("A", (), {"radius": radius, "ignite_z": ignite_z})()


def test_manoeuvre_ignites_once_not_the_approach():
    d = run(_args())
    igs = [t for t in d["ticks"] if t.event is not None]
    assert len(igs) == 1
    assert igs[0].tick >= 14                              # at the turn, not the approach


def test_dropout_is_coasted():
    d = run(_args())
    assert any(t.source == "imagined" for t in d["ticks"])
    assert d["ticks"][-1].source == "live"               # recovers


def test_zone_is_breached():
    d = run(_args())
    assert any(t.inside_zone for t in d["ticks"])
    assert any(not t.inside_zone for t in d["ticks"])


def test_command_bounded_and_slew_limited():
    d = run(_args())
    cmds = [t.command for t in d["ticks"]]
    for c in cmds:
        assert math.isfinite(c) and -math.pi / 2 - 1e-9 <= c <= math.pi / 2 + 1e-9
    for i in range(len(cmds) - 1):
        assert abs(cmds[i + 1] - cmds[i]) <= math.radians(20.0) + 1e-6


def test_lower_ignite_z_is_at_least_as_sensitive():
    strict = run(_args(ignite_z=4.0))
    loose = run(_args(ignite_z=2.0))
    n_strict = sum(t.event is not None for t in strict["ticks"])
    n_loose = sum(t.event is not None for t in loose["ticks"])
    assert n_loose >= n_strict


def test_demo_is_deterministic():
    a = run(_args())
    b = run(_args())
    assert [(t.source, round(t.command, 6), t.event is not None) for t in a["ticks"]] == \
           [(t.source, round(t.command, 6), t.event is not None) for t in b["ticks"]]


def test_report_prints_map_log_and_verdict(capsys):
    print_report(run(_args()))
    out = capsys.readouterr().out
    assert "VERDICT [OK]" in out
    assert "DUAL-SPEED SENTRY" in out
    assert "IGNITE" in out                               # the event reached the log
    assert "H" in out                                    # the sensor head on the map
    assert out.isascii()                                 # Windows/GBK console safe


def test_ingest_front_end_detects_the_dropout_as_a_coverage_gap():
    d = run(_args())
    ing = d["ingest"]
    # the lost frames are exactly the samples that never reached the dt grid
    assert ing["n_lost"] == ing["n_grid"] - ing["n_samples"]
    assert ing["n_lost"] >= 1
    # and the coverage-derived dropout is what the sentry actually coasts
    assert any(t.source == "imagined" for t in d["ticks"])


def test_align_feed_resamples_covered_steps_to_the_true_trajectory():
    # on a clean clock the front-end is identity at every covered step, so the
    # live ticks see the original trajectory unchanged (the dropout aside).
    steps = scenario()
    frames, _ = align_feed(steps, dt=1.0, max_gap=0.9)
    assert len(frames) == len(steps)
    for (p_in, _, ok_in), (p_out, _, ok_cov) in zip(steps, frames):
        if ok_in and ok_cov:
            assert torch.allclose(p_in, p_out, atol=1e-6)


def test_report_mentions_the_ingest_alignment(capsys):
    print_report(run(_args()))
    out = capsys.readouterr().out
    assert "ingest:" in out
    assert "coverage gap" in out
    assert out.isascii()


def test_report_surfaces_the_slow_layer_verdict(capsys):
    print_report(run(_args()))
    out = capsys.readouterr().out
    assert "SLOW[" in out                                # slow layer verdict in the log
    assert "slow layer @tick" in out                     # and in the summary
    # the ignition is at the breach, so the woken assessment escalates to ENGAGE
    assert "threat ENGAGE" in out
