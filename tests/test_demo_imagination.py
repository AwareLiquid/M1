"""
tests/test_demo_imagination.py -- L4 imagination-rollout demo contract.

Pins the behavioural verdict of ``examples/demo_imagination.py`` so it cannot
silently regress:

  * the rollout adds zero parameters to the live model and confidence decays
    with the horizon (Part 1 integration onto a real MTLNNModel);
  * on a head trained on a rotating world, the composed imagination tracks the
    true k-step future markedly better than a static "nothing changes" baseline
    (Part 2 -- the "it actually composes" verdict);
  * the printed report carries the [OK] verdict;
  * deterministic for a fixed seed.
"""
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.demo_imagination import run_demo, print_report  # noqa: E402


def _args(**kw):
    base = dict(
        seed=0, horizon=5, prompt_len=6, d_model=104, n_layers=2,
        trust_decay=0.85, novelty_penalty=0.5, theta=0.5, train_steps=600,
    )
    base.update(kw)
    return type("A", (), base)()


def test_run_demo_verdict_and_report(capsys):
    summary = run_demo(_args())
    assert summary["verdict"] is True
    p1, p2 = summary["part1"], summary["part2"]
    assert p1["added_params"] == 0
    assert p1["conf_decreases"] is True
    assert p2["far_imag"] > p2["far_static"] + 0.05
    assert p2["imag_mean"] > p2["static_mean"]
    print_report(summary)
    out = capsys.readouterr().out
    assert "VERDICT" in out and "[OK]" in out


def test_part1_integration_zero_params():
    from examples.demo_imagination import part1_real_model
    p1 = part1_real_model(_args())
    assert p1["added_params"] == 0 and p1["imag_params"] == 0
    assert len(p1["confidence"]) == 5 and len(p1["novelty"]) == 5
    assert all(0.0 <= c <= 1.0 for c in p1["confidence"])


def test_part2_composition_beats_static():
    from examples.demo_imagination import part2_composition
    p2 = part2_composition(_args())
    # imagined tracking dominates at every horizon for a learned rotation
    assert all(ci > cs for ci, cs in zip(p2["cos_imag"], p2["cos_static"]))


def test_deterministic_for_fixed_seed():
    a = run_demo(_args(seed=2))
    b = run_demo(_args(seed=2))
    assert a["part2"]["imag_mean"] == pytest.approx(b["part2"]["imag_mean"])
    assert a["part1"]["confidence"] == b["part1"]["confidence"]


@pytest.mark.parametrize("seed", [0, 1, 3])
def test_verdict_robust_across_seeds(seed):
    assert run_demo(_args(seed=seed))["verdict"] is True
