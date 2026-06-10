"""
tests/test_demo_causal_spatial_steering.py — L3 causal spatial steering demo.

Pins the behavioural contract of ``examples/demo_causal_spatial_steering.py``
(the roadshow prototype) so it cannot silently regress:

  * a legal trajectory stays on its manifold (off-manifold drift ~ 0);
  * an injected off-manifold teleport blows up the UNSTEERED belief and the
    error persists (a recurrent system carries it forward forever);
  * the steerer detects the break, fires, and the orthogonal projection +
    recurrent re-feed snaps the belief back onto the legal manifold and keeps
    it there — post-teleport drift collapses to a small fraction of the
    unsteered baseline;
  * the result is deterministic for a fixed seed.

These assertions are intentionally loose (order-of-magnitude) so the test pins
the *mechanism* (steering recovers the trajectory) rather than exact floats.
"""
import os
import sys

import pytest
import torch

# Make the repo root importable so ``examples`` resolves regardless of how
# pytest is invoked (the package has no __init__.py).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.demo_causal_spatial_steering import (  # noqa: E402
    build_scene,
    rollout,
    run_demo,
    _segment_mean,
)
from mt_lnn.causal_steering import CausalActivationSteerer  # noqa: E402


def _post_break(logs, scene, key="off_manifold"):
    T = scene.legal.shape[0]
    return _segment_mean(logs, scene.break_step, T, key)


def _pre_break(logs, scene, key="off_manifold"):
    return _segment_mean(logs, 0, scene.break_step, key)


def _make(seed=0):
    scene = build_scene(seed=seed)
    steerer = CausalActivationSteerer(strength=1.0, floor=0.5, adaptive=False)
    base = rollout(scene, steerer=None)
    steered = rollout(scene, steerer=steerer)
    return scene, base, steered


def test_legal_trajectory_stays_on_manifold():
    scene, base, steered = _make()
    # Before the teleport, both rollouts hug the legal plane.
    assert _pre_break(base, scene) < 0.1
    assert _pre_break(steered, scene) < 0.1


def test_unsteered_teleport_causes_lasting_damage():
    scene, base, _ = _make()
    # The teleport persists in the recurrent state: the unsteered belief stays
    # far off-manifold for the whole post-teleport tail, not just one step.
    assert _post_break(base, scene) > 1.0
    # And it really is a *lasting* jump vs. the pre-teleport baseline.
    assert _post_break(base, scene) > 10.0 * _pre_break(base, scene)


def test_steering_recovers_the_trajectory():
    scene, base, steered = _make()
    base_post = _post_break(base, scene)
    steer_post = _post_break(steered, scene)
    # Steering collapses the lasting drift to a small fraction of the baseline.
    assert steer_post < 0.2 * base_post
    # And the recovered belief is genuinely back near the manifold.
    assert steer_post < 0.5


def test_steering_fires_exactly_at_the_break():
    scene, _, steered = _make()
    fired = [s.step for s in steered if s.applied]
    # The detector should trip on the teleport step (and the demo's single-shot
    # teleport means it should not need to fire repeatedly afterwards).
    assert scene.break_step in fired
    assert len(fired) >= 1
    # No spurious firing on the clean pre-teleport segment.
    assert all(step >= scene.break_step for step in fired)


def test_deterministic_for_fixed_seed():
    _, _, steered_a = _make(seed=3)
    _, _, steered_b = _make(seed=3)
    a = [s.off_manifold for s in steered_a]
    b = [s.off_manifold for s in steered_b]
    assert a == pytest.approx(b)


@pytest.mark.parametrize("seed", [0, 1, 2, 5])
def test_recovery_is_robust_across_seeds(seed):
    scene, base, steered = _make(seed=seed)
    assert _post_break(steered, scene) < 0.2 * _post_break(base, scene)


def test_run_demo_smoke(capsys):
    # The top-level entry point runs end-to-end and reports a positive verdict.
    args = type("A", (), dict(
        seed=0, steps=32, d_model=48, break_step=18, teleport_scale=6.0,
        floor=0.5, plot=False, plot_path="unused.png",
    ))()
    summary = run_demo(args)
    assert summary["verdict"] is True
    assert summary["n_fired"] >= 1
    out = capsys.readouterr().out
    assert "VERDICT" in out and "[OK]" in out
