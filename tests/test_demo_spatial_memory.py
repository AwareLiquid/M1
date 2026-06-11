"""
tests/test_demo_spatial_memory.py -- L2 spatial sequence memory demo.

Pins the behavioural contract of ``examples/demo_spatial_memory.py`` (the L2
roadshow prototype) so it cannot silently regress:

  * recall at (or very near) a written landmark is near-perfect;
  * recall degrades *gracefully* as the positional cue gets noisier -- it falls
    smoothly rather than collapsing, and large noise drops it well below clean;
  * addressing is decisive -- the right content beats the best wrong content by
    a clear margin (the memory retrieves, it does not blur);
  * the demo's memory is zero-parameter;
  * the result is deterministic for a fixed seed and robust across seeds.

Assertions are intentionally loose (order-of-magnitude) so the test pins the
*mechanism* (place-cued pattern completion) rather than exact floats.
"""
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.demo_spatial_memory import (  # noqa: E402
    NOISE_GRID,
    build_world,
    build_memory,
    recall_accuracy,
    discrimination_margin,
    run_demo,
)


def _args(**kw):
    base = dict(
        seed=0, n_landmarks=8, d_model=32, n_place=256, arena=2.2,
        sigma=0.12, plot=False, plot_path="unused.png",
    )
    base.update(kw)
    return type("A", (), base)()


def _make(seed=0, n_landmarks=8):
    world = build_world(n_landmarks=n_landmarks, d_model=32, arena=2.2, seed=seed)
    mem = build_memory(world, n_place=256, sigma=0.12)
    return world, mem


def test_clean_recall_is_near_perfect():
    world, mem = _make()
    assert recall_accuracy(mem, world, noise=0.0, seed=1) > 0.95


def test_small_noise_still_recovers():
    world, mem = _make()
    # A cue within a place-field width still completes to the right content.
    assert recall_accuracy(mem, world, noise=0.05, seed=1) > 0.9


def test_graceful_degradation():
    world, mem = _make()
    accs = [recall_accuracy(mem, world, noise=nz, seed=1) for nz in NOISE_GRID]
    clean, degraded = accs[0], accs[-1]
    # Large noise drops recall clearly below clean (recall tracks position) ...
    assert degraded < clean - 0.2
    # ... and the fall is monotone-ish: each step never *rises* by much.
    assert all(accs[i + 1] <= accs[i] + 1e-3 for i in range(len(accs) - 1))


def test_addressing_is_decisive():
    world, mem = _make()
    assert discrimination_margin(mem, world) > 0.5


def test_memory_is_zero_parameter():
    _, mem = _make()
    assert sum(p.numel() for p in mem.parameters()) == 0


def test_run_demo_verdict(capsys):
    summary = run_demo(_args())
    assert summary["verdict"] is True
    assert summary["clean_recall"] > 0.95
    out = capsys.readouterr().out
    assert "VERDICT" in out and "[OK]" in out


def test_deterministic_for_fixed_seed():
    a = run_demo(_args(seed=3))
    b = run_demo(_args(seed=3))
    assert a["curve"] == pytest.approx(b["curve"])
    assert a["discrimination_margin"] == pytest.approx(b["discrimination_margin"])


@pytest.mark.parametrize("seed", [0, 1, 2, 5])
def test_verdict_robust_across_seeds(seed):
    assert run_demo(_args(seed=seed))["verdict"] is True


def test_more_landmarks_still_recalls():
    world, mem = _make(n_landmarks=12)
    assert recall_accuracy(mem, world, noise=0.0, seed=1) > 0.9
    assert discrimination_margin(mem, world) > 0.4
