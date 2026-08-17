"""
tests/test_demo_streaming_continual.py -- streaming continual-learning demo
contract.

Pins ``examples/demo_streaming_continual.py``: the real MTLNNModel trained across
a task sequence twice -- naive vs reservoir replay -- and scored with the
continual-learning metrics. The load-bearing claim is honest and quantitative:

  * the task family genuinely induces catastrophic forgetting under naive
    sequential training (its forgetting measure is near-total);
  * reservoir replay learns each task about as well (comparable learning
    accuracy) while forgetting far less and ending much more accurate overall;
  * replay adds ZERO model parameters (it is data rehearsal, not capacity);
  * the synthetic task generator obeys its own cyclic rule (so "accuracy" means
    something), and the report prints an ASCII-only OK verdict.

One shared training run (module-scoped fixture) backs the slow assertions; the
generator/structure checks are instant.
"""
import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.demo_streaming_continual import (  # noqa: E402
    build_tasks,
    make_batch,
    print_report,
    run_demo,
    vocab_size_for,
)


def _args(seed=0):
    # small but sufficient: naive forgetting is total, replay clearly recovers
    return type("A", (), {
        "seed": seed, "n_tasks": 3, "v_data": 12, "seq_len": 9,
        "steps": 30, "batch": 16, "lr": 4e-3,
        "replay_cap": 256, "replay_batch": 16,
    })()


@pytest.fixture(scope="module")
def result():
    """Train both regimes once and share the outcome across the slow tests."""
    return run_demo(_args())


# --- the headline: forgetting is real, replay cures it ---------------------

@pytest.mark.slow
def test_naive_training_catastrophically_forgets(result):
    naive = result["naive"]
    # final accuracy collapses toward "only the last task" = 1/n_tasks
    assert naive.average_accuracy < 0.45
    assert naive.forgetting > 0.5          # near-total forgetting
    assert naive.backward_transfer < -0.5  # earlier tasks wiped


@pytest.mark.slow
def test_replay_retains_while_still_learning(result):
    naive, replay = result["naive"], result["replay"]
    assert replay.learning_accuracy >= naive.learning_accuracy - 0.15  # learns it
    assert replay.forgetting <= naive.forgetting - 0.10                # forgets less
    assert replay.average_accuracy >= naive.average_accuracy + 0.10    # ends ahead
    assert replay.average_accuracy > 0.75                              # actually good


@pytest.mark.slow
def test_replay_adds_zero_parameters(result):
    # replay is data rehearsal: the verdict's own checks must all be satisfied
    assert all(result["checks"].values())
    assert result["ok"]


@pytest.mark.slow
def test_accuracy_matrices_are_square(result):
    n = result["args"].n_tasks
    for key in ("naive_R", "replay_R"):
        R = result[key]
        assert len(R) == n and all(len(row) == n for row in R)


# --- generator / structure (instant, no training) -------------------------

def test_make_batch_follows_the_cyclic_rule():
    tasks = build_tasks(n_tasks=3, v_data=12)
    t = tasks[2]                                  # shift = 3
    gen = torch.Generator().manual_seed(0)
    ids, labels = make_batch(t, n_seq=8, seq_len=9, v_data=12, gen=gen)
    assert ids.shape == (8, 9)
    assert torch.equal(ids, labels)              # labels aligned, model shifts
    assert torch.all(ids[:, 0] == t.task_tok)    # prefix is the task id
    # every data step advances by the task's shift (mod v_data)
    data = ids[:, 1:]
    expect_next = (data[:, :-1] + t.shift) % 12
    assert torch.equal(data[:, 1:], expect_next)


def test_build_tasks_have_distinct_nonzero_shifts():
    tasks = build_tasks(n_tasks=4, v_data=10)
    shifts = [t.shift for t in tasks]
    assert shifts == [1, 2, 3, 4]
    assert all(s != 0 for s in shifts)
    # task tokens sit above the data block and are unique
    toks = [t.task_tok for t in tasks]
    assert toks == [10, 11, 12, 13]
    assert vocab_size_for(4, 10) == 14


# --- report ----------------------------------------------------------------

@pytest.mark.slow
def test_report_prints_ascii_ok_verdict(result, capsys):
    print_report(result)
    out = capsys.readouterr().out
    assert "VERDICT [OK]" in out
    assert "STREAMING CONTINUAL LEARNING" in out
    assert out.isascii()                          # ASCII-only discipline


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
