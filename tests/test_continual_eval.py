"""
tests/test_continual_eval.py -- continual-learning metrics, pinned against
hand-computed accuracy matrices with known forgetting behaviour.

Three canonical regimes anchor every metric:
* perfect retention (lower triangle == diagonal): BWT == 0, FM == 0;
* total forgetting (old tasks collapse to 0 by the end): BWT and FM exact;
* positive backward transfer (later learning lifts earlier tasks): BWT > 0,
  FM == 0 (FM never rewards transfer).
Plus the literal formula checks for ACC / LA / FWT and the validation contract.
"""
import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mt_lnn.continual_eval import (  # noqa: E402
    ContinualReport,
    as_accuracy_matrix,
    average_accuracy,
    backward_transfer,
    evaluate_continual,
    forgetting,
    forward_transfer,
    learning_accuracy,
)


def _close(x, v, tol=1e-6):
    # 1e-6 (not 1e-9): some inputs arrive as float32 tensors, whose values are
    # already rounded before we upcast -- accuracy scores do not need more.
    return abs(float(x) - v) < tol


# --------------------------------------------------------------------------- #
# perfect retention                                                          #
# --------------------------------------------------------------------------- #
def test_perfect_retention_has_no_forgetting():
    # learned each task to 0.9 and never lost a thing
    R = [
        [0.90, 0.00, 0.00],
        [0.90, 0.90, 0.00],
        [0.90, 0.90, 0.90],
    ]
    assert _close(average_accuracy(R), 0.90)
    assert _close(learning_accuracy(R), 0.90)
    assert _close(backward_transfer(R), 0.0)
    assert _close(forgetting(R), 0.0)


# --------------------------------------------------------------------------- #
# total forgetting                                                           #
# --------------------------------------------------------------------------- #
def test_total_forgetting_is_quantified_exactly():
    # each task learned to 1.0, then wiped to 0.0 once the next arrives
    R = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    # final row: tasks 0,1 at 0.0, task 2 at 1.0 -> ACC = 1/3
    assert _close(average_accuracy(R), 1.0 / 3.0)
    assert _close(learning_accuracy(R), 1.0)
    # BWT = mean(R[2,0]-R[0,0], R[2,1]-R[1,1]) = mean(-1, -1) = -1
    assert _close(backward_transfer(R), -1.0)
    # FM = mean(best-final) for tasks 0,1 = mean(1-0, 1-0) = 1
    assert _close(forgetting(R), 1.0)


# --------------------------------------------------------------------------- #
# positive backward transfer                                                 #
# --------------------------------------------------------------------------- #
def test_positive_backward_transfer_lifts_old_tasks():
    # task 0 learned to 0.6, later training RAISES it to 0.8
    R = [
        [0.60, 0.00],
        [0.80, 0.70],
    ]
    # BWT = R[1,0]-R[0,0] = 0.8-0.6 = +0.2
    assert _close(backward_transfer(R), 0.2)
    # FM ignores positive transfer: best(task0 while old) = R[0,0]=0.6, final=0.8
    # -> drop = 0.6-0.8 = -0.2, but FM clamps to "loss only"? No: FM = best-final
    #   where best is over l in [i, T-1) = just R[0,0]=0.6 -> 0.6-0.8 = -0.2.
    # The standard FM can go negative under positive transfer; we report it as-is.
    assert _close(forgetting(R), -0.2)


def test_forgetting_uses_best_ever_not_just_learned():
    # task 0: learned 0.5, bumped to 0.9 mid-way, then dropped to 0.4 at the end.
    # best-ever (over old-task snapshots rows 0..T-2) = max(0.5, 0.9) = 0.9
    # final = 0.4 -> FM contribution = 0.9 - 0.4 = 0.5
    R = [
        [0.5, 0.0, 0.0],
        [0.9, 0.6, 0.0],
        [0.4, 0.3, 0.7],
    ]
    # tasks: i=0 best=max(0.5,0.9)=0.9 final=0.4 drop=0.5
    #        i=1 best=max(0.6)=0.6 final=0.3 drop=0.3
    assert _close(forgetting(R), (0.5 + 0.3) / 2.0)


# --------------------------------------------------------------------------- #
# forward transfer                                                           #
# --------------------------------------------------------------------------- #
def test_forward_transfer_above_baseline():
    R = [
        [0.90, 0.40, 0.20],   # zero-shot on task1 = 0.40, on task2 (after t0) n/a
        [0.85, 0.88, 0.55],   # zero-shot on task2 = 0.55
        [0.80, 0.84, 0.90],
    ]
    baseline = [0.10, 0.10, 0.10]    # random-init scores
    # FWT = mean(R[0,1]-b1, R[1,2]-b2) = mean(0.40-0.10, 0.55-0.10) = mean(.3,.45)
    assert _close(forward_transfer(R, baseline), (0.30 + 0.45) / 2.0)


# --------------------------------------------------------------------------- #
# one-call bundle + dataclass                                                #
# --------------------------------------------------------------------------- #
def test_evaluate_continual_bundles_everything():
    R = torch.tensor([
        [1.0, 0.0],
        [0.5, 0.9],
    ])
    rep = evaluate_continual(R, baseline=[0.1, 0.2])
    assert isinstance(rep, ContinualReport)
    assert rep.n_tasks == 2
    assert _close(rep.average_accuracy, (0.5 + 0.9) / 2.0)
    assert _close(rep.learning_accuracy, (1.0 + 0.9) / 2.0)
    assert _close(rep.backward_transfer, 0.5 - 1.0)        # -0.5
    assert _close(rep.forgetting, 1.0 - 0.5)               # +0.5
    assert _close(rep.forward_transfer, 0.0 - 0.2)         # R[0,1]-b1 = 0-0.2
    with pytest.raises(Exception):
        rep.n_tasks = 3                                     # frozen


def test_bundle_without_baseline_has_no_fwt():
    rep = evaluate_continual([[0.8, 0.0], [0.7, 0.9]])
    assert rep.forward_transfer is None


# --------------------------------------------------------------------------- #
# single-task edge case + validation                                         #
# --------------------------------------------------------------------------- #
def test_single_task_transfer_metrics_are_zero():
    R = [[0.7]]
    assert _close(backward_transfer(R), 0.0)
    assert _close(forgetting(R), 0.0)
    assert _close(forward_transfer(R, [0.1]), 0.0)
    assert _close(average_accuracy(R), 0.7)
    assert _close(learning_accuracy(R), 0.7)


def test_validation_rejects_bad_shapes():
    with pytest.raises(ValueError):
        as_accuracy_matrix([[0.1, 0.2, 0.3]])        # not square
    with pytest.raises(ValueError):
        as_accuracy_matrix(torch.zeros(2, 2, 2))     # not 2-D
    with pytest.raises(ValueError):
        as_accuracy_matrix(torch.zeros(0, 0))        # empty
    with pytest.raises(ValueError):
        forward_transfer([[1.0, 0.0], [0.5, 0.9]], baseline=[0.1])  # wrong len


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
