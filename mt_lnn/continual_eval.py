"""
mt_lnn/continual_eval.py -- composable continual-learning metrics: how much a
model forgets, retains, and transfers across a task sequence (P2 evaluation).

Why this module exists
----------------------
Training MT-LNN as a *continual* learner -- on a sequence of tasks, one after
another, without revisiting old data freely -- only matters if we can *measure*
the thing it is supposed to fight: catastrophic forgetting. "Final accuracy is
high" is not enough; a model can score well on the last task while having wiped
out the first. The field settled on a small set of metrics computed from one
object, the **accuracy matrix**, and this module computes exactly those, so a
streaming-training run and (later) the theory paper share one definition.

The accuracy matrix
-------------------
After training on each task in turn we evaluate on *every* task. This fills a
``T x T`` matrix ``R`` with::

    R[i, j] = test accuracy on task j  AFTER finishing training on task i.

The diagonal ``R[i, i]`` is "just-learned" accuracy; the last row ``R[T-1, :]``
is end-of-training accuracy on everything; entries below the diagonal show how an
old task fared as new tasks arrived; the super-diagonal ``R[i-1, i]`` is zero-shot
accuracy on a task the model has not yet trained on.

The metrics (Lopez-Paz & Ranzato 2017; Chaudhry et al. 2018)
------------------------------------------------------------
* **Average / final accuracy** ``ACC = mean_j R[T-1, j]`` -- end-of-training
  accuracy averaged over all tasks (the headline number).
* **Learning accuracy** ``LA = mean_i R[i, i]`` -- how well each task was learned
  the moment it was trained (the plasticity ceiling, before any forgetting).
* **Backward transfer** ``BWT = mean_{i<T-1} (R[T-1, i] - R[i, i])`` -- the net
  effect of later learning on earlier tasks. ``BWT < 0`` is forgetting;
  ``BWT > 0`` is positive backward transfer (later tasks *helped* earlier ones).
* **Forgetting measure** ``FM = mean_{i<T-1} (max_{l>=i, l<T-1} R[l, i] - R[T-1, i])``
  -- the drop from each task's *best-ever* score to its final score. ``FM >= 0``;
  larger is worse. (Unlike BWT this never rewards, only penalises -- it ignores
  positive transfer and isolates pure loss.)
* **Forward transfer** ``FWT = mean_{i>=1} (R[i-1, i] - b_i)`` -- zero-shot
  accuracy on task ``i`` (just before it is trained) above a random-init baseline
  ``b_i``. Requires the baseline vector; omitted if not supplied.

Design / coupling
-----------------
* Pure functions + a frozen ``ContinualReport`` dataclass. **Stateless, zero
  trainable parameters**, nothing is an ``nn.Module``; imports only ``torch`` /
  ``dataclasses`` / ``typing``; **never imports model.py**. The matrix ``R`` is
  the entire interface -- where the accuracies came from (this backbone, a
  baseline, a competitor) is the caller's business.
* Only the entries each metric is defined on are read, so the unused
  upper-triangle (future tasks you chose not to evaluate) may be left as NaN /
  zero without corrupting any score.

Pinned against hand-computed matrices (perfect retention, total forgetting,
positive transfer) in ``tests/test_continual_eval.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
from torch import Tensor

__all__ = [
    "ContinualReport",
    "as_accuracy_matrix",
    "average_accuracy",
    "learning_accuracy",
    "backward_transfer",
    "forgetting",
    "forward_transfer",
    "evaluate_continual",
]

MatrixLike = Union[Tensor, Sequence[Sequence[float]]]


@dataclass(frozen=True)
class ContinualReport:
    """A read-only bundle of continual-learning scores for one task sequence.

    Attributes
    ----------
    n_tasks:            number of tasks ``T``.
    average_accuracy:   ``ACC`` -- mean end-of-training accuracy over all tasks.
    learning_accuracy:  ``LA``  -- mean just-learned (diagonal) accuracy.
    backward_transfer:  ``BWT`` -- net effect of later training on earlier tasks
                        (negative = forgetting).
    forgetting:         ``FM``  -- mean best-to-final drop (>= 0, larger worse).
    forward_transfer:   ``FWT`` -- zero-shot transfer above baseline, or ``None``
                        if no baseline was supplied.
    """

    n_tasks: int
    average_accuracy: float
    learning_accuracy: float
    backward_transfer: float
    forgetting: float
    forward_transfer: Optional[float]


# --------------------------------------------------------------------------- #
# input coercion                                                              #
# --------------------------------------------------------------------------- #
def as_accuracy_matrix(R: MatrixLike) -> Tensor:
    """Coerce ``R`` to a square float ``Tensor`` and validate its shape.

    Accepts a tensor or a nested sequence. Raises if it is not 2-D and square.
    """
    # always parse / compute in float64: these are tiny aggregate scores that the
    # theory paper will cite, so exactness beats matching the caller's dtype.
    # Parsing a python list straight to float64 also avoids a lossy float32
    # round-trip (torch.as_tensor([[0.7]]) would otherwise be float32 first).
    M = R.to(torch.float64) if isinstance(R, Tensor) else torch.as_tensor(R, dtype=torch.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"accuracy matrix must be square (T, T), got {tuple(M.shape)}")
    if M.shape[0] == 0:
        raise ValueError("accuracy matrix must have at least one task")
    return M


# --------------------------------------------------------------------------- #
# individual metrics                                                          #
# --------------------------------------------------------------------------- #
def average_accuracy(R: MatrixLike) -> Tensor:
    """``ACC = mean_j R[T-1, j]`` -- final-row mean (end-of-training accuracy)."""
    M = as_accuracy_matrix(R)
    return M[-1].mean()


def learning_accuracy(R: MatrixLike) -> Tensor:
    """``LA = mean_i R[i, i]`` -- mean diagonal (accuracy right after learning)."""
    M = as_accuracy_matrix(R)
    return torch.diagonal(M).mean()


def backward_transfer(R: MatrixLike) -> Tensor:
    """``BWT = mean_{i<T-1} (R[T-1, i] - R[i, i])``.

    Net change to each earlier task's accuracy from the end of training relative
    to when it was first learned. Negative => forgetting; positive => later tasks
    reinforced earlier ones. Returns ``0`` for a single task (no earlier task).
    """
    M = as_accuracy_matrix(R)
    T = M.shape[0]
    if T < 2:
        return M.new_zeros(())
    i = torch.arange(T - 1)
    return (M[-1, i] - M[i, i]).mean()


def forgetting(R: MatrixLike) -> Tensor:
    """``FM = mean_{i<T-1} (max_{i<=l<T-1} R[l, i] - R[T-1, i])``.

    Average drop from each task's best-observed accuracy (over the snapshots
    taken while it was an "old" task) to its final accuracy. Always ``>= 0``;
    isolates pure loss (ignores positive transfer). ``0`` for a single task.
    """
    M = as_accuracy_matrix(R)
    T = M.shape[0]
    if T < 2:
        return M.new_zeros(())
    drops = []
    for i in range(T - 1):
        best = M[i:T - 1, i].max()           # best while task i was "previous"
        drops.append(best - M[T - 1, i])
    return torch.stack(drops).mean()


def forward_transfer(R: MatrixLike, baseline: MatrixLike) -> Tensor:
    """``FWT = mean_{i>=1} (R[i-1, i] - b_i)`` -- zero-shot transfer above baseline.

    ``baseline`` is a length-``T`` vector of random-init accuracies; only entries
    ``1..T-1`` are used (task 0 has nothing before it). Returns ``0`` for a single
    task.
    """
    M = as_accuracy_matrix(R)
    T = M.shape[0]
    b = torch.as_tensor(baseline, dtype=M.dtype).flatten()
    if b.shape[0] != T:
        raise ValueError(f"baseline must have length T={T}, got {b.shape[0]}")
    if T < 2:
        return M.new_zeros(())
    i = torch.arange(1, T)
    return (M[i - 1, i] - b[i]).mean()


# --------------------------------------------------------------------------- #
# one-call bundle                                                             #
# --------------------------------------------------------------------------- #
def evaluate_continual(R: MatrixLike,
                       baseline: Optional[MatrixLike] = None) -> ContinualReport:
    """Compute the full metric bundle from an accuracy matrix in one call.

    ``baseline`` (optional, length ``T``) enables forward transfer; without it
    ``report.forward_transfer is None``.
    """
    M = as_accuracy_matrix(R)
    fwt = None if baseline is None else float(forward_transfer(M, baseline))
    return ContinualReport(
        n_tasks=int(M.shape[0]),
        average_accuracy=float(average_accuracy(M)),
        learning_accuracy=float(learning_accuracy(M)),
        backward_transfer=float(backward_transfer(M)),
        forgetting=float(forgetting(M)),
        forward_transfer=fwt,
    )
