"""
mt_lnn/attractor_ops.py -- composable attractor / self-stabilization operators:
how a settling dynamical system converges, and how robust that convergence is.

Why this module exists
----------------------
The defining object of this architecture is a *continuous-time* recurrent core:
``ProtofilamentLTC`` evolves ``h_t = e^{-dt/tau} h_{t-1} + ...``, a contraction
toward a state-dependent equilibrium. The whole "settle into a stable
representation" story -- memory recall relaxing to a stored pattern, an
imagination rollout coasting to a fixed point, a place code locking onto a
basin -- is a statement about **attractors**: where the dynamics settle, how
fast, and how large a perturbation they can absorb before falling into a
different basin. None of the existing layers actually *measures* that. This
module supplies the missing dynamical-systems instrumentation as composable,
model-free operators.

What it measures
----------------
Two complementary views, cross-checked against each other:

* **Analytic (a known linear map** ``x -> A x + b``**)** -- everything is
  closed-form and exact:
    - :func:`fixed_point`        the equilibrium ``x* = (I - A)^{-1} b``;
    - :func:`spectral_radius`    ``rho(A)`` = the slowest mode's gain;
    - :func:`asymptotic_rate`    ``-log rho`` = the per-step convergence rate
      (positive iff it is a contraction, i.e. ``rho < 1``);
    - :func:`is_contraction`     does it settle at all?

* **Empirical (any observed settling trajectory, incl. the nonlinear LTC)** --
  estimated directly from the state sequence, so it works on the real core:
    - :func:`relax`               iterate a step map into a trajectory;
    - :func:`settling_time`       first step inside a tolerance band;
    - :func:`convergence_rate`    log-distance slope -> the empirical rate
      (equals :func:`asymptotic_rate` for a linear map -- pinned in tests);
    - :func:`lyapunov_descent`    per-step energy ratios ``V_{t+1}/V_t`` with
      ``V = ||x - x*||^2``; all ``< 1`` is a discrete Lyapunov stability
      certificate (monotone energy descent).

* **Robustness** -- :func:`basin_radius`: bisection on the perturbation size to
  find the largest push (along a direction) from which the system still returns
  to the attractor. For a global contraction this is unbounded (returns the
  probe cap); for a system with a nearby unstable boundary it recovers that
  boundary (e.g. the basin of ``x=0`` under ``xdot = -x + x^3`` is ``(-1, 1)``,
  and the probe returns ``1`` -- pinned in tests).

:class:`AttractorReport` / :func:`analyze_linear_attractor` compose these into a
single diagnostic and confirm the empirical measures match the closed form.

Design / coupling
-----------------
* Pure functions + a frozen :class:`AttractorReport` dataclass. **Stateless,
  zero trainable parameters**; nothing is an ``nn.Module``. Imports only
  ``torch`` / ``dataclasses`` / ``math`` / ``typing``; **never imports
  model.py**. The step map handed to :func:`relax` / :func:`basin_radius` is an
  arbitrary callable ``x -> x`` (a vector field), so this stays a model-free
  primitive that can wrap the real core later without depending on it.
* The linear analytics and ``relax`` are differentiable; ``settling_time`` and
  ``basin_radius`` involve thresholds / bisection and are piecewise-constant --
  documented, not hidden.

Pinned against the analytic ground truth in ``tests/test_attractor_ops.py`` and
over the whole input space in ``tests/test_attractor_ops_properties.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch

__all__ = [
    "fixed_point",
    "spectral_radius",
    "asymptotic_rate",
    "is_contraction",
    "linear_step",
    "relax",
    "settling_time",
    "convergence_rate",
    "lyapunov_descent",
    "basin_radius",
    "AttractorReport",
    "analyze_linear_attractor",
]


# --------------------------------------------------------------------------- #
# linear-map analytics (exact, closed form)                                   #
# --------------------------------------------------------------------------- #
def fixed_point(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The equilibrium ``x*`` of the affine map ``x -> A x + b``.

    Solves ``(I - A) x* = b``. Raises ``ValueError`` if ``I - A`` is singular
    (a unit eigenvalue -- no isolated fixed point). Differentiable in ``A, b``.
    """
    A = torch.as_tensor(A)
    b = torch.as_tensor(b)
    n = A.shape[-1]
    eye = torch.eye(n, dtype=A.dtype, device=A.device)
    M = eye - A
    if torch.linalg.matrix_rank(M) < n:
        raise ValueError("I - A is singular (unit eigenvalue): no isolated fixed point")
    return torch.linalg.solve(M, b)


def spectral_radius(A: torch.Tensor) -> torch.Tensor:
    """The spectral radius ``rho(A) = max_i |lambda_i|`` (slowest-mode gain).

    A linear map is a contraction (settles to its fixed point from every start)
    iff ``rho(A) < 1``. Returns a real scalar tensor.
    """
    A = torch.as_tensor(A)
    eig = torch.linalg.eigvals(A)             # complex
    return eig.abs().max()


def asymptotic_rate(A: torch.Tensor) -> torch.Tensor:
    """The asymptotic convergence rate ``-log rho(A)`` per step.

    ``> 0`` for a contraction (distance to ``x*`` shrinks like ``e^{-rate t}``),
    ``0`` at the stability boundary, ``< 0`` for a divergent map. Returns a
    scalar tensor.
    """
    rho = spectral_radius(A)
    return -torch.log(rho)


def is_contraction(A: torch.Tensor, *, tol: float = 0.0) -> bool:
    """Whether ``rho(A) < 1 - tol`` -- i.e. the map provably settles."""
    return bool(spectral_radius(A) < (1.0 - tol))


# --------------------------------------------------------------------------- #
# rollout of an arbitrary step map                                            #
# --------------------------------------------------------------------------- #
def linear_step(A: torch.Tensor, b: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the affine step map ``x -> A x + b`` as a callable for :func:`relax`."""
    A = torch.as_tensor(A)
    b = torch.as_tensor(b)

    def step(x: torch.Tensor) -> torch.Tensor:
        return A @ x + b

    return step


def relax(step: Callable[[torch.Tensor], torch.Tensor], x0: torch.Tensor,
          steps: int) -> torch.Tensor:
    """Iterate ``step`` from ``x0`` for ``steps`` steps; return the trajectory.

    ``step`` is any pure map ``x -> x`` (a discrete vector field). The result is
    a ``(steps + 1, n)`` tensor stacking ``[x0, x1, ..., x_steps]``.
    Differentiable through a differentiable ``step``.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative")
    x = torch.as_tensor(x0)
    traj = [x]
    for _ in range(steps):
        x = step(x)
        traj.append(x)
    return torch.stack(traj, dim=0)


# --------------------------------------------------------------------------- #
# empirical measures from an observed trajectory                              #
# --------------------------------------------------------------------------- #
def _distances(trajectory: torch.Tensor, target: Optional[torch.Tensor]) -> torch.Tensor:
    traj = torch.as_tensor(trajectory)
    if traj.dim() == 1:
        traj = traj.unsqueeze(-1)
    tgt = traj[-1] if target is None else torch.as_tensor(target)
    return (traj - tgt).norm(dim=-1)


def settling_time(trajectory: torch.Tensor, *, target: Optional[torch.Tensor] = None,
                  tol: float = 1e-3, relative: bool = False) -> int:
    """First step index at which the trajectory has entered (and stays in) the
    tolerance band around ``target`` (default: the trajectory's endpoint).

    With ``relative=True`` the band is ``tol * ||x0 - target||``; otherwise it is
    the absolute ``tol``. Returns ``len(trajectory) - 1`` if it never settles.
    Piecewise-constant in the inputs (a threshold crossing).
    """
    d = _distances(trajectory, target)
    band = tol * d[0] if relative else tol
    inside = d <= band
    if not bool(inside.any()):
        return int(d.shape[0] - 1)
    # first index from which ALL subsequent steps stay inside the band
    stays = torch.flip(torch.cumprod(torch.flip(inside.to(torch.int8), [0]), 0), [0])
    idx = int((stays > 0).to(torch.int8).argmax())
    return idx


def convergence_rate(trajectory: torch.Tensor, *,
                     target: Optional[torch.Tensor] = None,
                     eps: float = 1e-12) -> torch.Tensor:
    """Empirical asymptotic convergence rate from a settling trajectory.

    Fits ``log ||x_t - target||`` against ``t`` by least squares (over the steps
    whose distance exceeds ``eps``) and returns ``-slope``. For a geometric decay
    ``d_t = d_0 rho^t`` this is exactly ``-log rho = asymptotic_rate`` of the
    underlying linear map (cross-checked in the tests). Returns a scalar tensor;
    differentiable in the trajectory.
    """
    d = _distances(trajectory, target)
    mask = d > eps
    if int(mask.sum()) < 2:
        return torch.zeros((), dtype=d.dtype, device=d.device)
    t = torch.arange(d.shape[0], dtype=d.dtype, device=d.device)[mask]
    ld = torch.log(d[mask])
    tm = t - t.mean()
    slope = (tm * (ld - ld.mean())).sum() / (tm * tm).sum()
    return -slope


def lyapunov_descent(trajectory: torch.Tensor, *,
                     target: Optional[torch.Tensor] = None,
                     eps: float = 1e-12) -> torch.Tensor:
    """Per-step energy ratios ``V_{t+1} / V_t`` with ``V_t = ||x_t - target||^2``.

    A returned vector with every entry ``< 1`` certifies monotone energy descent
    -- a discrete Lyapunov stability certificate for the settling. Steps where
    ``V_t`` underflows ``eps`` are reported as ratio ``0`` (already settled).
    Returns a length-``(T-1)`` tensor.

    Honest caveat: this is the *Euclidean* energy, so the all-``< 1`` certificate
    is sufficient but **not** necessary for stability. A NON-NORMAL contraction
    (``rho < 1`` but ``A^T A != A A^T``) can grow ``V`` transiently before it
    decays -- the asymptotic guarantee is still ``rho(A)`` (see
    :func:`asymptotic_rate`). For a normal/symmetric map every ratio is bounded
    by ``rho^2`` and descent is monotone.
    """
    d = _distances(trajectory, target)
    V = d * d
    num = V[1:]
    den = V[:-1]
    safe = den > eps
    ratios = torch.zeros_like(num)
    ratios[safe] = num[safe] / den[safe]
    return ratios


# --------------------------------------------------------------------------- #
# basin of attraction (robustness probe)                                      #
# --------------------------------------------------------------------------- #
def basin_radius(step: Callable[[torch.Tensor], torch.Tensor], center: torch.Tensor,
                 direction: torch.Tensor, *, max_radius: float = 1.0,
                 steps: int = 2000, tol: float = 1e-3, n_iter: int = 40) -> torch.Tensor:
    """Largest perturbation along ``direction`` from which ``step`` still returns
    to ``center`` -- the basin half-width in that direction.

    Probes ``x0 = center + alpha * dir_unit`` for ``alpha`` in ``[0, max_radius]``
    and calls ``x0`` *converged* if, after ``steps`` iterations, it lies within
    ``tol`` of ``center``. If the cap ``max_radius`` itself converges the basin
    is (at least) unbounded in this direction and ``max_radius`` is returned;
    otherwise a ``n_iter``-step bisection localises the boundary. Deterministic;
    piecewise-constant (a boundary search), not differentiable.
    """
    center = torch.as_tensor(center)
    direction = torch.as_tensor(direction)
    norm = direction.norm()
    if float(norm) == 0.0:
        raise ValueError("direction must be non-zero")
    unit = direction / norm

    def converges(alpha: float) -> bool:
        x = center + alpha * unit
        for _ in range(steps):
            x = step(x)
            if not torch.isfinite(x).all():
                return False
        return bool((x - center).norm() <= tol)

    if not converges(0.0):
        # the centre itself is not a (tol-)attractor under this map
        return torch.zeros((), dtype=center.dtype, device=center.device)
    if converges(max_radius):
        return torch.as_tensor(max_radius, dtype=center.dtype, device=center.device)

    lo, hi = 0.0, float(max_radius)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if converges(mid):
            lo = mid
        else:
            hi = mid
    return torch.as_tensor(lo, dtype=center.dtype, device=center.device)


# --------------------------------------------------------------------------- #
# flagship: compose into a single diagnostic                                  #
# --------------------------------------------------------------------------- #
@dataclass
class AttractorReport:
    """Outcome of analysing a linear attractor ``x -> A x + b``.

    Attributes
    ----------
    fixed_point : (n,)
        The equilibrium ``x*``.
    spectral_radius : scalar tensor
        ``rho(A)``.
    is_contraction : bool
        ``rho(A) < 1`` -- whether it settles at all.
    asymptotic_rate : scalar tensor
        Analytic ``-log rho`` convergence rate per step.
    empirical_rate : scalar tensor
        Convergence rate fitted from the relaxation trajectory (matches
        ``asymptotic_rate`` for a contraction).
    settling_time : int
        Empirical steps to enter the ``tol`` band around ``x*``.
    trajectory : (steps + 1, n)
        The relaxation trajectory from ``x0``.
    """

    fixed_point: torch.Tensor
    spectral_radius: torch.Tensor
    is_contraction: bool
    asymptotic_rate: torch.Tensor
    empirical_rate: torch.Tensor
    settling_time: int
    trajectory: torch.Tensor


def analyze_linear_attractor(A: torch.Tensor, b: torch.Tensor, *,
                             x0: Optional[torch.Tensor] = None,
                             tol: float = 1e-3, max_steps: int = 2000) -> AttractorReport:
    """Full attractor diagnostic for the affine map ``x -> A x + b``.

    Computes the fixed point / spectral radius / analytic rate, relaxes from
    ``x0`` (default: the origin) for up to ``max_steps``, and reads off the
    empirical convergence rate and settling time toward ``x*``. The empirical and
    analytic rates agree for a contraction (pinned in the tests).
    """
    A = torch.as_tensor(A)
    b = torch.as_tensor(b)
    xstar = fixed_point(A, b)
    rho = spectral_radius(A)
    rate = asymptotic_rate(A)
    contracts = bool(rho < 1.0)

    start = torch.zeros_like(xstar) if x0 is None else torch.as_tensor(x0)
    traj = relax(linear_step(A, b), start, max_steps)
    emp = convergence_rate(traj, target=xstar)
    st = settling_time(traj, target=xstar, tol=tol)

    return AttractorReport(
        fixed_point=xstar,
        spectral_radius=rho,
        is_contraction=contracts,
        asymptotic_rate=rate,
        empirical_rate=emp,
        settling_time=st,
        trajectory=traj,
    )
