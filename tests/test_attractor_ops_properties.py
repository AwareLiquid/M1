"""
tests/test_attractor_ops_properties.py -- property-based invariants for the
composable attractor / self-stabilization operators.

``tests/test_attractor_ops.py`` pins these against chosen maps; here we pin the
*dynamical-systems laws themselves* with Hypothesis, over a family of random
contractions:

- the fixed point is genuinely fixed (``A x* + b == x*``);
- a diagonally-built contraction has spectral radius = the largest |diagonal|,
  and ``asymptotic_rate = -log rho``;
- ``is_contraction`` agrees with ``rho < 1``;
- relaxing a contraction from any start converges to ``x*``, the empirical rate
  matches ``-log rho``, and the Lyapunov energy never increases;
- a global linear contraction's basin probe always returns the cap;
- the analysis is invariant to relabelling the coordinates (a permutation
  similarity leaves the spectrum, rate, and settling behaviour unchanged).

All tensors are explicit float64; the global default dtype is left untouched.

Run:  python -m pytest tests/test_attractor_ops_properties.py -v
"""
import math
import sys

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

sys.path.insert(0, ".")

from mt_lnn.attractor_ops import (                                 # noqa: E402
    asymptotic_rate,
    basin_radius,
    convergence_rate,
    fixed_point,
    is_contraction,
    linear_step,
    lyapunov_descent,
    relax,
    spectral_radius,
)

PROP = settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

_F = dict(allow_nan=False, allow_infinity=False, width=64)
_f64 = torch.float64


def _t(x):
    return torch.tensor(x, dtype=_f64)


# --------------------------------------------------------------------------- #
# strategies                                                                  #
# --------------------------------------------------------------------------- #
@st.composite
def contraction(draw, max_n=4):
    """A random diagonalisable contraction A = P diag(d) P^{-1} with rho < 1,
    paired with an offset b. Built from |d_i| <= 0.95 so it provably settles."""
    n = draw(st.integers(1, max_n))
    diag = draw(st.lists(st.floats(-0.95, 0.95, **_F), min_size=n, max_size=n))
    d = _t(diag)
    # a well-conditioned similarity so eigenvalues == diag entries but A is dense
    P = torch.eye(n, dtype=_f64) + 0.1 * torch.arange(1.0, n * n + 1.0).reshape(n, n)
    A = P @ torch.diag(d) @ torch.linalg.inv(P)
    b = _t(draw(st.lists(st.floats(-5.0, 5.0, **_F), min_size=n, max_size=n)))
    return A, b, d


@st.composite
def symmetric_contraction(draw, max_n=4):
    """A random SYMMETRIC contraction with rho = 0.9. Symmetric => normal, so the
    Euclidean energy ``||x - x*||^2`` IS a valid Lyapunov function (it descends
    monotonically). For a non-normal contraction the Euclidean energy can grow
    transiently even though rho < 1 -- so the monotone-descent law is pinned only
    here, where it genuinely holds."""
    n = draw(st.integers(1, max_n))
    vals = draw(st.lists(st.floats(-1.0, 1.0, **_F), min_size=n * n, max_size=n * n))
    M = _t(vals).reshape(n, n)
    S = 0.5 * (M + M.T)
    rho = float(spectral_radius(S))
    if rho < 1e-3:                                   # avoid the (near-)zero matrix
        S = S + torch.eye(n, dtype=_f64)
        rho = float(spectral_radius(S))
    A = 0.9 * S / rho                                # symmetric, rho(A) == 0.9
    b = _t(draw(st.lists(st.floats(-5.0, 5.0, **_F), min_size=n, max_size=n)))
    return A, b


# --------------------------------------------------------------------------- #
# fixed point + spectrum                                                      #
# --------------------------------------------------------------------------- #
@PROP
@given(data=contraction())
def test_fixed_point_is_fixed(data):
    A, b, _ = data
    xstar = fixed_point(A, b)
    assert torch.allclose(A @ xstar + b, xstar, atol=1e-6)


@PROP
@given(data=contraction())
def test_spectral_radius_equals_largest_eigenvalue_magnitude(data):
    A, _, d = data
    expect = float(d.abs().max())
    assert torch.isclose(spectral_radius(A), _t(expect), atol=1e-6)


@PROP
@given(data=contraction())
def test_rate_is_minus_log_rho_and_contraction_flag_agrees(data):
    A, _, _ = data
    rho = float(spectral_radius(A))
    assert is_contraction(A) == (rho < 1.0)
    if rho < 1e-3:                                   # -log(rho) explodes at rho->0
        return
    assert torch.isclose(asymptotic_rate(A), _t(-math.log(rho)), atol=1e-6)


# --------------------------------------------------------------------------- #
# relaxation                                                                  #
# --------------------------------------------------------------------------- #
@PROP
@given(data=contraction(),
       start=st.lists(st.floats(-3.0, 3.0, **_F), min_size=1, max_size=4))
def test_relaxation_converges_to_the_fixed_point(data, start):
    A, b, _ = data
    n = A.shape[0]
    x0 = _t((start * n)[:n])
    xstar = fixed_point(A, b)
    traj = relax(linear_step(A, b), x0, 400)
    assert torch.allclose(traj[-1], xstar, atol=1e-3)


@PROP
@given(data=symmetric_contraction(),
       start=st.lists(st.floats(-3.0, 3.0, **_F), min_size=1, max_size=4))
def test_lyapunov_energy_never_increases_for_a_symmetric_contraction(data, start):
    A, b = data
    n = A.shape[0]
    x0 = _t((start * n)[:n])
    xstar = fixed_point(A, b)
    traj = relax(linear_step(A, b), x0, 200)
    ratios = lyapunov_descent(traj, target=xstar)
    # symmetric + rho = 0.9 => every step contracts the Euclidean energy by <= 0.81
    assert bool((ratios <= 0.81 + 1e-6).all())


@PROP
@given(data=contraction())
def test_empirical_rate_matches_asymptotic_rate(data):
    A, b, d = data
    # need a genuine gap to the origin: skip the (measure-zero) all-tiny case
    if float(d.abs().max()) < 0.2:
        return
    xstar = fixed_point(A, b)
    # Excite the DOMINANT (slowest) eigenmode, not a uniform offset. The
    # asymptotic rate is by definition the decay of the slowest mode -- a uniform
    # offset also excites faster modes whose transient biases the finite-window
    # slope steeper, and for a small dominant rho the trajectory underflows to eps
    # before that transient clears, so the bias is never washed out. Starting on
    # the dominant eigenvector makes the error a PURE geometric ||e_t|| = c*rho^t
    # (the regime convergence_rate documents), so the empirical slope recovers
    # -log rho to floating precision. (A is real-diagonalisable here, so its
    # spectrum is real and the dominant eigenvector is real.)
    evals, evecs = torch.linalg.eig(A)
    v = evecs[:, int(evals.abs().argmax())].real
    if float(v.norm()) < 1e-8:           # defensive: no real dominant direction
        return
    x0 = xstar + v / v.norm()
    traj = relax(linear_step(A, b), x0, 400)
    emp = convergence_rate(traj, target=xstar)
    assert torch.isclose(emp, asymptotic_rate(A), atol=1e-3)


# --------------------------------------------------------------------------- #
# basin + relabelling invariance                                             #
# --------------------------------------------------------------------------- #
@PROP
@given(data=contraction())
def test_global_contraction_basin_hits_the_cap(data):
    A, b, _ = data
    xstar = fixed_point(A, b)
    direction = torch.ones(A.shape[0], dtype=_f64)
    cap = 3.0
    r = basin_radius(linear_step(A, b), xstar, direction, max_radius=cap, steps=600)
    assert float(r) == cap


@PROP
@given(data=contraction())
def test_spectrum_is_invariant_under_coordinate_permutation(data):
    A, b, _ = data
    n = A.shape[0]
    perm = torch.randperm(n)
    Pmat = torch.eye(n, dtype=_f64)[perm]
    Ap = Pmat @ A @ Pmat.T               # similarity by a permutation
    assert torch.isclose(spectral_radius(Ap), spectral_radius(A), atol=1e-6)
    assert torch.isclose(asymptotic_rate(Ap), asymptotic_rate(A), atol=1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
