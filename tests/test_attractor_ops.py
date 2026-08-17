"""
tests/test_attractor_ops.py -- composable attractor / self-stabilization
operators pinned against the analytic dynamical-systems ground truth.

A linear map ``x -> A x + b`` has a closed-form fixed point, spectral radius,
convergence rate, and (geometric) settling behaviour; a trajectory relaxing into
it has a known log-distance slope and Lyapunov descent. A nonlinear map with a
known unstable boundary has a known basin width. These fix all of that by hand.
``tests/test_attractor_ops_properties.py`` pins the laws over the input space.

All tensors are explicit float64; the global default dtype is left untouched.

Run:  python -m pytest tests/test_attractor_ops.py -v
"""
import math
import sys

import pytest
import torch

sys.path.insert(0, ".")

from mt_lnn.attractor_ops import (                                 # noqa: E402
    AttractorReport,
    analyze_linear_attractor,
    asymptotic_rate,
    basin_radius,
    convergence_rate,
    fixed_point,
    is_contraction,
    linear_step,
    lyapunov_descent,
    relax,
    settling_time,
    spectral_radius,
)

_f64 = torch.float64


def _t(x):
    return torch.tensor(x, dtype=_f64)


_A = torch.diag(_t([0.5, 0.9]))          # contraction, rho = 0.9
_B = _t([1.0, 2.0])                      # -> fixed point (2, 20)


# --------------------------------------------------------------------------- #
# linear analytics                                                            #
# --------------------------------------------------------------------------- #
def test_fixed_point_solves_the_affine_equation():
    xstar = fixed_point(_A, _B)
    assert torch.allclose(xstar, _t([2.0, 20.0]), atol=1e-10)
    # x* is genuinely fixed: A x* + b == x*
    assert torch.allclose(_A @ xstar + _B, xstar, atol=1e-10)


def test_fixed_point_rejects_unit_eigenvalue():
    A = torch.eye(2, dtype=_f64)          # I - A == 0, singular
    with pytest.raises(ValueError):
        fixed_point(A, _B)


def test_spectral_radius_is_the_slowest_mode():
    assert torch.isclose(spectral_radius(_A), _t(0.9), atol=1e-12)
    # a rotation-scaling block has spectral radius == the scale
    r, theta = 0.7, 0.4
    R = _t([[r * math.cos(theta), -r * math.sin(theta)],
            [r * math.sin(theta), r * math.cos(theta)]])
    assert torch.isclose(spectral_radius(R), _t(r), atol=1e-12)


def test_asymptotic_rate_is_minus_log_rho():
    assert torch.isclose(asymptotic_rate(_A), _t(-math.log(0.9)), atol=1e-12)


def test_is_contraction_thresholds_at_unit_spectral_radius():
    assert is_contraction(_A)
    assert not is_contraction(torch.diag(_t([1.2, 0.5])))
    assert not is_contraction(torch.eye(2, dtype=_f64))    # rho == 1 exactly


# --------------------------------------------------------------------------- #
# rollout + empirical measures                                                #
# --------------------------------------------------------------------------- #
def test_relax_reproduces_the_geometric_trajectory():
    x0 = _t([0.0, 0.0])
    traj = relax(linear_step(_A, _B), x0, 5)
    assert traj.shape == (6, 2)
    xstar = fixed_point(_A, _B)
    # closed form: x_t = x* + A^t (x0 - x*)
    for t in range(6):
        expect = xstar + torch.matrix_power(_A, t) @ (x0 - xstar)
        assert torch.allclose(traj[t], expect, atol=1e-10)


def test_convergence_rate_matches_minus_log_rho():
    traj = relax(linear_step(_A, _B), _t([0.0, 0.0]), 300)
    emp = convergence_rate(traj, target=fixed_point(_A, _B))
    assert torch.isclose(emp, asymptotic_rate(_A), atol=1e-4)


def test_settling_time_matches_the_geometric_estimate():
    xstar = fixed_point(_A, _B)
    x0 = _t([0.0, 0.0])
    traj = relax(linear_step(_A, _B), x0, 2000)
    tol = 1e-3
    st = settling_time(traj, target=xstar, tol=tol)
    d0 = float((x0 - xstar).norm())
    analytic = math.ceil(math.log(tol / d0) / math.log(0.9))
    assert abs(st - analytic) <= 1                    # off-by-one from ceil/band


def test_lyapunov_descent_certifies_monotone_settling():
    traj = relax(linear_step(_A, _B), _t([0.0, 0.0]), 100)
    ratios = lyapunov_descent(traj, target=fixed_point(_A, _B))
    assert ratios.shape == (100,)
    # the worst-case per-step energy ratio is bounded by rho^2 = 0.81
    nz = ratios[ratios > 0]
    assert float(nz.max()) <= 0.81 + 1e-6
    assert bool((ratios <= 1.0 + 1e-9).all())         # never increases


def test_lyapunov_increases_for_a_divergent_map():
    A = torch.diag(_t([1.3, 1.1]))
    traj = relax(linear_step(A, _t([0.0, 0.0])), _t([1.0, 1.0]), 10)
    ratios = lyapunov_descent(traj, target=_t([0.0, 0.0]))
    assert float(ratios.min()) > 1.0                  # energy grows every step


# --------------------------------------------------------------------------- #
# basin of attraction                                                         #
# --------------------------------------------------------------------------- #
def test_basin_radius_finds_the_unstable_boundary():
    # xdot = -x + x^3 : stable at 0, unstable at +-1 -> basin of 0 is (-1, 1)
    dt = 0.01

    def step(x):
        return x + dt * (-x + x ** 3)

    r = basin_radius(step, _t([0.0]), _t([1.0]), max_radius=2.0, steps=4000)
    assert abs(float(r) - 1.0) < 1e-2


def test_basin_radius_is_capped_for_a_global_contraction():
    # a linear contraction settles from everywhere -> basin is unbounded
    cap = 5.0
    r = basin_radius(linear_step(_A, _B), fixed_point(_A, _B), _t([1.0, 0.0]),
                     max_radius=cap, steps=500)
    assert float(r) == cap


def test_basin_radius_zero_when_centre_is_not_an_attractor():
    # a divergent map: even alpha=0 (the centre itself) does not stay put
    A = torch.diag(_t([1.2, 1.2]))
    r = basin_radius(linear_step(A, _t([0.0, 0.0])), _t([1.0, 1.0]), _t([1.0, 0.0]),
                     max_radius=2.0, steps=200, tol=1e-3)
    assert float(r) == 0.0


def test_basin_radius_rejects_zero_direction():
    with pytest.raises(ValueError):
        basin_radius(linear_step(_A, _B), fixed_point(_A, _B), _t([0.0, 0.0]))


# --------------------------------------------------------------------------- #
# flagship report                                                             #
# --------------------------------------------------------------------------- #
def test_analyze_linear_attractor_composes_consistently():
    rep = analyze_linear_attractor(_A, _B, tol=1e-3)
    assert isinstance(rep, AttractorReport)
    assert torch.allclose(rep.fixed_point, _t([2.0, 20.0]), atol=1e-10)
    assert torch.isclose(rep.spectral_radius, _t(0.9), atol=1e-12)
    assert rep.is_contraction
    assert torch.isclose(rep.empirical_rate, rep.asymptotic_rate, atol=1e-4)
    assert rep.settling_time > 0
    assert rep.trajectory.shape[1] == 2


def test_analyze_flags_a_divergent_map():
    rep = analyze_linear_attractor(torch.diag(_t([1.5, 0.2])), _t([0.0, 0.0]),
                                   x0=_t([1.0, 1.0]), max_steps=20)
    assert not rep.is_contraction
    assert float(rep.asymptotic_rate) < 0.0           # negative rate = blows up


def test_relax_rejects_negative_steps():
    with pytest.raises(ValueError):
        relax(linear_step(_A, _B), _t([0.0, 0.0]), -1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
