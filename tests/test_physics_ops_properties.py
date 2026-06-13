"""
tests/test_physics_ops_properties.py -- property-based physical invariants for
the Newtonian dynamics operators.

``tests/test_physics_ops.py`` pins these operators against a handful of analytic
scenarios. This file pins the *conservation laws and closed forms themselves*
with Hypothesis, over the whole input space:

- symplectic Euler reproduces its exact closed form under constant acceleration;
- N-body gravity obeys Newton's third law (sum of m*a == 0) and is translation
  invariant;
- sphere collisions conserve momentum always, conserve kinetic energy when
  elastic (e=1), and never *create* energy when inelastic (e<=1);
- box-wall reflection clamps inside the box and preserves speed when elastic;
- a closed N-body rollout conserves total momentum across every snapshot.

All tensors are explicit float64 (the global default dtype is left untouched so
the float32 model tests sharing this process are unaffected). Rollout tests feed
*batched* states so the documented ``PhysicsRollout`` API is exercised on its
happy path.

Run:  python -m pytest tests/test_physics_ops_properties.py -v
"""
import sys

import pytest
import torch
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

sys.path.insert(0, ".")

from mt_lnn.physics_ops import (                                  # noqa: E402
    integrate,
    kinetic_energy,
    momentum,
    pairwise_gravity,
    reflect_in_box,
    resolve_sphere_collisions,
    rollout,
    uniform_gravity,
)

PROP = settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

_F = dict(allow_nan=False, allow_infinity=False, width=64)
_f64 = torch.float64


def _t(rows):
    return torch.tensor(rows, dtype=_f64)


def _close(a, b, atol=1e-6, rtol=1e-6):
    return torch.allclose(a, b, atol=atol, rtol=rtol)


# --------------------------------------------------------------------------- #
# strategies                                                                  #
# --------------------------------------------------------------------------- #
@st.composite
def vecs(draw, n, d, lo, hi):
    rows = draw(st.lists(
        st.lists(st.floats(lo, hi, **_F), min_size=d, max_size=d),
        min_size=n, max_size=n))
    return _t(rows)


@st.composite
def system(draw, min_n=2, max_n=4):
    d = draw(st.sampled_from([2, 3]))
    n = draw(st.integers(min_n, max_n))
    pos = draw(vecs(n, d, -10.0, 10.0))
    vel = draw(vecs(n, d, -5.0, 5.0))
    mass = _t(draw(st.lists(st.floats(0.2, 5.0, **_F), min_size=n, max_size=n)))
    return pos, vel, mass, n, d


# --------------------------------------------------------------------------- #
# integration -- symplectic Euler closed form                                 #
# --------------------------------------------------------------------------- #
@PROP
@given(data=system(), dt=st.floats(0.01, 0.3, **_F), k=st.integers(1, 30))
def test_constant_acceleration_matches_the_symplectic_closed_form(data, dt, k):
    """v_k = v0 + k*dt*a and x_k = x0 + k*dt*v0 + a*dt^2*k(k+1)/2 exactly."""
    pos, vel, _, n, d = data
    a = draw_accel(d)
    x, v = pos, vel
    for _ in range(k):
        x, v = integrate(x, v, a, dt)
    v_pred = vel + k * dt * a
    x_pred = pos + k * dt * vel + a * (dt * dt) * (k * (k + 1) / 2.0)
    assert _close(v, v_pred)
    assert _close(x, x_pred)


def draw_accel(d):
    # a fixed, deterministic acceleration vector of dimension d
    base = torch.tensor([0.5, -3.0, 1.25], dtype=_f64)
    return base[:d].clone()


# --------------------------------------------------------------------------- #
# forces                                                                       #
# --------------------------------------------------------------------------- #
@PROP
@given(data=system())
def test_pairwise_gravity_obeys_newtons_third_law(data):
    """A closed system feels zero net force: sum_i m_i a_i == 0."""
    pos, _, mass, n, d = data
    a = pairwise_gravity(pos, mass)
    net = (mass.view(n, 1) * a).sum(dim=0)
    assert _close(net, torch.zeros(d, dtype=_f64), atol=1e-7)


@PROP
@given(data=system(), shift=st.lists(st.floats(-5.0, 5.0, **_F), min_size=3, max_size=3))
def test_pairwise_gravity_is_translation_invariant(data, shift):
    """Force depends only on separations, so rigidly shifting the scene is a no-op."""
    pos, _, mass, n, d = data
    c = _t(shift)[:d]
    a0 = pairwise_gravity(pos, mass)
    a1 = pairwise_gravity(pos + c, mass)
    assert _close(a0, a1, atol=1e-6, rtol=1e-5)


@PROP
@given(data=system(), g=st.lists(st.floats(-20.0, 20.0, **_F), min_size=3, max_size=3))
def test_uniform_gravity_applies_g_to_every_body(data, g):
    pos, _, _, n, d = data
    gv = _t(g)[:d]
    a = uniform_gravity(pos, gv)
    assert _close(a, gv.expand(n, d))


# --------------------------------------------------------------------------- #
# diagnostics                                                                  #
# --------------------------------------------------------------------------- #
@PROP
@given(data=system(), c=st.floats(-4.0, 4.0, **_F))
def test_kinetic_energy_is_nonneg_and_scales_quadratically(data, c):
    _, vel, mass, _, _ = data
    ke = kinetic_energy(vel, mass)
    assert float(ke) >= -1e-12
    assert _close(kinetic_energy(c * vel, mass), (c * c) * ke)


@PROP
@given(data=system(), c=st.floats(-4.0, 4.0, **_F))
def test_momentum_is_linear_in_velocity(data, c):
    _, vel, mass, _, _ = data
    assert _close(momentum(c * vel, mass), c * momentum(vel, mass))


# --------------------------------------------------------------------------- #
# collisions                                                                   #
# --------------------------------------------------------------------------- #
@PROP
@given(data=system(), radius=st.floats(0.5, 4.0, **_F),
       e=st.floats(0.0, 1.0, **_F))
def test_sphere_collisions_always_conserve_momentum(data, radius, e):
    """Equal-and-opposite impulses => total momentum is invariant, collision or not."""
    pos, vel, mass, _, _ = data
    v2 = resolve_sphere_collisions(pos, vel, radius=radius, mass=mass, restitution=e)
    assert _close(momentum(v2, mass), momentum(vel, mass), atol=1e-6, rtol=1e-5)


@st.composite
def head_on_pair(draw):
    """Two bodies on the x-axis, overlapping and approaching (collision guaranteed)."""
    gap = draw(st.floats(0.1, 1.8, **_F))          # < r0 + r1 = 2.0
    s0 = draw(st.floats(0.1, 5.0, **_F))
    s1 = draw(st.floats(0.1, 5.0, **_F))
    m0 = draw(st.floats(0.2, 5.0, **_F))
    m1 = draw(st.floats(0.2, 5.0, **_F))
    pos = _t([[0.0, 0.0], [gap, 0.0]])
    vel = _t([[s0, 0.0], [-s1, 0.0]])              # moving toward each other
    mass = _t([m0, m1])
    return pos, vel, mass


@PROP
@given(pack=head_on_pair())
def test_elastic_head_on_collision_conserves_energy_and_changes_velocity(pack):
    pos, vel, mass = pack
    v2 = resolve_sphere_collisions(pos, vel, radius=1.0, mass=mass, restitution=1.0)
    assert _close(kinetic_energy(v2, mass), kinetic_energy(vel, mass), atol=1e-6, rtol=1e-6)
    assert _close(momentum(v2, mass), momentum(vel, mass), atol=1e-6, rtol=1e-6)
    assert not _close(v2, vel)                     # the collision actually fired


@PROP
@given(pack=head_on_pair(), e=st.floats(0.0, 1.0, **_F))
def test_inelastic_collision_never_creates_energy(pack, e):
    pos, vel, mass = pack
    v2 = resolve_sphere_collisions(pos, vel, radius=1.0, mass=mass, restitution=e)
    assert float(kinetic_energy(v2, mass)) <= float(kinetic_energy(vel, mass)) + 1e-9


# --------------------------------------------------------------------------- #
# walls                                                                        #
# --------------------------------------------------------------------------- #
@PROP
@given(data=system())
def test_reflect_in_box_clamps_inside_and_elastic_preserves_speed(data):
    pos, vel, _, n, d = data
    lo = _t([-6.0, -6.0, -6.0])[:d]
    hi = _t([6.0, 6.0, 6.0])[:d]
    p2, v2 = reflect_in_box(pos, vel, lo, hi, restitution=1.0)
    assert torch.all(p2 >= lo - 1e-9) and torch.all(p2 <= hi + 1e-9)
    # an elastic wall only flips sign(s), so each component's magnitude is kept
    assert _close(v2.abs(), vel.abs())


# --------------------------------------------------------------------------- #
# rollout (batched -> documented API)                                          #
# --------------------------------------------------------------------------- #
@PROP
@given(data=system(), dt=st.floats(0.01, 0.2, **_F), k=st.integers(1, 20))
def test_force_free_rollout_is_a_straight_line_at_constant_energy(data, dt, k):
    pos, vel, mass, n, d = data
    p0 = pos.unsqueeze(0)                           # batch it: (1, N, D)
    v0 = vel.unsqueeze(0)
    r = rollout(p0, v0, steps=k, dt=dt, mass=mass)  # no gravity, no contacts
    # straight line: x(t) = x0 + v0 * t
    expected = p0 + v0 * (k * dt)
    assert _close(r.final_positions, expected)
    # and kinetic energy is constant across every snapshot
    ke = r.kinetic_energy                           # (1, T+1)
    assert _close(ke, ke[:, :1].expand_as(ke))


@PROP
@given(data=system(), dt=st.floats(0.005, 0.05, **_F), k=st.integers(1, 15))
def test_closed_nbody_rollout_conserves_total_momentum(data, dt, k):
    pos, vel, mass, n, d = data
    p0 = pos.unsqueeze(0)
    v0 = vel.unsqueeze(0)
    r = rollout(p0, v0, steps=k, dt=dt, mass=mass,
                accel_fn=lambda p, v: pairwise_gravity(p, mass))
    # momentum at every snapshot equals the initial momentum (internal forces only)
    m0 = momentum(v0, mass)                          # (1, D)
    for t in range(r.velocities.shape[1]):
        assert _close(momentum(r.velocities[:, t], mass), m0, atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
