"""
mt_lnn/physics_ops.py -- composable Newtonian dynamics operators (L4 step 3).

Why this module exists
----------------------
``spatial_ops.py`` lets the agent *compute over geometry* (distances, proximity
graphs, reachability). But a scene is not static: "在脑海中模拟事件进展" -- to
mentally simulate how events unfold -- needs a way to *compute physical
dynamics*, i.e. push a configuration forward in time under forces and contacts.
This module adds that missing layer: a small set of **composable,
zero-parameter, differentiable-where-it-is-physical** operators that evolve a
state ``(positions, velocities)`` under Newtonian mechanics:

    integration   -- semi-implicit (symplectic) Euler, energy-stable;
    forces        -- uniform gravity field, and softened pairwise (N-body)
                     gravity that conserves momentum exactly;
    contacts      -- sphere-sphere collision detection + impulse response with
                     restitution (momentum-conserving; energy-conserving when
                     e == 1), and axis-aligned box-wall reflection;
    diagnostics   -- kinetic energy and total momentum (conservation probes);
    ROLLOUT       -- compose the above into a trajectory: the agent's internal
                     "what happens next" simulation.

The point, as with ``spatial_ops``, is **composition**: a flagship query like
"will the bouncing ball clear the wall?" is *computed* by rolling the dynamics
forward from ``(g, v0, restitution)`` -- emergent, not memorised. Raise the
restitution and the same code reports a different outcome.

Honest scope
------------
This is a deterministic physics *operator* layer (the engine), not a learned
dynamics model. It is the substrate the latent imagination rollout
(:mod:`mt_lnn.imagination`) can be grounded against and the perception encoders
(:mod:`mt_lnn.spatial`) feed into -- the "compute, don't memorise" counterpart
to a stored physics fact. Integration is first-order symplectic Euler: it is
energy-stable (bounded shadow Hamiltonian), not exact; tests pin it against the
analytic invariants it actually preserves (constant momentum, exact velocity
under constant acceleration, swapped velocities in an elastic head-on hit, a
near-constant two-body orbit radius), not against an unrealistic closed form.

Design / coupling
-----------------
* Pure functions, **stateless, zero trainable parameters**. Nothing here is an
  ``nn.Module``; there is no backbone coupling. Imports only ``torch`` (+
  ``dataclasses``); never imports ``model.py``.
* Batched: state is ``positions`` / ``velocities`` shaped ``(B, N, D)`` or
  ``(N, D)`` (a missing batch dim is added and squeezed back). ``D`` is the
  spatial dimension (2 or 3 typically). ``N`` is the number of bodies.
* The force operators and the integrator are differentiable; contact resolution
  thresholds (overlap / approaching tests) and is piecewise-constant -- this is
  documented per-function, not hidden.

Every operator is pinned against an analytic ground truth in
``tests/test_physics_ops.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch

__all__ = [
    "PhysicsRollout",
    "integrate",
    "integrate_verlet",
    "uniform_gravity",
    "pairwise_gravity",
    "kinetic_energy",
    "momentum",
    "overlapping_pairs",
    "resolve_sphere_collisions",
    "reflect_in_box",
    "rollout",
]


def _batched(state: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """Return ``(state_3d, had_batch)``; add a batch dim if ``state`` is 2-D."""
    if state.dim() == 2:
        return state.unsqueeze(0), False
    if state.dim() == 3:
        return state, True
    raise ValueError(
        f"state must be (N, D) or (B, N, D), got shape {tuple(state.shape)}"
    )


def _as_per_body(x, N: int, dtype, device) -> torch.Tensor:
    """Broadcast a scalar / (N,) spec for mass or radius to a ``(N,)`` tensor."""
    t = torch.as_tensor(x, dtype=dtype, device=device)
    if t.dim() == 0:
        t = t.expand(N).clone()
    if t.shape[-1] != N:
        raise ValueError(f"expected a scalar or ({N},) per-body spec, got {tuple(t.shape)}")
    return t


# ---------------------------------------------------------------------------
# Integration (differentiable, symplectic)
# ---------------------------------------------------------------------------

def integrate(positions: torch.Tensor, velocities: torch.Tensor,
              acceleration: torch.Tensor, dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """One semi-implicit (symplectic) Euler step.

    ``v' = v + a*dt`` then ``x' = x + v'*dt`` (note: the *updated* velocity moves
    the position -- this is what makes the scheme symplectic / energy-stable, as
    opposed to forward Euler which uses the old velocity and drifts in energy).

    Shapes broadcast naturally; ``positions``/``velocities``/``acceleration`` are
    all ``(B, N, D)`` or ``(N, D)``. Differentiable in every input. Returns
    ``(positions_next, velocities_next)``.
    """
    v_next = velocities + acceleration * float(dt)
    x_next = positions + v_next * float(dt)
    return x_next, v_next


def integrate_verlet(positions: torch.Tensor, velocities: torch.Tensor,
                     accel_fn: Callable[[torch.Tensor], torch.Tensor], dt: float,
                     *, accel: Optional[torch.Tensor] = None
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One velocity-Verlet step -- a 2nd-order, time-reversible symplectic integrator.

    The kick-drift-kick form::

        v_half = v + 0.5 a(x)  dt
        x'     = x + v_half    dt
        v'     = v_half + 0.5 a(x') dt

    Unlike semi-implicit Euler (:func:`integrate`, 1st order), velocity Verlet is
    *second* order and *time-reversible*: its energy error is O(dt^2) and stays
    bounded over arbitrarily long runs (no secular drift), and integrating
    forward then negating the velocity retraces the path exactly. These are the
    properties a long imagination rollout / orbit needs, and they are pinned
    against the analytic harmonic oscillator in the tests (energy drift scales as
    dt^2 vs dt for Euler; reversal error ~ machine epsilon).

    ``accel_fn`` maps *positions -> acceleration* (a conservative, velocity-
    independent force field, e.g. :func:`uniform_gravity` / :func:`pairwise_gravity`
    wrapped in a closure). Pass ``accel`` to reuse an already-computed ``a(x)``
    (the previous step's ``a(x')``) and save one force evaluation. Returns
    ``(positions_next, velocities_next, accel_at_x_next)`` -- the third value is
    ``a(x')``, ready to feed back as ``accel`` next step. Differentiable.

    Note: Verlet assumes the force depends on position only; for a velocity-
    dependent ``accel_fn`` it degrades to an approximation (documented, not hidden).
    """
    a = accel_fn(positions) if accel is None else accel
    v_half = velocities + 0.5 * a * float(dt)
    x_next = positions + v_half * float(dt)
    a_next = accel_fn(x_next)
    v_next = v_half + 0.5 * a_next * float(dt)
    return x_next, v_next, a_next


# ---------------------------------------------------------------------------
# Forces -> accelerations
# ---------------------------------------------------------------------------

def uniform_gravity(positions: torch.Tensor, g) -> torch.Tensor:
    """A constant gravitational acceleration field, broadcast to every body.

    ``g`` is a vector broadcastable to ``(D,)`` (e.g. ``[0.0, -9.81]``). Returns
    an acceleration tensor with the same shape as ``positions``. Differentiable.
    """
    p, had_batch = _batched(positions)
    g = torch.as_tensor(g, dtype=p.dtype, device=p.device)
    if g.shape[-1] != p.shape[-1]:
        raise ValueError(f"g must be ({p.shape[-1]},), got {tuple(g.shape)}")
    a = g.expand_as(p).clone()
    return a if had_batch else a.squeeze(0)


def pairwise_gravity(positions: torch.Tensor, mass, *, G: float = 1.0,
                     softening: float = 1e-3) -> torch.Tensor:
    """Softened N-body gravitational acceleration on every body.

    ``a_i = G * sum_{j != i} m_j (p_j - p_i) / (|p_j - p_i|^2 + softening^2)^{3/2}``.

    The Plummer ``softening`` keeps the force finite at zero separation (and the
    gradient well-defined). The reaction is exact: ``sum_i m_i a_i = 0`` to
    floating-point, so a closed system conserves momentum. ``mass`` is a scalar
    or ``(N,)``. Returns acceleration shaped like ``positions``. Differentiable.
    """
    p, had_batch = _batched(positions)
    B, N, D = p.shape
    m = _as_per_body(mass, N, p.dtype, p.device)            # (N,)
    diff = p.unsqueeze(1) - p.unsqueeze(2)                  # (B, N, N, D): diff[i,j]=p_j-p_i
    r2 = diff.pow(2).sum(-1) + float(softening) ** 2        # (B, N, N)
    inv = r2.pow(-1.5)                                      # (B, N, N)
    eye = torch.eye(N, dtype=torch.bool, device=p.device)
    inv = inv.masked_fill(eye.unsqueeze(0), 0.0)            # drop self term
    w = (inv * m.view(1, 1, N))                             # (B, N, N): m_j / r^3
    a = float(G) * (w.unsqueeze(-1) * diff).sum(dim=2)      # (B, N, D)
    return a if had_batch else a.squeeze(0)


# ---------------------------------------------------------------------------
# Conservation diagnostics
# ---------------------------------------------------------------------------

def kinetic_energy(velocities: torch.Tensor, mass) -> torch.Tensor:
    """Total kinetic energy ``0.5 * sum_i m_i |v_i|^2`` per batch element.

    Returns ``(B,)`` or a scalar. Differentiable.
    """
    v, had_batch = _batched(velocities)
    B, N, D = v.shape
    m = _as_per_body(mass, N, v.dtype, v.device)
    ke = 0.5 * (m.view(1, N) * v.pow(2).sum(-1)).sum(-1)    # (B,)
    return ke if had_batch else ke.squeeze(0)


def momentum(velocities: torch.Tensor, mass) -> torch.Tensor:
    """Total linear momentum ``sum_i m_i v_i`` per batch element.

    Returns ``(B, D)`` or ``(D,)``. Differentiable.
    """
    v, had_batch = _batched(velocities)
    B, N, D = v.shape
    m = _as_per_body(mass, N, v.dtype, v.device)
    pmom = (m.view(1, N, 1) * v).sum(dim=1)                 # (B, D)
    return pmom if had_batch else pmom.squeeze(0)


# ---------------------------------------------------------------------------
# Contacts: detection + response
# ---------------------------------------------------------------------------

def overlapping_pairs(positions: torch.Tensor, radius) -> torch.Tensor:
    """Boolean adjacency of overlapping spheres: ``|p_i - p_j| < r_i + r_j``.

    ``radius`` is a scalar or ``(N,)``. Returns a boolean ``(B, N, N)`` / ``(N, N)``
    matrix, symmetric, with a zero diagonal. Thresholded (discrete), not
    differentiable.
    """
    p, had_batch = _batched(positions)
    B, N, D = p.shape
    r = _as_per_body(radius, N, p.dtype, p.device)
    diff = p.unsqueeze(2) - p.unsqueeze(1)                  # (B, N, N, D)
    dist = diff.norm(dim=-1)                                # (B, N, N)
    rsum = r.view(N, 1) + r.view(1, N)                      # (N, N)
    over = dist < rsum.unsqueeze(0)
    eye = torch.eye(N, dtype=torch.bool, device=p.device).unsqueeze(0)
    over = over & ~eye
    return over if had_batch else over.squeeze(0)


def resolve_sphere_collisions(positions: torch.Tensor, velocities: torch.Tensor, *,
                              radius, mass=None, restitution: float = 1.0) -> torch.Tensor:
    """Impulse-based response for overlapping, *approaching* sphere pairs.

    For every pair ``(i, j)`` whose spheres overlap and whose relative velocity
    closes the gap, apply an impulse along the line of centres:

        j_mag = -(1 + e) (v_rel . n) / (1/m_i + 1/m_j),   n = (p_j - p_i)/|.|

    with ``v_rel = v_i - v_j`` (bodies approach when ``v_rel . n > 0``).
    Velocities update by ``v_i += j_mag n / m_i`` and ``v_j -= j_mag n / m_j``.
    Momentum
    is conserved exactly; kinetic energy is conserved when ``restitution == 1``
    and dissipated for ``e < 1``. Separating pairs are left untouched (so resting
    contacts do not jitter). Simultaneous multi-body contacts are resolved in a
    single momentum-conserving pass (an approximation for dense pile-ups, exact
    for isolated pairs).

    ``radius`` / ``mass`` are scalars or ``(N,)`` (mass defaults to unit). Returns
    the post-collision ``velocities`` (same shape as input). Position is *not*
    modified here -- pair detection thresholds, so this is piecewise-constant in
    the state (not differentiable across a contact event).
    """
    p, had_batch = _batched(positions)
    v, _ = _batched(velocities)
    B, N, D = p.shape
    r = _as_per_body(radius, N, p.dtype, p.device)
    m = _as_per_body(1.0 if mass is None else mass, N, p.dtype, p.device)
    e = float(restitution)

    diff = p.unsqueeze(1) - p.unsqueeze(2)                  # (B,N,N,D): diff[i,j]=p_j-p_i
    dist = diff.norm(dim=-1).clamp_min(1e-12)               # (B,N,N)
    n = diff / dist.unsqueeze(-1)                           # unit i->j
    vrel = v.unsqueeze(2) - v.unsqueeze(1)                  # (B,N,N,D): v_i - v_j
    vn = (vrel * n).sum(-1)                                 # (B,N,N): closing speed (>0 approaching)

    rsum = (r.view(N, 1) + r.view(1, N)).unsqueeze(0)       # (1,N,N)
    iu = torch.triu(torch.ones(N, N, device=p.device), diagonal=1).bool().unsqueeze(0)
    active = iu & (dist < rsum) & (vn > 0)                  # each unordered pair once

    inv_m = (1.0 / m).view(1, N, 1) + (1.0 / m).view(1, 1, N)   # (1,N,N): 1/m_i + 1/m_j
    jmag = torch.where(active, -(1.0 + e) * vn / inv_m,
                       torch.zeros_like(vn))                # (B,N,N)
    J = jmag.unsqueeze(-1) * n                              # (B,N,N,D): impulse i->j

    # for a stored pair i<j: body i gets +J/m_i, body j gets -J/m_j
    dv = (J.sum(dim=2) - J.sum(dim=1)) / m.view(1, N, 1)    # (B,N,D)
    v_next = v + dv
    return v_next if had_batch else v_next.squeeze(0)


def reflect_in_box(positions: torch.Tensor, velocities: torch.Tensor, lo, hi, *,
                   restitution: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reflect bodies off the walls of an axis-aligned box ``[lo, hi]``.

    A body crossing a wall is clamped back to it and the offending velocity
    component is flipped and scaled by ``restitution`` (``e == 1`` is a perfect
    bounce; ``e == 0`` sticks to the wall). ``lo`` / ``hi`` broadcast to ``(D,)``;
    use ``+-inf`` for an open side (e.g. a floor at ``y = 0`` with no ceiling).
    Returns ``(positions, velocities)``. Thresholded at the wall (discrete).
    """
    p, had_batch = _batched(positions)
    v, _ = _batched(velocities)
    lo = torch.as_tensor(lo, dtype=p.dtype, device=p.device)
    hi = torch.as_tensor(hi, dtype=p.dtype, device=p.device)
    e = float(restitution)

    below = p < lo
    above = p > hi
    crossed = below | above
    p_clamped = torch.minimum(torch.maximum(p, lo), hi)
    v_reflected = torch.where(crossed, -e * v, v)
    p_out = p_clamped if had_batch else p_clamped.squeeze(0)
    v_out = v_reflected if had_batch else v_reflected.squeeze(0)
    return p_out, v_out


# ---------------------------------------------------------------------------
# Rollout: compose the operators into a trajectory
# ---------------------------------------------------------------------------

@dataclass
class PhysicsRollout:
    """A simulated trajectory.

    Attributes
    ----------
    positions : (B, T+1, N, D)   states at t = 0 .. T*dt (snapshots include t=0)
    velocities : (B, T+1, N, D)
    times : (T+1,)               wall-clock time of each snapshot
    kinetic_energy : (B, T+1)    total KE at each snapshot (conservation probe)
    dt : float
    """

    positions: torch.Tensor
    velocities: torch.Tensor
    times: torch.Tensor
    kinetic_energy: torch.Tensor
    dt: float

    @property
    def _t_axis(self) -> int:
        # time axis is 0 for an unbatched (T+1, N, D) rollout and 1 for a
        # batched (B, T+1, N, D) one -- index it relative to the trailing
        # (N, D) so these accessors are correct for BOTH layouts.
        return self.positions.dim() - 3

    @property
    def steps(self) -> int:
        return int(self.positions.shape[self._t_axis] - 1)

    @property
    def final_positions(self) -> torch.Tensor:
        ax = self._t_axis
        return self.positions.select(ax, self.positions.shape[ax] - 1)

    @property
    def final_velocities(self) -> torch.Tensor:
        ax = self.velocities.dim() - 3
        return self.velocities.select(ax, self.velocities.shape[ax] - 1)


def rollout(positions: torch.Tensor, velocities: torch.Tensor, *,
            steps: int, dt: float,
            gravity=None,
            accel_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
            mass=None, radius=None, restitution: float = 1.0,
            bounds: Optional[Tuple] = None,
            integrator: str = "symplectic_euler") -> PhysicsRollout:
    """Compose the operators forward in time: the agent's "what happens next".

    Each step: (1) acceleration = uniform ``gravity`` (if given) + ``accel_fn``
    (if given, e.g. :func:`pairwise_gravity` via a closure); (2) integrate with
    the chosen ``integrator``; (3) if ``radius`` is set, resolve sphere
    collisions; (4) if ``bounds`` is set, reflect off the box walls. Snapshots
    (including the initial state) are stacked into a :class:`PhysicsRollout`.

    ``integrator`` selects the time-stepper:
      * ``"symplectic_euler"`` (default) -- 1st-order semi-implicit Euler
        (:func:`integrate`); the established behaviour, unchanged.
      * ``"verlet"`` -- 2nd-order, time-reversible velocity Verlet
        (:func:`integrate_verlet`); far smaller energy drift over long
        horizons (O(dt^2) vs O(dt)), the better choice for long orbits /
        imagination rollouts. Verlet treats the force as position-only, so it
        is exact for ``gravity`` / ``accel_fn(pos)`` and an approximation if
        ``accel_fn`` actually depends on velocity (documented).

    Parameters
    ----------
    positions, velocities : (B, N, D) or (N, D)
    steps : int               number of integration steps (>= 1)
    dt : float                timestep
    gravity : optional        constant acceleration vector ``(D,)``
    accel_fn : optional       ``(pos, vel) -> accel`` for state-dependent forces
    mass, radius : optional   scalar or ``(N,)``; ``radius`` enables collisions
    restitution : float       bounce elasticity for collisions and walls
    bounds : optional         ``(lo, hi)``, each broadcastable to ``(D,)``;
                              enables box-wall reflection (use +-inf for open sides)

    Returns a :class:`PhysicsRollout`. Differentiable through integration and
    forces; contact/wall events threshold (piecewise-constant there).
    """
    if int(steps) < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if integrator not in ("symplectic_euler", "verlet"):
        raise ValueError(
            f"integrator must be 'symplectic_euler' or 'verlet', got {integrator!r}"
        )
    p, had_batch = _batched(positions)
    v, _ = _batched(velocities)
    B, N, D = p.shape
    m = _as_per_body(1.0 if mass is None else mass, N, p.dtype, p.device)

    def _accel(pos: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
        a = torch.zeros_like(pos)
        if gravity is not None:
            a = a + uniform_gravity(pos, gravity)
        if accel_fn is not None:
            a = a + accel_fn(pos, vel)
        return a

    pos_hist: List[torch.Tensor] = [p]
    vel_hist: List[torch.Tensor] = [v]
    for _ in range(int(steps)):
        if integrator == "verlet":
            # position-only force closure (freeze v across the sub-step); recomputed
            # each step so contacts/walls that change v between steps stay correct
            v_frozen = v
            p, v, _ = integrate_verlet(p, v, lambda x: _accel(x, v_frozen), dt)
        else:
            p, v = integrate(p, v, _accel(p, v), dt)
        if radius is not None:
            v = resolve_sphere_collisions(p, v, radius=radius, mass=m,
                                          restitution=restitution)
        if bounds is not None:
            lo, hi = bounds
            p, v = reflect_in_box(p, v, lo, hi, restitution=restitution)
        pos_hist.append(p)
        vel_hist.append(v)

    pos = torch.stack(pos_hist, dim=1)                      # (B, T+1, N, D)
    vel = torch.stack(vel_hist, dim=1)
    times = torch.arange(int(steps) + 1, dtype=p.dtype, device=p.device) * float(dt)
    ke = 0.5 * (m.view(1, 1, N) * vel.pow(2).sum(-1)).sum(-1)   # (B, T+1)

    if not had_batch:
        pos = pos.squeeze(0)
        vel = vel.squeeze(0)
        ke = ke.squeeze(0)
    return PhysicsRollout(positions=pos, velocities=vel, times=times,
                          kinetic_energy=ke, dt=float(dt))
