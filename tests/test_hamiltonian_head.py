"""Tests for the hard-constraint physics-informed head (mt_lnn/hamiltonian_head.py).

These pin the ARCHITECTURAL guarantees that make it a "physics-written-into-the-
network" module rather than a soft penalty: symplectic energy stability and exact
time-reversibility of the integrator, plus trainability. The "does structure beat
an unstructured net on a real system" comparison lives in
benchmarks/physics_rollout_eval.py (it needs training).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mt_lnn.hamiltonian_head import HamiltonianHead, MLPFieldHead


def test_symplectic_conserves_energy_better_than_forward_euler():
    """The architectural property: velocity-Verlet on H = T(p)+V(q) keeps energy
    BOUNDED, whereas forward Euler on the SAME conservative field drifts. Same
    field, only the integrator differs — this isolates 'symplectic structure' as
    the source of conservation, untrained (holds for any T, V)."""
    torch.manual_seed(0)
    head = HamiltonianHead(dim=1, hidden_dim=32, depth=2)
    head.eval()
    # Gentle init so the random field is not stiff (keeps both integrators in
    # their stable regime; the point is the RELATIVE drift, not absolute scale).
    with torch.no_grad():
        for m in (head.T, head.V):
            m[-1].weight.mul_(0.3)
            m[-1].bias.zero_()

    q = torch.tensor([[1.0]])
    p = torch.tensor([[0.0]])
    dt, steps = 0.05, 300

    # Symplectic (the module's own rollout). Field is an autograd gradient, so
    # run under enable_grad and detach.
    with torch.enable_grad():
        qs, ps = head.rollout(q, p, steps, dt)
    E_sympl = head.energy(qs.detach(), ps.detach()).squeeze(-1)  # (steps+1,)

    # Forward Euler on the SAME field (old q,p drive the update — the classic
    # energy-drifting scheme).
    qe, pe = q.clone(), p.clone()
    E_euler = [head.energy(qe, pe).item()]
    with torch.enable_grad():
        for _ in range(steps):
            dq = head.dT_dp(pe).detach()
            dp = (-head.dV_dq(qe)).detach()
            qe, pe = qe + dt * dq, pe + dt * dp
            E_euler.append(head.energy(qe, pe).item())
    E_euler = torch.tensor(E_euler)

    E0 = E_sympl[0].abs().clamp_min(1e-6)
    drift_sympl = (E_sympl - E_sympl[0]).abs().max() / E0
    drift_euler = (E_euler - E_euler[0]).abs().max() / E0
    print(f"[1] energy drift  symplectic {drift_sympl:.4e}  vs  forward-Euler {drift_euler:.4e}")
    assert torch.isfinite(E_sympl).all(), "symplectic rollout diverged"
    assert drift_sympl < drift_euler, "symplectic should conserve energy better than Euler"
    assert drift_sympl < 0.05, f"symplectic energy drift should be small, got {drift_sympl:.3e}"


def test_integrator_is_exactly_time_reversible():
    """Velocity-Verlet is algebraically reversible: one step of +dt followed by
    one step of -dt returns to the exact start, for ANY T, V. This is the
    strongest architectural proof (holds to machine precision, no training)."""
    torch.manual_seed(1)
    head = HamiltonianHead(dim=3, hidden_dim=32, depth=2)
    head.eval()
    q = torch.randn(4, 3)
    p = torch.randn(4, 3)
    with torch.enable_grad():
        q1, p1 = head.step(q, p, dt=0.1)
        q2, p2 = head.step(q1, p1, dt=-0.1)   # reverse
    dq = (q2 - q).abs().max().item()
    dp = (p2 - p).abs().max().item()
    print(f"[2] reversal error  dq {dq:.2e}  dp {dp:.2e}")
    assert dq < 1e-4 and dp < 1e-4, f"not time-reversible: dq={dq:.2e} dp={dp:.2e}"


def test_gradients_reach_energy_params_and_shapes():
    """A supervised 1-step MSE must backprop through the symplectic step into
    BOTH T and V parameters (the integrator is differentiable), and batched
    phase states flow through cleanly."""
    torch.manual_seed(2)
    head = HamiltonianHead(dim=2, hidden_dim=16, depth=2)
    head.train()
    q = torch.randn(8, 2)
    p = torch.randn(8, 2)
    q_tgt = torch.randn(8, 2)
    p_tgt = torch.randn(8, 2)

    q1, p1 = head.step(q, p, dt=0.1)
    assert q1.shape == (8, 2) and p1.shape == (8, 2)
    loss = ((q1 - q_tgt) ** 2 + (p1 - p_tgt) ** 2).mean()
    loss.backward()

    t_grad = head.T[0].weight.grad
    v_grad = head.V[0].weight.grad
    assert t_grad is not None and torch.isfinite(t_grad).all() and t_grad.abs().sum() > 0, \
        "no gradient into kinetic-energy params T"
    assert v_grad is not None and torch.isfinite(v_grad).all() and v_grad.abs().sum() > 0, \
        "no gradient into potential-energy params V"
    print("[3] gradients reach both T and V; batched shapes OK")


def test_mlp_field_baseline_runs_and_differs():
    """The unstructured control head runs and is a genuinely different model
    (no energy method, plain Euler) — the honest counterfactual for the
    benchmark."""
    torch.manual_seed(3)
    base = MLPFieldHead(dim=2, hidden_dim=16, depth=2)
    q = torch.randn(5, 2)
    p = torch.randn(5, 2)
    qs, ps = base.rollout(q, p, steps=10, dt=0.05)
    assert qs.shape == (11, 5, 2) and ps.shape == (11, 5, 2)
    assert torch.isfinite(qs).all() and torch.isfinite(ps).all()
    assert not hasattr(base, "energy"), "baseline must NOT define a conserved energy"
    print("[4] MLPField baseline rollout OK, has no energy structure")


if __name__ == "__main__":
    test_symplectic_conserves_energy_better_than_forward_euler()
    test_integrator_is_exactly_time_reversible()
    test_gradients_reach_energy_params_and_shapes()
    test_mlp_field_baseline_runs_and_differs()
    print("all hamiltonian_head tests passed")
