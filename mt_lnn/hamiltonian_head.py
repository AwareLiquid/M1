"""
mt_lnn/hamiltonian_head.py — hard-constraint physics-informed head (HNN).

Why this module exists
----------------------
The PINN / physics-informed-ML investigation (see BENCHMARKS.md "Physics-informed
head" and the architecture notes) reached one strong, evidence-backed conclusion:
the way to "write physics INTO the network" that actually fits a continuous-time
(liquid ODE) backbone is the HARD-CONSTRAINT route — a Hamiltonian Neural Network
(Greydanus et al. 2019) integrated with a SYMPLECTIC scheme — NOT a soft PDE-
residual loss bolted onto a language model.

  * SOFT PINN puts physics in the LOSS (a PDE residual penalty); the net only
    learns to approximately obey it, and it adds nothing over MT-LNN's already-
    exact hand-coded engine (physics_ops.py) when the equations are known.
  * HARD (this module) bakes physics into the ARCHITECTURE: the net outputs a
    scalar energy H(q,p) = T(p) + V(q), and the state is advanced by a velocity-
    Verlet (leapfrog) symplectic integrator. Total energy is then conserved BY
    CONSTRUCTION — bounded O(dt^2) drift, exactly time-reversible — for ANY
    learned T, V, trained or not. That is the "conservation is a property of the
    architecture, not a soft penalty" guarantee.

Honest scope (do not overclaim)
-------------------------------
* Operates on a CONTINUOUS phase state (q, p) in R^dim — generalized positions and
  momenta — NOT token latents. It is a world-model / trajectory-prediction
  component: a symplectic, conservation-guaranteed alternative to
  world_model.PredictiveStateHead for physics-governed continuous state.
* A physics prior is category-mismatched to language: it does NOT reduce LLM
  hallucination and is PPL-neutral on next-token modelling (the repo's own
  switch-matrix already showed physics/spatial modules are PPL-neutral). This
  module is therefore DELIBERATELY OFF the served-LM path — it is validated only
  on physics metrics (k-step rollout MSE, energy drift), never on perplexity.
  See benchmarks/physics_rollout_eval.py.
* The affinity that IS real: MT-LNN's closed-form LTC core is a continuous-time
  ODE, so a learned-Hamiltonian vector field integrated symplectically is the
  natural continuous-state counterpart to physics_ops.py's hand-coded symplectic
  Euler / velocity-Verlet integrators.

Design / coupling
-----------------
* Self-contained nn.Module. Imports only torch; never imports model.py (same
  loose-coupling discipline as physics_ops.py / astrocyte.py). Nothing here
  touches the language-model forward.

Reference: Greydanus, Dzamba, Yosinski, "Hamiltonian Neural Networks", NeurIPS
2019 (arXiv:1906.01563); velocity-Verlet is the same 2nd-order symplectic scheme
physics_ops.integrate_verlet uses.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _energy_mlp(dim: int, hidden_dim: int, depth: int) -> nn.Sequential:
    """A smooth (Tanh) MLP R^dim -> R for a scalar energy. Tanh, not ReLU: the
    field is the GRADIENT of this net, so a C^inf activation gives a smooth,
    differentiable-twice force field (ReLU's kinks make dH/dx discontinuous)."""
    layers: list[nn.Module] = []
    d = dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden_dim), nn.Tanh()]
        d = hidden_dim
    layers += [nn.Linear(d, 1)]
    return nn.Sequential(*layers)


class HamiltonianHead(nn.Module):
    """Separable Hamiltonian H(q,p) = T(p) + V(q) advanced by velocity-Verlet.

    q, p are (..., dim). Energy conservation is architectural: leapfrog on a
    separable H has bounded O(dt^2) energy error and is time-reversible for any
    T, V. Training (a supervised 1-step or k-step trajectory MSE) shapes T, V to
    match observed dynamics while the CONSERVATION structure is never something
    the optimizer can break.
    """

    def __init__(self, dim: int, hidden_dim: int = 64, depth: int = 2,
                 context_dim: int = 0):
        super().__init__()
        self.dim = int(dim)
        self.context_dim = int(context_dim)
        # T(p): kinetic energy, a function of momentum only (standard physics).
        # V(q | ctx): potential energy. When context_dim > 0 the potential is
        # CONDITIONED on an external context vector — the hook by which the
        # MT-LNN liquid core sets the energy landscape the particle moves in
        # ("the liquid ODE core is the physics substrate", realized in code, not
        # just asserted). context_dim = 0 (default) is the pure autonomous
        # Hamiltonian: bit-identical to the pre-context module.
        self.T = _energy_mlp(dim, hidden_dim, depth)
        self.V = _energy_mlp(dim + self.context_dim, hidden_dim, depth)

    # -- energy & its gradients (the conservative vector field) ---------------

    def energy(self, q: torch.Tensor, p: torch.Tensor,
               context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """H(q,p) = T(p) + V(q | ctx).  (..., dim) -> (...,) scalar."""
        v_in = q if context is None else torch.cat([q, context], dim=-1)
        return self.T(p).squeeze(-1) + self.V(v_in).squeeze(-1)

    def _grad(self, net: nn.Module, x: torch.Tensor,
              context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """dsum(net([x, ctx]))/dx — the gradient of a scalar (possibly context-
        conditioned) potential w.r.t. x ONLY (context is fixed within a step).

        create_graph follows self.training so that during training the gradient
        of the 1-step loss flows through the integrator into T/V's parameters
        AND (via context) into whatever produced the context (the liquid core);
        at eval it is a plain (graph-free) field evaluation. Requires grad to be
        enabled by the caller (the field IS an autograd gradient), so eval
        rollouts run under torch.enable_grad() and detach afterwards.
        """
        if not x.requires_grad:
            # Ground-truth leaves / detached inputs: differentiate w.r.t. a LOCAL
            # detached copy instead of flipping requires_grad on the caller's
            # tensor in place (that side effect leaked out of this function).
            x = x.detach().requires_grad_(True)
        inp = x if context is None else torch.cat([x, context], dim=-1)
        (g,) = torch.autograd.grad(net(inp).sum(), x, create_graph=self.training)
        return g

    def dT_dp(self, p: torch.Tensor) -> torch.Tensor:
        """dH/dp = generalized velocity q_dot."""
        return self._grad(self.T, p)

    def dV_dq(self, q: torch.Tensor,
              context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """dH/dq;  p_dot = -dH/dq  (the force).  Context sets the potential."""
        return self._grad(self.V, q, context)

    # -- symplectic integration ----------------------------------------------

    def step(self, q: torch.Tensor, p: torch.Tensor, dt: float,
             context: Optional[torch.Tensor] = None
             ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One velocity-Verlet (leapfrog, kick-drift-kick) symplectic step:

            p_half = p - (dt/2) dV/dq(q | ctx)
            q'     = q + dt      dT/dp(p_half)
            p'     = p_half - (dt/2) dV/dq(q' | ctx)

        Second order, time-reversible, energy error bounded (no secular drift).
        `context` (if given) is held fixed across the step, so the local energy
        H(.|ctx) is conserved by the symplectic scheme exactly as in the
        autonomous case; across steps a changing context does real work (the
        liquid core drives the system), which is physically correct.
        """
        dt = float(dt)
        p_half = p - 0.5 * dt * self.dV_dq(q, context)
        q_next = q + dt * self.dT_dp(p_half)
        p_next = p_half - 0.5 * dt * self.dV_dq(q_next, context)
        return q_next, p_next

    def rollout(self, q0: torch.Tensor, p0: torch.Tensor, steps: int, dt: float,
                context: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integrate `steps` symplectic steps from (q0, p0) under a FIXED context.

        Returns (qs, ps), each (steps + 1, ..., dim) including the initial state.
        The field is an autograd gradient, so call under enable_grad; at eval,
        detach the returned trajectory.
        """
        qs, ps = [q0], [p0]
        q, p = q0, p0
        for _ in range(int(steps)):
            q, p = self.step(q, p, dt, context)
            qs.append(q)
            ps.append(p)
        return torch.stack(qs), torch.stack(ps)


class MLPFieldHead(nn.Module):
    """Baseline / ablation control: an UNSTRUCTURED learned vector field
    f(q,p) -> (q_dot, p_dot), advanced with plain (non-symplectic) Euler. It has
    NO conservation structure — the honest counterfactual for measuring what the
    Hamiltonian + symplectic architecture actually buys on long-horizon rollout
    (energy drift).

    Fairness note: this control differs from HamiltonianHead on TWO axes — the
    energy parametrization (structured H vs unstructured field) AND the
    integrator (symplectic velocity-Verlet vs forward Euler). The benchmark
    reports both models' param counts (HamiltonianHead runs two dim->1 energy
    MLPs, so it is ~1.5-2x the params of this single 2dim->2dim field MLP — NOT
    matched); the reported advantage is therefore "structure + integrator", and
    the decisive signal is ENERGY DRIFT, which no amount of extra capacity in an
    unstructured Euler field can fix (it has no conserved quantity by
    construction). The integrator axis is isolated separately in
    tests/test_hamiltonian_head.py (symplectic vs forward-Euler on the SAME
    field).
    """

    def __init__(self, dim: int, hidden_dim: int = 64, depth: int = 2):
        super().__init__()
        self.dim = int(dim)
        layers: list[nn.Module] = []
        d = 2 * dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden_dim), nn.Tanh()]
            d = hidden_dim
        layers += [nn.Linear(d, 2 * dim)]
        self.f = nn.Sequential(*layers)

    def field(self, q: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.f(torch.cat([q, p], dim=-1))
        return out[..., :self.dim], out[..., self.dim:]

    def step(self, q: torch.Tensor, p: torch.Tensor, dt: float
             ) -> Tuple[torch.Tensor, torch.Tensor]:
        dt = float(dt)
        dq, dp = self.field(q, p)
        return q + dt * dq, p + dt * dp

    def rollout(self, q0: torch.Tensor, p0: torch.Tensor, steps: int, dt: float
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        qs, ps = [q0], [p0]
        q, p = q0, p0
        for _ in range(int(steps)):
            q, p = self.step(q, p, dt)
            qs.append(q)
            ps.append(p)
        return torch.stack(qs), torch.stack(ps)


class HamiltonianWorldModelHead(nn.Module):
    """Physics-informed world-model head — the GENUINE integration of the
    Hamiltonian prior into the MT-LNN architecture (not the free-standing
    HamiltonianHead island).

    It predicts the next hidden latent through a SYMPLECTIC phase-space
    bottleneck, i.e. it is a structured, conservation-biased drop-in ALTERNATIVE
    to world_model.PredictiveStateHead that occupies the same architectural slot
    and the same self-supervised next-latent objective.

    Pipeline (x: (B, T, d_model) from the liquid core, after final_norm):
        (q, p) = split(to_phase(x))                # d_model -> phase state   [the latent->phase BRIDGE]
        ctx    = context_proj(x)     (optional)    # liquid core conditions the potential
        (q',p') = ham.step(q, p, dt | ctx)         # symplectic Hamiltonian evolution   [the physics]
        h_pred  = from_phase([q', p'])             # phase -> predicted next hidden      [world-model's job]
        L = shift-by-1 MSE(h_pred[:, :-1], x[:, 1:].detach())   # self-supervised next-latent prediction

    Why this is real integration (closes the 5 island gaps):
      1. LIQUID SUBSTRATE — the Hamiltonian potential V(q | ctx) is CONDITIONED on
         the liquid core's hidden state (context_proj(x)), so the continuous-time
         core sets the energy landscape; the physics is driven by MT-LNN, not a
         generic backbone.
      2. WORLD-MODEL SLOT — same role/objective as PredictiveStateHead (predict the
         next hidden latent); config selects one, and this folds into the same
         model.forward aux-loss plumbing.
      3. PHYSICS_OPS COUPLING — dV_dq exposes the learned force field, which a
         training objective can regularize against physics_ops' exact accelerations
         (see benchmarks/physics_rollout_eval.py --physics_ops_consistency).
      4. THE BRIDGE — to_phase (d_model->2*phase_dim) and from_phase back; the
         missing d_model<->(q,p) read-out the review flagged as the decisive gap.
      5. REACHABLE — built from config.use_hamiltonian_world_model and wired into
         MTLNNModel.forward; a trained model can actually turn it on.

    Honest scope: the symplectic phase-space bottleneck is a structural INDUCTIVE
    BIAS on the next-latent predictor. It is self-supervised (trains on ANY
    sequence, so no dead params), but on language PPL it is expected to be neutral
    — like PredictiveStateHead itself, and like every physics/bio module in the
    switch-matrix. Its measurable value is on continuous-state / trajectory
    prediction (physics metrics), validated in benchmarks/physics_rollout_eval.py.
    It does NOT reduce hallucination (a category error). Default OFF -> not built.
    """

    def __init__(
        self,
        d_model: int,
        phase_dim: int = 32,
        hidden_dim: int = 64,
        depth: int = 2,
        dt: float = 0.1,
        condition_on_context: bool = True,
        context_dim: int = 32,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.phase_dim = int(phase_dim)
        self.dt = float(dt)
        self.condition_on_context = bool(condition_on_context)

        # The BRIDGE: hidden latent <-> physical phase state (q, p).
        self.to_phase = nn.Linear(d_model, 2 * phase_dim)
        self.from_phase = nn.Linear(2 * phase_dim, d_model)
        # Small init (not zero): keep the initial next-latent prediction near the
        # projection scale, the same discipline as PredictiveStateHead.
        for lin in (self.to_phase, self.from_phase):
            nn.init.normal_(lin.weight, std=0.02)
            nn.init.zeros_(lin.bias)

        # The liquid core conditions the potential landscape (context hook).
        ctx = int(context_dim) if self.condition_on_context else 0
        self.context_proj = nn.Linear(d_model, ctx) if self.condition_on_context else None
        if self.context_proj is not None:
            nn.init.normal_(self.context_proj.weight, std=0.02)
            nn.init.zeros_(self.context_proj.bias)

        self.ham = HamiltonianHead(phase_dim, hidden_dim=hidden_dim, depth=depth,
                                   context_dim=ctx)

        # Diagnostic surprise buffer (mirrors PredictiveStateHead.last_pred_error),
        # so this head can feed the same LAVI / monitoring consumers.
        # persistent=True so a checkpointed model resumes with the last observed
        # surprise instead of silently resetting to 0 in a fresh process.
        self.register_buffer("last_pred_error", torch.zeros(()), persistent=True)

    def to_phase_state(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        phase = self.to_phase(x)
        return phase[..., :self.phase_dim], phase[..., self.phase_dim:]

    def forward(
        self,
        x: torch.Tensor,                 # (B, T, d_model) hidden states (post final_norm)
        compute_loss: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Returns (h_pred, loss). h_pred (B,T,d_model) is the predicted NEXT
        hidden latent; loss is the shift-by-1 self-supervised MSE (or None when
        T<=1 / compute_loss False)."""
        q, p = self.to_phase_state(x)
        context = self.context_proj(x) if self.context_proj is not None else None
        # The symplectic field is an autograd gradient (torch.autograd.grad inside
        # ham.step), so it CANNOT run with grad disabled — e.g. under the
        # @torch.no_grad() generate() path. When grad is off, re-enable it locally
        # on detached requires-grad copies of the phase state, then detach the
        # result: no graph leaks out, and the training path is unchanged.
        if torch.is_grad_enabled():
            q1, p1 = self.ham.step(q, p, self.dt, context=context)
        else:
            with torch.enable_grad():
                q_local = q.detach().requires_grad_(True)
                p_local = p.detach().requires_grad_(True)
                ctx_local = context.detach() if context is not None else None
                q1, p1 = self.ham.step(q_local, p_local, self.dt, context=ctx_local)
            q1, p1 = q1.detach(), p1.detach()
        h_pred = self.from_phase(torch.cat([q1, p1], dim=-1))

        loss: Optional[torch.Tensor] = None
        if compute_loss and x.shape[1] > 1:
            pred = h_pred[:, :-1, :]
            target = x[:, 1:, :].detach()
            # Scale-free normalised MSE (2*(1-cos)), matching PredictiveStateHead's
            # convention so the two heads' losses are comparable in magnitude.
            pred_n = F.normalize(pred, dim=-1, eps=1e-8)
            tgt_n = F.normalize(target, dim=-1, eps=1e-8)
            loss = (pred_n - tgt_n).pow(2).sum(dim=-1).mean()
            with torch.no_grad():
                cos = (pred_n * tgt_n).sum(dim=-1).mean()
                self.last_pred_error = ((1.0 - cos) * 0.5).clamp(0.0, 1.0)
        return h_pred, loss
