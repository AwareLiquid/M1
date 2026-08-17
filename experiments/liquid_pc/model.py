"""
experiments/liquid_pc/model.py — the INTEGRATED predictive-coding liquid core,
and parameter-matched baselines, for end-to-end validation.

This is the architecture-integration deliverable: not a bolt-on module, but a
single, always-on system where the liquid core's continuous-time dynamics ARE
the predictive-coding loop. There is no switch and no bypass — predictive coding
is the forward pass.

The integrated model (PCLiquidCore)
-----------------------------------
A hierarchy of ``L`` representation levels ``r_1 … r_L`` (level 0 is the embedded
input), each an O(1)-memory liquid state with its OWN continuous-time constant
``tau_l`` (geometrically slower with depth → the multi-timescale core). Per step:

  1. Top-down generation: level ``l+1`` predicts level ``l`` via ``G_{l+1}``:
         phat_l = G_{l+1}(r_{l+1})
  2. Bottom-up error: the prediction error at each level,
         eps_l = r_l - phat_l
     is what flows UP (only the error propagates, the predictive-coding tenet).
  3. Liquid update (the SAME exp(-dt/tau) kernel as ProtofilamentLTC), driven by
     the bottom-up error from below and pulled by its own top-down error:
         drive_l = tanh( W_l(eps_{l-1}) - eps_l )
         r_l <- decay_l * r_l + (1 - decay_l) * drive_l,   decay_l = exp(-dt/tau_l)

So "predict → error → update" is the recurrence itself. The next-step output is
read from the (multi-timescale) state. Trained on plain next-step MSE — the SAME
objective as every baseline, so any advantage is architectural, not from a
different loss.

Fidelity to the brain-like core
-------------------------------
* Continuous-time, not discrete stepping: the state evolves through the
  exp(-dt/tau) leak; tau is a learnable per-level parameter initialised
  multi-scale, exactly as in the production liquid core.
* O(1) memory / streaming: a single state per level is carried; no KV cache, no
  attention over the past. The whole sequence is processed by recurrence.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# the integrated predictive-coding liquid core                                #
# --------------------------------------------------------------------------- #
class PCLiquidCore(nn.Module):
    """Hierarchical predictive coding embedded in continuous-time liquid dynamics.

    Parameters
    ----------
    d_in : input/output width.
    d : hidden width per level.
    n_levels : number of representation levels (>= 1).
    dt : integration step for the exp(-dt/tau) liquid kernel.
    tau_min, tau_max : range of per-level time constants; level 0 starts near
        tau_min (fast) and the top level near tau_max (slow) — the multi-timescale
        ladder that gives the core its name.
    free_energy_weight : optional weight on the internal sum-of-squared prediction
        errors (a free-energy term). 0.0 (default) keeps the training objective
        identical to the baselines (pure next-step MSE) for a fair comparison.
    """

    def __init__(
        self,
        d_in: int,
        *,
        d: int = 48,
        n_levels: int = 3,
        dt: float = 1.0,
        tau_min: float = 1.0,
        tau_max: float = 40.0,
        free_energy_weight: float = 0.0,
        learn_precision: bool = True,
        precision_init: float | None = None,
        dynamic_precision: bool = False,
        use_astrocyte: bool = False,
        astro_tau: float | None = None,
        astro_ca_peak: float = 0.3,
        astro_width: float = 0.3,
    ):
        super().__init__()
        if n_levels < 1:
            raise ValueError(f"n_levels must be >= 1, got {n_levels}")
        self.d_in = int(d_in)
        self.d = int(d)
        self.n_levels = int(n_levels)
        self.dt = float(dt)
        self.free_energy_weight = float(free_energy_weight)
        self.dynamic_precision = bool(dynamic_precision)
        self.use_astrocyte = bool(use_astrocyte)

        # Level 0 = embedded input (clamped to data each step).
        self.embed = nn.Linear(d_in, d)

        # Top-down generative weights: G[l] maps level l+1 -> a prediction of level l.
        # There are n_levels of them: G[0] predicts the input embedding (level 0)
        # from r_1, …, G[L-1] predicts r_{L-1} from r_L.
        self.generate = nn.ModuleList(
            [nn.Linear(d, d) for _ in range(self.n_levels)]
        )
        # Bottom-up recognition weights: R[l] maps the error below into level l's drive.
        self.recognize = nn.ModuleList(
            [nn.Linear(d, d) for _ in range(self.n_levels)]
        )

        # Per-level log-tau (softplus), initialised geometrically fast->slow.
        if self.n_levels == 1:
            taus = torch.tensor([0.5 * (tau_min + tau_max)])
        else:
            taus = torch.logspace(
                math.log10(tau_min), math.log10(tau_max), self.n_levels
            )
        log_tau_init = torch.log(torch.expm1((taus - tau_min).clamp_min(1e-4)))
        self.log_tau = nn.Parameter(log_tau_init)            # (L,)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)

        # Per-level error PRECISION (inverse-variance), softplus-parameterised.
        # The predictive-coding tenet (Friston free-energy): reliable, high-
        # precision errors drive the state more, noisy low-precision ones less.
        # Here it weights the error in the DRIVE itself (not just an aux loss),
        # so the core can LEARN to damp unreliable errors during self-fed rollout
        # — an architectural fix for rollout instability rather than a training
        # trick. Default init makes softplus(log_precision)==1.0 so a freshly
        # built model is bit-for-bit identical to the precision-free core (clean
        # ablation); training is then free to move precisions away from 1.
        if precision_init is None:
            precision_init = math.log(math.expm1(1.0))       # softplus(.) == 1.0
        logp = torch.full((self.n_levels,), float(precision_init))
        if learn_precision:
            self.log_precision = nn.Parameter(logp)
        else:
            self.register_buffer("log_precision", logp, persistent=True)

        # Optional INPUT-DEPENDENT (dynamic) precision: instead of a single
        # learned scalar per level, the precision is modulated each step by the
        # current context (the level's own state), so the core can raise gain on
        # channels that are momentarily reliable and lower it on noisy ones in
        # real time — Friston's "precision IS attention". A per-level gate adds a
        # log-precision offset: pi_l(t) = softplus(log_precision[l] + gate_l(r_l)).
        # The gate is zero-initialised, so at construction the dynamic model is
        # bit-for-bit identical to the static-precision core (and hence to the
        # precision-free core) — completing the clean ablation chain
        # none -> static -> dynamic.
        if self.dynamic_precision:
            self.prec_gate = nn.ModuleList(
                [nn.Linear(d, 1) for _ in range(self.n_levels)]
            )
            for g in self.prec_gate:
                nn.init.zeros_(g.weight)
                nn.init.zeros_(g.bias)

        # Optional ASTROCYTE consolidation gate (tripartite synapse, Araque 1999;
        # De Pittà 2016). Each level carries a SLOW leaky-calcium variable on a
        # time constant far slower than the state's own tau, and an inverted-U
        # band over that calcium gates how strongly NEW drive is written into the
        # slow state: a productive mid-band deepens consolidation, while quiescent
        # OR saturated (over-driven) channels hold back — a homeostatic brake that
        # protects already-consolidated structure from being overwritten. This
        # targets the one ROBUST, error-barred PCLiquidCore advantage found in the
        # multi-seed pass: less catastrophic forgetting than RNNs. The per-level
        # gate scale is zero-initialised so a fresh astrocyte model is bit-for-bit
        # identical to the non-astrocyte core (the ablation chain extends to
        # none -> static -> dynamic -> astrocyte), and the gate is parameterised
        # additively as gate = 1 + scale * bump so scale=0 is a literal no-op.
        if self.use_astrocyte:
            self.astro_tau = float(astro_tau if astro_tau is not None
                                   else 4.0 * tau_max)
            self.astro_ca_peak = float(astro_ca_peak)
            self.astro_width = float(astro_width)
            # learnable per-level consolidation depth, zero-init -> identity.
            self.astro_scale = nn.Parameter(torch.zeros(self.n_levels))

        # Read next-step prediction from the full multi-timescale state.
        self.readout = nn.Linear(self.n_levels * d, d_in)

    def _decay(self) -> torch.Tensor:
        tau = (F.softplus(self.log_tau) + self.tau_min).clamp(self.tau_min, self.tau_max)
        return torch.exp(-self.dt / tau)                     # (L,)

    def _precision(self) -> torch.Tensor:
        return F.softplus(self.log_precision)                # (L,), > 0

    def forward(
        self, x: torch.Tensor, return_aux: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        """Run the PC-liquid recurrence over ``x`` (B, T, d_in).

        Returns the next-step prediction ``xhat`` (B, T, d_in): ``xhat[:, t]`` is
        the model's prediction of ``x[:, t+1]`` from inputs up to ``t``.
        """
        B, T, _ = x.shape
        decay = self._decay()                                # (L,)
        prec = self._precision()                             # (L,) error precisions
        # Initialise level states to zero.
        r = [x.new_zeros(B, self.d) for _ in range(self.n_levels)]
        # Astrocyte slow-calcium state (one per level), if enabled.
        if self.use_astrocyte:
            ca = [x.new_zeros(B, 1) for _ in range(self.n_levels)]
            astro_leak = min(1.0, self.dt / self.astro_tau)
            # running per-level sum of the consolidation gate (for EWC weighting).
            gate_sum = [x.new_zeros(()) for _ in range(self.n_levels)]

        outs: List[torch.Tensor] = []
        fe_total = x.new_zeros(())
        for t in range(T):
            r0 = torch.tanh(self.embed(x[:, t, :]))          # level-0 = embedded input

            # 1) top-down predictions + 2) bottom-up errors (from current states).
            below = [r0] + r[:-1]                             # level l-1 representation
            eps: List[torch.Tensor] = []
            for l in range(self.n_levels):
                phat = self.generate[l](r[l])                # predict level l-1 from r_l
                eps.append(below[l] - phat)                  # error at level l-1

            # Per-step precision: static scalar per level, or — when dynamic —
            # context-modulated from each level's current state (B, 1), so the
            # gain on each error channel adapts in real time.
            if self.dynamic_precision:
                prec_t = [
                    F.softplus(self.log_precision[l] + self.prec_gate[l](r[l]))
                    for l in range(self.n_levels)
                ]                                            # each (B, 1)
            else:
                prec_t = prec                               # (L,) scalar broadcast

            # Astrocyte slow-calcium update + inverted-U consolidation gate. The
            # calcium integrates each level's sustained activity on a slow tau;
            # the band gate (1 + scale*bump) deepens or holds back how much new
            # drive is written this step (scale=0 -> gate==1 -> no-op).
            if self.use_astrocyte:
                astro_gate: List[torch.Tensor] = []
                for l in range(self.n_levels):
                    act = r[l].abs().mean(dim=-1, keepdim=True)   # (B,1) busy-ness
                    ca[l] = (1.0 - astro_leak) * ca[l] + astro_leak * act
                    z = (ca[l] - self.astro_ca_peak) / self.astro_width
                    bump = torch.exp(-0.5 * z * z)                # (B,1) in (0,1]
                    g_l = 1.0 + self.astro_scale[l] * bump
                    astro_gate.append(g_l)
                    gate_sum[l] = gate_sum[l] + g_l.mean()

            # 3) liquid update: PRECISION-weighted error from below drives, own
            #    precision-weighted top-down error pulls. Weighting the error
            #    residual (not the synaptic map) keeps it a true inverse-variance
            #    term: prec[l] scales error eps[l] everywhere it appears.
            new_r: List[torch.Tensor] = []
            for l in range(self.n_levels):
                bu = self.recognize[l](prec_t[l] * eps[l])   # bottom-up error drive
                td = prec_t[l + 1] * eps[l + 1] if l + 1 < self.n_levels else 0.0
                drive = torch.tanh(bu - td)
                if self.use_astrocyte:
                    drive = astro_gate[l] * drive            # glial consolidation gate
                new_r.append(decay[l] * r[l] + (1.0 - decay[l]) * drive)
            r = new_r

            if self.free_energy_weight > 0.0:
                fe_total = fe_total + sum(
                    (prec_t[i] * e.pow(2).mean() for i, e in enumerate(eps))
                )

            outs.append(self.readout(torch.cat(r, dim=-1)))  # predict next input

        xhat = torch.stack(outs, dim=1)                      # (B, T, d_in)
        if return_aux:
            aux = {"free_energy": fe_total / max(1, T)}
            if self.use_astrocyte:
                # mean consolidation gate per level over the sequence (detached:
                # used as a static per-level importance weight, not a grad path).
                aux["astro_gate_mean"] = torch.stack(
                    [g / max(1, T) for g in gate_sum]
                ).detach()                                   # (L,)
            return xhat, aux
        return xhat

    # ------------------------------------------------------------------ #
    # continual-learning support: calcium-weighted EWC-lite               #
    # ------------------------------------------------------------------ #
    # When a task is finished we "consolidate": estimate each weight's
    # importance via a diagonal-Fisher proxy (squared gradient of the task
    # loss) and anchor the current weights. A later task then pays a
    # quadratic penalty for moving important weights — Elastic Weight
    # Consolidation (Kirkpatrick 2017). The PCLiquidCore-SPECIFIC twist is
    # CALCIUM WEIGHTING: per-level Fisher is scaled by that level's mean
    # astrocyte consolidation gate, so weights in levels the glial gate
    # marked as "consolidated" are protected MORE. This is an architectural
    # capability of the integrated core, not a generic add-on.

    def _param_level(self, name: str) -> Optional[int]:
        """Map a parameter name to its representation level (0..L-1), or None
        for global params (embed/readout/log_tau/log_precision/astro_scale)."""
        for prefix in ("generate.", "recognize.", "prec_gate."):
            if name.startswith(prefix):
                rest = name[len(prefix):]
                try:
                    return int(rest.split(".", 1)[0])
                except ValueError:
                    return None
        return None

    def consolidate(
        self,
        x: torch.Tensor,
        *,
        calcium_weighted: bool = True,
        fisher_batch: int = 8,
    ) -> None:
        """Estimate the diagonal EMPIRICAL Fisher importance on task ``x`` and
        anchor the current weights.

        The empirical Fisher averages PER-(mini)SAMPLE squared gradients rather
        than squaring the gradient of the batch-mean loss. This matters: at a
        task-A minimum the mean gradient ~ 0, so ``(mean grad)**2`` collapses to
        ~0 and the EWC penalty becomes a no-op (verified empirically — plain EWC
        with that proxy did NOT reduce forgetting). Per-sample squared gradients
        stay positive at the minimum (individual samples still pull, they only
        cancel in the mean), giving a meaningful importance estimate.

        Stores ``self._ewc_omega`` (per-param importance) and
        ``self._ewc_anchor`` (per-param frozen reference). If ``calcium_weighted``
        and the astrocyte gate is active, per-level Fisher is scaled by the
        (mean-1 normalised) mean consolidation gate of that level — the
        PCLiquidCore-specific twist.
        """
        # per-level calcium importance from a single full forward (detached).
        level_scale = None
        if calcium_weighted and self.use_astrocyte:
            with torch.no_grad():
                _, aux = self.forward(x, return_aux=True)
            g = aux["astro_gate_mean"].detach().clamp_min(1e-6)
            level_scale = (g / g.mean()).tolist()            # mean-1 normalised

        fisher: dict = {n: torch.zeros_like(p)
                        for n, p in self.named_parameters()}
        n = x.shape[0]
        n_batches = 0
        for i in range(0, n, fisher_batch):
            xb = x[i:i + fisher_batch]
            self.zero_grad(set_to_none=True)
            xhat = self.forward(xb)
            loss = F.mse_loss(xhat[:, :-1, :], xb[:, 1:, :])
            loss.backward()
            for name, p in self.named_parameters():
                if p.grad is not None:
                    fisher[name] += p.grad.detach() ** 2
            n_batches += 1
        n_batches = max(1, n_batches)

        omega: dict = {}
        anchor: dict = {}
        for name, p in self.named_parameters():
            f = fisher[name] / n_batches
            if level_scale is not None:
                lvl = self._param_level(name)
                if lvl is not None and 0 <= lvl < len(level_scale):
                    f = f * level_scale[lvl]
            omega[name] = f
            anchor[name] = p.detach().clone()
        self._ewc_omega = omega
        self._ewc_anchor = anchor
        self.zero_grad(set_to_none=True)

    def ewc_loss(self, lam: float) -> torch.Tensor:
        """Quadratic EWC penalty ``lam * sum omega * (theta - theta*)^2``.

        Returns a zero scalar if no consolidation has been done or ``lam<=0``."""
        dev = self.readout.weight.device
        if lam <= 0.0 or not getattr(self, "_ewc_omega", None):
            return torch.zeros((), device=dev)
        total = torch.zeros((), device=dev)
        params = dict(self.named_parameters())
        for name, om in self._ewc_omega.items():
            p = params.get(name)
            if p is None:
                continue
            total = total + (om * (p - self._ewc_anchor[name]) ** 2).sum()
        return lam * total


# --------------------------------------------------------------------------- #
# baselines (matched-ish parameter budgets, identical next-step objective)     #
# --------------------------------------------------------------------------- #
class _RNNBaseline(nn.Module):
    """GRU/LSTM next-step predictor."""

    def __init__(self, d_in: int, hidden: int, kind: str = "gru", n_layers: int = 1):
        super().__init__()
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}[kind]
        self.rnn = rnn_cls(d_in, hidden, num_layers=n_layers, batch_first=True)
        self.readout = nn.Linear(hidden, d_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(x)                                   # (B, T, hidden)
        return self.readout(h)                               # (B, T, d_in)


def GRUBaseline(d_in, hidden=64, n_layers=1):
    return _RNNBaseline(d_in, hidden, "gru", n_layers)


def LSTMBaseline(d_in, hidden=64, n_layers=1):
    return _RNNBaseline(d_in, hidden, "lstm", n_layers)


class TinyTransformer(nn.Module):
    """Causal Transformer next-step predictor (KV-cache-free training)."""

    def __init__(
        self,
        d_in: int,
        d_model: int = 48,
        n_heads: int = 4,
        n_layers: int = 2,
        max_len: int = 512,
        ff_mult: int = 2,
    ):
        super().__init__()
        self.inp = nn.Linear(d_in, d_model)
        self.pos = nn.Parameter(0.02 * torch.randn(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            batch_first=True,
            dropout=0.0,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.readout = nn.Linear(d_model, d_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h = self.inp(x) + self.pos[:, :T, :]
        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )
        h = self.enc(h, mask=mask)
        return self.readout(h)


def build_models(d_in: int, *, pc_kwargs: dict | None = None) -> dict:
    """Construct the integrated model and baselines at comparable param budgets.

    ``pc_kwargs`` are forwarded to ``PCLiquidCore`` (e.g.
    ``{"dynamic_precision": True}``) so a runner can pick the integrated-model
    variant without touching the baselines.
    """
    pc_kwargs = dict(pc_kwargs or {})
    return {
        "PCLiquidCore": PCLiquidCore(d_in, d=48, n_levels=3, **pc_kwargs),
        "GRU": GRUBaseline(d_in, hidden=70),
        "LSTM": LSTMBaseline(d_in, hidden=60),
        "Transformer": TinyTransformer(
            d_in, d_model=32, n_heads=4, n_layers=2, max_len=160
        ),
    }
