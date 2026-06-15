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

            # 3) liquid update: PRECISION-weighted error from below drives, own
            #    precision-weighted top-down error pulls. Weighting the error
            #    residual (not the synaptic map) keeps it a true inverse-variance
            #    term: prec[l] scales error eps[l] everywhere it appears.
            new_r: List[torch.Tensor] = []
            for l in range(self.n_levels):
                bu = self.recognize[l](prec_t[l] * eps[l])   # bottom-up error drive
                td = prec_t[l + 1] * eps[l + 1] if l + 1 < self.n_levels else 0.0
                drive = torch.tanh(bu - td)
                new_r.append(decay[l] * r[l] + (1.0 - decay[l]) * drive)
            r = new_r

            if self.free_energy_weight > 0.0:
                fe_total = fe_total + sum(
                    (prec_t[i] * e.pow(2).mean() for i, e in enumerate(eps))
                )

            outs.append(self.readout(torch.cat(r, dim=-1)))  # predict next input

        xhat = torch.stack(outs, dim=1)                      # (B, T, d_in)
        if return_aux:
            return xhat, {"free_energy": fe_total / max(1, T)}
        return xhat


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
