"""
mt_lnn/stdp_ops.py -- composable spike-timing-dependent-plasticity (STDP)
operators: local, backprop-free weight updates from spike timing (P0 learning).

Why this module exists (and how it relates to plasticity.py)
------------------------------------------------------------
``plasticity.py`` deliberately does NOT use biological STDP, and for a good
reason it states up front: the ``ProtofilamentLTC`` core evolves in *continuous
time* (``h_t = e^{-dt/tau} h_{t-1} + ...``); there are no discrete pre/post spike
events to time, so STDP would mean artificially discretising the LTC and
destroying its defining property. That module therefore implements Hebbian
consolidation at the *loss* level for the continuous core.

But the architecture is not only the continuous core. The salience layer
(``salience_events.py``) emits genuinely **discrete, timestamped ignition
events**, and the L2 place-code writes are event-like associative updates. Those
are exactly where spike *timing* is defined and where STDP -- a *local,
backprop-free* learning rule -- is the right tool. This module supplies that
rule as a composable operator, **parallel to** (not a replacement for)
``plasticity.py``: the loss-level Hebbian governs the smooth core, event-driven
STDP governs the discrete event streams.

The rule
--------
The canonical asymmetric exponential STDP window, with ``dt = t_post - t_pre``::

    W(dt) =  A_plus  * exp(-dt / tau_plus)     for dt > 0   (pre BEFORE post: LTP)
    W(dt) = -A_minus * exp( dt / tau_minus)    for dt < 0   (post before pre: LTD)
    W(0)  =  0                                              (simultaneous: no causal order)

"Pre before post" (causal, the pre-synaptic spike helped cause the post one) is
*potentiated*; the reverse (acausal) is *depressed*. The strength decays
exponentially with the timing gap. This is local in time and uses no global loss
gradient -- the learning signal is the spike timing itself.

Two equivalent computations, both provided and cross-checked:
* :func:`pairwise_stdp` -- the explicit all-to-all sum of ``W(t_post - t_pre)``
  over every pre/post spike pair (the definition);
* :func:`stdp_trace_update` -- the standard online **eligibility-trace** form
  (decaying pre/post traces; a post spike reads the pre trace for LTP, a pre
  spike reads the post trace for LTD). ``O(T)`` over a spike train and the way
  STDP is actually run on hardware; provably equal to the all-pairs sum (pinned
  in the tests).

Design / coupling
-----------------
* Pure functions + a frozen ``STDPParams`` dataclass. **Stateless, zero trainable
  parameters**, nothing is an ``nn.Module``. Imports only ``torch`` /
  ``dataclasses`` / ``math``; **never imports model.py**.
* The window value is differentiable in the timing gap; the trace update is
  differentiable in the (real-valued) spike amplitudes. Spike *counts* are
  discrete -- documented, not hidden.
* Honest scope: this is the plasticity *rule* (the operator). Wiring it to gate
  L2 writes on salience ignitions is a later integration step, kept separate so
  this layer stays a pinned, model-free primitive.

Pinned against the analytic STDP window in ``tests/test_stdp_ops.py`` and over
the whole input space in ``tests/test_stdp_ops_properties.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = [
    "STDPParams",
    "stdp_kernel",
    "window_integral",
    "pairwise_stdp",
    "stdp_trace_update",
    "STDPResult",
]


@dataclass(frozen=True)
class STDPParams:
    """Hyper-parameters of the exponential STDP window.

    Attributes
    ----------
    a_plus, a_minus : float
        Maximum potentiation / depression amplitudes (>= 0).
    tau_plus, tau_minus : float
        Potentiation / depression time constants (> 0), in the same time unit as
        the spike times / ``dt``.
    dt : float
        Time step for the trace form (the spacing between consecutive entries of
        a spike train). Unused by :func:`pairwise_stdp`, which takes real times.
    """

    a_plus: float = 1.0
    a_minus: float = 1.0
    tau_plus: float = 20.0
    tau_minus: float = 20.0
    dt: float = 1.0

    def __post_init__(self) -> None:
        if self.a_plus < 0 or self.a_minus < 0:
            raise ValueError("a_plus and a_minus must be non-negative")
        if self.tau_plus <= 0 or self.tau_minus <= 0:
            raise ValueError("tau_plus and tau_minus must be positive")
        if self.dt <= 0:
            raise ValueError("dt must be positive")


# --------------------------------------------------------------------------- #
# the window                                                                  #
# --------------------------------------------------------------------------- #
def stdp_kernel(delta_t: torch.Tensor, params: STDPParams = STDPParams()) -> torch.Tensor:
    """The STDP weight change ``W(dt)`` for ``dt = t_post - t_pre`` (elementwise).

    Potentiation for ``dt > 0`` (pre leads post), depression for ``dt < 0``, and
    exactly ``0`` at ``dt == 0`` (simultaneous spikes carry no causal order).
    Differentiable in ``delta_t`` away from the origin.
    """
    dt = torch.as_tensor(delta_t, dtype=torch.get_default_dtype()) \
        if not torch.is_tensor(delta_t) else delta_t
    pot = params.a_plus * torch.exp(-dt.clamp_min(0.0) / params.tau_plus)
    dep = -params.a_minus * torch.exp(dt.clamp_max(0.0) / params.tau_minus)
    out = torch.where(dt > 0, pot,
                      torch.where(dt < 0, dep, torch.zeros_like(dt)))
    return out


def window_integral(params: STDPParams = STDPParams()) -> tuple:
    """Analytic areas under the two window lobes: ``(ltp_area, ltd_area)``.

    ``ltp_area = A_plus * tau_plus`` (integral of the potentiation branch over
    ``dt in (0, inf)``) and ``ltd_area = A_minus * tau_minus`` likewise for
    depression. A window is *balanced* (no net drift under uncorrelated Poisson
    spiking) iff these are equal -- a classic STDP stability condition. Returns
    Python floats.
    """
    return (params.a_plus * params.tau_plus, params.a_minus * params.tau_minus)


# --------------------------------------------------------------------------- #
# all-to-all pairwise form (the definition)                                   #
# --------------------------------------------------------------------------- #
def pairwise_stdp(pre_times: torch.Tensor, post_times: torch.Tensor,
                  params: STDPParams = STDPParams()) -> torch.Tensor:
    """Net weight change for one synapse from explicit spike *times*.

    Sums ``W(t_post - t_pre)`` over every (pre, post) spike pair -- the all-to-all
    interaction scheme. ``pre_times`` / ``post_times`` are 1-D tensors of spike
    times (any units consistent with the taus). Returns a scalar tensor.
    Differentiable in the spike times.
    """
    pre = torch.as_tensor(pre_times).flatten()
    post = torch.as_tensor(post_times).flatten()
    if pre.numel() == 0 or post.numel() == 0:
        ref = pre if pre.numel() else post           # match an input's dtype/device
        dtype = ref.dtype if ref.is_floating_point() else torch.get_default_dtype()
        return torch.zeros((), dtype=dtype, device=ref.device)
    dt = post.unsqueeze(-1) - pre.unsqueeze(0)        # (n_post, n_pre)
    return stdp_kernel(dt, params).sum()


# --------------------------------------------------------------------------- #
# online eligibility-trace form (O(T), how it actually runs)                  #
# --------------------------------------------------------------------------- #
@dataclass
class STDPResult:
    """Outcome of a trace-based STDP sweep over a spike train.

    Attributes
    ----------
    delta_w : (N_pre, N_post)
        Net synaptic weight change ``potentiation - depression``.
    potentiation : (N_pre, N_post)
        Accumulated LTP (>= 0): each post spike credited by the pre trace.
    depression : (N_pre, N_post)
        Accumulated LTD (>= 0): each pre spike debited by the post trace.
    """

    delta_w: torch.Tensor
    potentiation: torch.Tensor
    depression: torch.Tensor


def stdp_trace_update(pre_spikes: torch.Tensor, post_spikes: torch.Tensor,
                      params: STDPParams = STDPParams()) -> STDPResult:
    """Online eligibility-trace STDP over aligned spike trains.

    Parameters
    ----------
    pre_spikes : (T,) or (T, N_pre)
        Spike train(s) on a uniform ``params.dt`` grid (typically {0, 1}, but any
        non-negative amplitudes are allowed and treated linearly).
    post_spikes : (T,) or (T, N_post)
        Post-synaptic spike train(s), same length ``T``.
    params : STDPParams

    Returns
    -------
    STDPResult with ``(N_pre, N_post)`` matrices.

    Method
    ------
    Maintains decaying traces ``x`` (pre) and ``y`` (post). At each step, AFTER
    decay but BEFORE adding the current step's spikes:
      * a post spike potentiates by ``a_plus * x`` (pre spikes strictly earlier);
      * a pre spike depresses by ``a_minus * y`` (post spikes strictly earlier).
    Simultaneous pre/post spikes (``dt = 0``) therefore contribute nothing,
    exactly matching :func:`stdp_kernel` at the origin. Provably equal to
    :func:`pairwise_stdp` summed over all pairs (cross-checked in the tests).
    """
    pre = torch.as_tensor(pre_spikes)
    post = torch.as_tensor(post_spikes)
    if pre.dim() == 1:
        pre = pre.unsqueeze(-1)                        # (T, 1)
    if post.dim() == 1:
        post = post.unsqueeze(-1)
    if pre.shape[0] != post.shape[0]:
        raise ValueError(
            f"pre and post spike trains must share T; got {pre.shape[0]} vs "
            f"{post.shape[0]}"
        )
    T, n_pre = pre.shape
    n_post = post.shape[1]
    dtype = torch.promote_types(pre.dtype, post.dtype)
    if not dtype.is_floating_point:
        dtype = torch.get_default_dtype()
    pre = pre.to(dtype)
    post = post.to(dtype)

    decay_pre = math.exp(-params.dt / params.tau_plus)
    decay_post = math.exp(-params.dt / params.tau_minus)

    x = torch.zeros(n_pre, dtype=dtype, device=pre.device)     # pre trace
    y = torch.zeros(n_post, dtype=dtype, device=pre.device)    # post trace
    pot = torch.zeros(n_pre, n_post, dtype=dtype, device=pre.device)
    dep = torch.zeros(n_pre, n_post, dtype=dtype, device=pre.device)

    for t in range(T):
        x = x * decay_pre
        y = y * decay_post
        pre_t = pre[t]                                  # (n_pre,)
        post_t = post[t]                                # (n_post,)
        # LTP: post spike now reads the pre trace (pre spikes strictly before)
        pot = pot + params.a_plus * torch.outer(x, post_t)
        # LTD: pre spike now reads the post trace (post spikes strictly before)
        dep = dep + params.a_minus * torch.outer(pre_t, y)
        # add today's spikes (so they only affect strictly later steps)
        x = x + pre_t
        y = y + post_t

    return STDPResult(delta_w=pot - dep, potentiation=pot, depression=dep)
