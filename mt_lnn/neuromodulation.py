"""
neuromodulation.py — a global multi-neuromodulator regulator for MT-LNN.

╔══════════════════════════════════════════════════════════════════════════╗
║ RESEARCH — NOT WIRED (architecture audit 2026-07-12).                     ║
║ NeuromodulationController is constructed on NO trained/served path — only ║
║ in tests/. Even if constructed, its ACh→GWTB actuator lands on the        ║
║ dynamic-bandwidth gate (gwtb_dynamic_bandwidth), which is default-OFF,    ║
║ has no train.py CLI flag, and is enabled/trained by no config — so        ║
║ set_bandwidth_bias_offset early-returns (a silent no-op). 3 of its 4      ║
║ signal channels (reward/risk/arousal) have no source in an LM loop. Sound ║
║ theory (Yu & Dayan 2005), ZERO empirical evidence in this repo. Decision: ║
║ ARCHIVE — wiring it requires first training the dynamic-bandwidth gate    ║
║ (large, unproven). Do not assume this is active. See the seam audit.      ║
╚══════════════════════════════════════════════════════════════════════════╝

Why this module exists (the "联调中枢" / integration hub)
--------------------------------------------------------
The brain does not run every circuit at a fixed gain. Four diffuse
neuromodulatory systems broadcast slow, global scalars that *retune* the whole
cortex from moment to moment:

    dopamine (DA)        — reward-prediction error; gates synaptic plasticity
                           (how much should I learn from what just happened?).
    acetylcholine (ACh)  — *expected* uncertainty; widens attentional /
                           workspace sampling and boosts feedforward gain when
                           the world is volatile (Yu & Dayan, 2005).
    serotonin (5-HT)     — patience / behavioural inhibition; lengthens the
                           effective integration time-constant, favouring slow
                           deliberation over fast reaction under risk.
    norepinephrine (NE)  — *unexpected* uncertainty / arousal; sets the global
                           response gain and triggers network "reset"
                           (Aston-Jones & Cohen, 2005; Yu & Dayan, 2005).

MT-LNN already has the four *targets* these systems should modulate:

    DA  → HebbianRegularizer.base_lr        (plasticity rate, plasticity.py)
    ACh → GWTBLayer dynamic-bandwidth gate   (workspace ignition, gwtb.py)
    5HT → ProtofilamentLTC time-constant τ   (liquid integration, mt_lnn_layer.py)
    NE  → output response gain               (a scalar multiplier)

This controller is the conductor that turns four observable signals (reward,
surprise, risk, arousal) into four modulatory scalars and *orchestrates* those
existing modules. That orchestration — not any single new operator — is the
"联调" (joint-tuning) deliverable.

Design / coupling (identical discipline to salience_events.py)
--------------------------------------------------------------
* **Zero model coupling.** Pure-Python + stdlib ``math`` only. It never imports
  ``model.py``, ``gwtb.py``, or ``plasticity.py``. It is **not** an
  ``nn.Module`` and holds **zero trainable parameters** — a slow neuromodulatory
  bus, not a learned layer.
* **Duck-typed actuators.** :meth:`modulate_gwtb` / :meth:`modulate_hebbian`
  reach their targets by attribute name, so the controller stays decoupled: a
  target just has to expose the documented hook (e.g. ``set_bandwidth_bias_offset``).
* **Default-off / zero-regression.** Nothing in MT-LNN constructs this
  controller automatically. A caller opts in, feeds it signals, and applies the
  outputs. If you never call it, the model is bit-identical. Resetting the gates
  to neutral (or never applying them) is a clean rollback.
* **Online & adaptive.** Like the salience detector, each channel keeps an EMA
  baseline so it responds to *changes* (z-scored departures), not absolute
  levels — phasic DA/NE encode prediction errors, not raw reward/arousal.

Interface
---------
>>> from mt_lnn.neuromodulation import NeuromodulationController
>>> nm = NeuromodulationController()
>>> state = nm.update(reward=0.8, surprise=0.6, risk=0.2, arousal=0.5)
>>> state.acetylcholine            # ∈ [0,1]
>>> nm.modulate_gwtb(gwtb_layer)    # ACh → workspace bandwidth (in-place)
>>> lr = nm.modulate_hebbian(hebbian_reg)   # DA → plasticity LR (returns new α)

Pinned in tests/test_neuromodulation.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

__all__ = [
    "NeuromodulatorState",
    "NeuromodulationController",
]


# ---------------------------------------------------------------------------
# Output state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NeuromodulatorState:
    """A snapshot of the four neuromodulatory levels, each in ``[0, 1]``.

    0.5 is the neutral / tonic baseline for every channel; departures encode
    phasic responses (e.g. dopamine > 0.5 == better-than-expected reward).

    Attributes
    ----------
    dopamine : float        plasticity drive (reward-prediction error)
    acetylcholine : float   expected-uncertainty / workspace-widening drive
    serotonin : float       patience / time-constant-lengthening drive
    norepinephrine : float  arousal / response-gain drive
    """

    dopamine: float
    acetylcholine: float
    serotonin: float
    norepinephrine: float

    def as_dict(self) -> dict:
        return {
            "dopamine": self.dopamine,
            "acetylcholine": self.acetylcholine,
            "serotonin": self.serotonin,
            "norepinephrine": self.norepinephrine,
        }


# ---------------------------------------------------------------------------
# Small online helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    # Numerically stable logistic squashing into (0, 1).
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class _EMA:
    """Exponential-moving-average tracker of mean and variance.

    Used to z-score each incoming signal against its *recent normal*, so the
    controller reacts to surprising departures rather than absolute magnitude
    (the defining property of phasic neuromodulation).
    """

    __slots__ = ("decay", "eps", "_mean", "_sq", "_init")

    def __init__(self, decay: float, eps: float) -> None:
        self.decay = decay
        self.eps = eps
        self._mean = 0.0
        self._sq = 0.0
        self._init = False

    def z(self, x: float) -> float:
        """Return the z-score of ``x`` against the current baseline, then update."""
        if not self._init:
            self._mean, self._sq, self._init = x, x * x, True
            return 0.0
        sigma = math.sqrt(max(self._sq - self._mean * self._mean, self.eps))
        z = (x - self._mean) / sigma
        d = self.decay
        self._mean = d * self._mean + (1.0 - d) * x
        self._sq = d * self._sq + (1.0 - d) * (x * x)
        return z

    @property
    def mean(self) -> float:
        return self._mean


# ---------------------------------------------------------------------------
# NeuromodulationController
# ---------------------------------------------------------------------------

class NeuromodulationController:
    """Turn observable signals into four global neuromodulatory scalars.

    Each :meth:`update` consumes up to four optional signals and returns a
    :class:`NeuromodulatorState`. Any signal left as ``None`` holds its channel
    at the neutral 0.5 baseline, so a caller can drive only the gates it has a
    signal for.

    Parameters
    ----------
    ema_decay : float
        Baseline retention in ``(0, 1)`` for every channel's z-scoring tracker;
        higher = slower adaptation (longer "recent normal").
    sensitivity : float
        Global slope on the z-score → level sigmoid. Higher = sharper phasic
        swings around the 0.5 baseline.
    bandwidth_span : float
        Maximum magnitude (in gate-bias units) of the ACh → GWT bandwidth
        offset. ACh = 1.0 adds +span (widest workspace), ACh = 0.0 adds -span
        (narrowest), ACh = 0.5 adds 0 (un-modulated).
    plasticity_span : float
        DA scales the Hebbian learning rate in ``[1-span, 1+span]`` around its
        base value (DA = 0.5 → ×1.0, i.e. no change).
    tau_span : float
        5-HT lengthens the effective time-constant by up to this fraction
        (5-HT = 1.0 → τ × (1+span); 5-HT = 0.5 → τ × 1.0).
    gain_span : float
        NE scales the response gain in ``[1-span, 1+span]`` (NE = 0.5 → ×1.0).
    eps : float
        Variance floor for the z-score, keeps it finite on a flat baseline.
    """

    def __init__(
        self,
        *,
        ema_decay: float = 0.9,
        sensitivity: float = 1.0,
        bandwidth_span: float = 4.0,
        plasticity_span: float = 1.0,
        tau_span: float = 1.0,
        gain_span: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        if not (0.0 < ema_decay < 1.0):
            raise ValueError(f"ema_decay must be in (0, 1), got {ema_decay}")
        if sensitivity <= 0.0:
            raise ValueError(f"sensitivity must be > 0, got {sensitivity}")
        for name, val in (
            ("bandwidth_span", bandwidth_span),
            ("plasticity_span", plasticity_span),
            ("tau_span", tau_span),
            ("gain_span", gain_span),
        ):
            if val < 0.0:
                raise ValueError(f"{name} must be >= 0, got {val}")
        if not (0.0 <= plasticity_span <= 1.0):
            raise ValueError(f"plasticity_span must be in [0, 1], got {plasticity_span}")
        if not (0.0 <= gain_span <= 1.0):
            raise ValueError(f"gain_span must be in [0, 1], got {gain_span}")

        self.ema_decay = float(ema_decay)
        self.sensitivity = float(sensitivity)
        self.bandwidth_span = float(bandwidth_span)
        self.plasticity_span = float(plasticity_span)
        self.tau_span = float(tau_span)
        self.gain_span = float(gain_span)
        self.eps = float(eps)

        # One EMA z-scorer per phasic channel.
        self._reward = _EMA(self.ema_decay, self.eps)    # → dopamine
        self._surprise = _EMA(self.ema_decay, self.eps)  # → acetylcholine
        self._risk = _EMA(self.ema_decay, self.eps)      # → serotonin
        self._arousal = _EMA(self.ema_decay, self.eps)   # → norepinephrine

        # Last computed state (neutral until the first update).
        self._state = NeuromodulatorState(0.5, 0.5, 0.5, 0.5)
        self._n_updates = 0

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def update(
        self,
        *,
        reward: Optional[float] = None,
        surprise: Optional[float] = None,
        risk: Optional[float] = None,
        arousal: Optional[float] = None,
    ) -> NeuromodulatorState:
        """Feed the four observable signals; return the new neuromodulator state.

        Parameters
        ----------
        reward :
            A scalar reward / value signal. Its *prediction error* (departure
            above the recent baseline) drives **dopamine** — phasic DA encodes
            "better than expected", which should boost plasticity.
        surprise :
            A non-negative salience / prediction-error magnitude (e.g.
            ``world_model.last_pred_error``). A rise above baseline raises
            **acetylcholine** → wider workspace ignition under volatility.
        risk :
            An aversive / threat / cost signal. A rise raises **serotonin** →
            longer time-constants → more patient, inhibited dynamics.
        arousal :
            An overall demand / activation signal. A rise raises
            **norepinephrine** → higher response gain (and, biologically, a
            network reset).
        """
        s = self.sensitivity

        da = self._state.dopamine
        ach = self._state.acetylcholine
        ht = self._state.serotonin
        ne = self._state.norepinephrine

        if reward is not None:
            da = _sigmoid(s * self._reward.z(float(reward)))
        if surprise is not None:
            ach = _sigmoid(s * self._surprise.z(float(surprise)))
        if risk is not None:
            ht = _sigmoid(s * self._risk.z(float(risk)))
        if arousal is not None:
            ne = _sigmoid(s * self._arousal.z(float(arousal)))

        self._state = NeuromodulatorState(da, ach, ht, ne)
        self._n_updates += 1
        return self._state

    @property
    def state(self) -> NeuromodulatorState:
        """The most recently computed neuromodulator state."""
        return self._state

    @property
    def n_updates(self) -> int:
        return self._n_updates

    def reset(self) -> None:
        """Clear all baselines and return every channel to neutral (0.5)."""
        self._reward = _EMA(self.ema_decay, self.eps)
        self._surprise = _EMA(self.ema_decay, self.eps)
        self._risk = _EMA(self.ema_decay, self.eps)
        self._arousal = _EMA(self.ema_decay, self.eps)
        self._state = NeuromodulatorState(0.5, 0.5, 0.5, 0.5)
        self._n_updates = 0

    # ------------------------------------------------------------------
    # Modulation read-outs (the four scalars → target-space units)
    # ------------------------------------------------------------------

    def bandwidth_bias_offset(self) -> float:
        """ACh → additive offset for the GWT dynamic-bandwidth gate bias.

        Maps acetylcholine ``[0,1]`` linearly to ``[-span, +span]``: ACh = 0.5
        (neutral) → 0.0 (no modulation), ACh → 1 widens, ACh → 0 narrows.
        """
        return self.bandwidth_span * 2.0 * (self._state.acetylcholine - 0.5)

    def plasticity_scale(self) -> float:
        """DA → multiplier for the Hebbian learning rate.

        Maps dopamine ``[0,1]`` to ``[1-span, 1+span]``; DA = 0.5 → ×1.0.
        """
        return 1.0 + self.plasticity_span * 2.0 * (self._state.dopamine - 0.5)

    def time_constant_scale(self) -> float:
        """5-HT → multiplier for the effective integration time-constant τ.

        Serotonin only *lengthens* τ (patience), so the multiplier is ``>= 1``:
        maps ``[0,1]`` to ``[1, 1+span]``; 5-HT = 0.5 → ×1.0 by mapping the
        lower half to no change. (Below-baseline serotonin does not shorten τ —
        we never speed the liquid core below its trained constant.)
        """
        return 1.0 + self.tau_span * max(0.0, self._state.serotonin - 0.5) * 2.0

    def gain_scale(self) -> float:
        """NE → multiplier for the response gain.

        Maps norepinephrine ``[0,1]`` to ``[1-span, 1+span]``; NE = 0.5 → ×1.0.
        """
        return 1.0 + self.gain_span * 2.0 * (self._state.norepinephrine - 0.5)

    # ------------------------------------------------------------------
    # Actuators (duck-typed; never import the targets)
    # ------------------------------------------------------------------

    def modulate_gwtb(self, gwtb_layer) -> float:
        """Apply the ACh gate to a GWTB layer's dynamic-bandwidth bias, in place.

        Duck-typed on ``set_bandwidth_bias_offset`` so this module never imports
        ``gwtb.py``. No-op (returns 0.0) if the layer lacks the hook or has the
        dynamic bandwidth disabled — keeping the integration strictly opt-in.
        Returns the offset that was applied.
        """
        offset = self.bandwidth_bias_offset()
        setter = getattr(gwtb_layer, "set_bandwidth_bias_offset", None)
        if callable(setter):
            setter(offset)
        return offset

    def modulate_hebbian(self, hebbian_reg) -> Optional[float]:
        """Apply the DA gate to a HebbianRegularizer's learning rate, in place.

        Duck-typed on ``base_lr``. Scales the *current* base_lr by
        :meth:`plasticity_scale`. Returns the new base_lr, or ``None`` if the
        target has no ``base_lr`` attribute. (Caller is responsible for keeping
        a reference base_lr if it wants to re-scale from a fixed value rather
        than compounding.)
        """
        if not hasattr(hebbian_reg, "base_lr"):
            return None
        new_lr = float(hebbian_reg.base_lr) * self.plasticity_scale()
        hebbian_reg.base_lr = new_lr
        return new_lr

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def run(self, stream: Sequence[dict]) -> List[NeuromodulatorState]:
        """Reset, then drive a whole sequence of signal dicts; collect states.

        Each item is a dict with any of ``reward / surprise / risk / arousal``.
        Handy for deterministic testing and offline analysis.
        """
        self.reset()
        out: List[NeuromodulatorState] = []
        for item in stream:
            out.append(self.update(**item))
        return out

    def __repr__(self) -> str:
        s = self._state
        return (
            "NeuromodulationController("
            f"DA={s.dopamine:.3f}, ACh={s.acetylcholine:.3f}, "
            f"5HT={s.serotonin:.3f}, NE={s.norepinephrine:.3f}, "
            f"updates={self._n_updates})"
        )
