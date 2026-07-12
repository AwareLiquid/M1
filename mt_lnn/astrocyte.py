"""
mt_lnn/astrocyte.py — slow astrocytic (glial) gating of Hebbian consolidation.

╔══════════════════════════════════════════════════════════════════════════╗
║ RESEARCH — NOT WIRED (architecture audit 2026-07-12).                     ║
║ AstrocyteGate.modulate/consolidation_scale have ZERO live callers — only  ║
║ tests/ and __init__ re-export. It gates a Hebbian term that the ablations ║
║ show is inert-or-harmful: legacy HebbianRegularizer contributes ~8e-5 of  ║
║ the gradient (|ΔPPL|<noise); the refactor HebbianPlasticity path HURTS    ║
║ (val PPL rises monotonically with its share, +9~18) and runs UNCAPPED in  ║
║ train.py (recalibrate() lives only in experiments/). Gating an inert-or-  ║
║ harmful loss adds no value. Decision: ARCHIVE — revisit only if/after     ║
║ Hebbian consolidation is first proven to help. Do not assume it's active. ║
╚══════════════════════════════════════════════════════════════════════════╝

Why this module exists
----------------------
Synapses are not just neuron-to-neuron: the *tripartite synapse* adds an
astrocyte as a third partner (Araque et al. 1999). Astrocytes sense aggregate
neuronal activity, integrate it through **slow intracellular calcium waves**
(seconds to tens of seconds — orders of magnitude slower than a spike), and
release gliotransmitters that *gate* synaptic plasticity. Crucially the gating
is **bidirectional and band-shaped**: a quiescent network and a saturated,
over-driven network both get little glial potentiation, while a sustained,
productive mid-level of activity gets the most (De Pittà et al. 2016) — a
homeostatic, BCM-like inverted-U over the slow calcium variable.

MT-LNN already consolidates via :class:`mt_lnn.plasticity.HebbianRegularizer`,
whose ``base_lr`` (α) sets how strongly co-activation is reinforced. That α is
modulated *fast* (per step, by the LAVI gate, and per signal by the
:class:`~mt_lnn.neuromodulation.NeuromodulationController` dopamine channel).
What was missing is the **slow** partner: a glial timescale that says "the
network has been productively busy for a while now — deepen consolidation" or
"it has been saturated/over-driven — hold back". This module is that partner.

What it does (and does NOT do)
------------------------------
* It is a tiny **stateful slow integrator**: you feed it a per-step scalar
  *activity*, it advances a leaky calcium variable on a slow time constant
  ``tau`` (>> 1 step), and reads out a bounded ``gate`` multiplier from an
  inverted-U over calcium. You then multiply the Hebbian α by that gate.
* It does **not** touch the model's forward pass, define any trainable
  parameter, or import ``model.py``/``plasticity.py``. The plasticity module is
  reached only by **duck typing** on a ``base_lr`` attribute (the same contract
  ``NeuromodulationController.modulate_hebbian`` uses), so it drops in with no
  glue and is removable with no trace.

Design / coupling
-----------------
* **Pure-python, zero trainable state.** Imports only stdlib ``math``/typing —
  no ``torch`` on the hot path. Not an ``nn.Module`` (the global model init pass
  can never touch it). ``activity_of`` is the one optional torch-friendly helper
  and is fully duck-typed.
* **Slow by construction.** A single high-activity step barely moves the gate;
  only activity *sustained over many steps* drives calcium across the band. That
  separation of timescales (fast Hebbian step vs slow glial gate) is the whole
  point and is what the tests pin.
* **Default-neutral / zero-regression.** Nothing in MT-LNN constructs it; if you
  never build it, plasticity is unchanged. The first ``modulate`` call captures
  the regularizer's original α so repeated gating never compounds.

Pinned in tests/test_astrocyte.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

__all__ = [
    "AstrocyteState",
    "AstrocyteGate",
]


def _to_float(x: Any) -> float:
    """Coerce a scalar / 0-d tensor / numpy scalar to a python float.

    Duck-typed so the module never has to import torch: anything exposing
    ``.item()`` (a tensor) is reduced via that; otherwise ``float(x)``.
    """
    item = getattr(x, "item", None)
    if callable(item):
        try:
            return float(item())
        except Exception:
            pass
    return float(x)


@dataclass(frozen=True)
class AstrocyteState:
    """A read-only snapshot of the glial gate after a step.

    Attributes
    ----------
    calcium : the slow leaky-integrated calcium variable (activity units).
    gate : the bounded consolidation multiplier read from the inverted-U
        (``gate_min <= gate <= gate_max``).
    n_steps : how many activity steps have been integrated since construction
        / last ``reset()``.
    """

    calcium: float
    gate: float
    n_steps: int

    def as_dict(self) -> dict:
        return {"calcium": self.calcium, "gate": self.gate, "n_steps": self.n_steps}


class AstrocyteGate:
    """Slow astrocytic calcium gate over Hebbian consolidation strength.

    Maintains a leaky calcium integrator
    ``ca <- (1 - dt/tau) * ca + (dt/tau) * activity`` with a **slow** time
    constant ``tau`` (in steps), and reads out a bounded gate from an
    inverted-U (Gaussian band) centred at ``ca_peak``::

        gate(ca) = gate_min + (gate_max - gate_min) * exp(-(ca - ca_peak)^2 / (2 width^2))

    Quiescent (``ca -> 0``) and saturated (``ca >> ca_peak``) states both yield
    ``~gate_min`` (little/depressed consolidation); the productive mid-band near
    ``ca_peak`` yields ``~gate_max`` (deepened consolidation). This is the
    homeostatic, band-shaped glial modulation of De Pittà et al. (2016).

    Parameters
    ----------
    tau : slow calcium time constant in *steps* (must be > 0; values >> 1 make
        the gate respond only to *sustained* activity, the biological point).
    dt : integration step (default 1.0; ``dt/tau`` is the leak rate, clamped to
        ``<= 1`` so a single step can never overshoot the input).
    ca_peak : calcium level (activity units) of maximal potentiation.
    width : half-width of the productive band (must be > 0).
    gate_min : floor multiplier at the band extremes (>= 0).
    gate_max : peak multiplier at ``ca_peak`` (must be >= ``gate_min``).
    ca_init : initial calcium (default 0.0 = rested).
    """

    def __init__(
        self,
        *,
        tau: float = 20.0,
        dt: float = 1.0,
        ca_peak: float = 1.0,
        width: float = 0.5,
        gate_min: float = 0.25,
        gate_max: float = 1.5,
        ca_init: float = 0.0,
    ):
        if not (tau > 0.0):
            raise ValueError(f"tau must be > 0, got {tau}")
        if not (dt > 0.0):
            raise ValueError(f"dt must be > 0, got {dt}")
        if not (width > 0.0):
            raise ValueError(f"width must be > 0, got {width}")
        if gate_min < 0.0:
            raise ValueError(f"gate_min must be >= 0, got {gate_min}")
        if gate_max < gate_min:
            raise ValueError(
                f"gate_max ({gate_max}) must be >= gate_min ({gate_min})"
            )

        self.tau = float(tau)
        self.dt = float(dt)
        self.ca_peak = float(ca_peak)
        self.width = float(width)
        self.gate_min = float(gate_min)
        self.gate_max = float(gate_max)
        self._ca_init = float(ca_init)

        # Leak rate per step, clamped so one step never overshoots the target.
        self._leak = min(1.0, self.dt / self.tau)

        self._ca = self._ca_init
        self._n_steps = 0
        # Captured on first modulate() so repeated gating never compounds.
        self._captured_base: Optional[float] = None

    # ------------------------------------------------------------------ #
    # core slow dynamics                                                 #
    # ------------------------------------------------------------------ #
    def _gate_of(self, ca: float) -> float:
        """Inverted-U readout: bounded multiplier from the calcium band."""
        z = (ca - self.ca_peak) / self.width
        bump = math.exp(-0.5 * z * z)                  # in (0, 1], peak at ca_peak
        return self.gate_min + (self.gate_max - self.gate_min) * bump

    def step(self, activity: Any) -> float:
        """Integrate one activity sample on the slow calcium timescale.

        ``activity`` may be a float or any 0-d tensor / scalar exposing
        ``.item()``. Returns the *new* gate multiplier after this step.
        """
        a = _to_float(activity)
        if not math.isfinite(a):
            raise ValueError(f"activity must be finite, got {a}")
        # Leaky integration toward the instantaneous activity (slow if tau >> dt).
        self._ca = (1.0 - self._leak) * self._ca + self._leak * a
        self._n_steps += 1
        return self.gate

    def run(self, stream: Iterable[Any]) -> List[float]:
        """Convenience: integrate a sequence of activities, return per-step gates."""
        return [self.step(a) for a in stream]

    # ------------------------------------------------------------------ #
    # read-outs                                                          #
    # ------------------------------------------------------------------ #
    @property
    def calcium(self) -> float:
        return self._ca

    @property
    def gate(self) -> float:
        """Current bounded consolidation multiplier."""
        return self._gate_of(self._ca)

    @property
    def n_steps(self) -> int:
        return self._n_steps

    @property
    def state(self) -> AstrocyteState:
        return AstrocyteState(calcium=self._ca, gate=self.gate, n_steps=self._n_steps)

    def consolidation_scale(self) -> float:
        """Alias for :pyattr:`gate`, named for offline-consolidation callers.

        A :class:`~mt_lnn.sleep_consolidation.SleepWakeConsolidator` can read this
        to let the *slow glial state* deepen or hold back NREM consolidation —
        the same band that gates online Hebbian α also gates offline replay.
        """
        return self.gate

    def reset(self) -> None:
        """Return calcium / step count to their initial values (base stays captured)."""
        self._ca = self._ca_init
        self._n_steps = 0

    # ------------------------------------------------------------------ #
    # duck-typed actuator                                                #
    # ------------------------------------------------------------------ #
    def modulate(self, regularizer: Any, *, base_lr: Optional[float] = None) -> Optional[float]:
        """Scale a Hebbian regularizer's ``base_lr`` by the current glial gate.

        Duck-typed: any object with a writable float ``base_lr`` works (e.g.
        :class:`mt_lnn.plasticity.HebbianRegularizer`); returns ``None`` for an
        object without it, so the call is always safe.

        The *original* α is captured on the first call (or taken from an explicit
        ``base_lr``), so repeated per-window gating multiplies that fixed base by
        the gate rather than compounding the previous gated value.
        """
        if not hasattr(regularizer, "base_lr"):
            return None
        if base_lr is not None:
            self._captured_base = float(base_lr)
        elif self._captured_base is None:
            self._captured_base = float(regularizer.base_lr)
        new_lr = self._captured_base * self.gate
        regularizer.base_lr = new_lr
        return new_lr

    def __repr__(self) -> str:
        return (
            f"AstrocyteGate(tau={self.tau}, ca_peak={self.ca_peak}, "
            f"width={self.width}, gate=[{self.gate_min}, {self.gate_max}], "
            f"calcium={self._ca:.4f}, gate={self.gate:.4f})"
        )

    # ------------------------------------------------------------------ #
    # optional activity helper (duck-typed, no hard torch dependency)    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def activity_of(*tensors: Any) -> float:
        """Derive a scalar activity from one or more activation tensors.

        Mean absolute value across all given tensors — a cheap, sign-agnostic
        proxy for "how busy the network is". Duck-typed on ``.abs().mean()`` /
        ``.item()`` so it works on torch tensors without importing torch; falls
        back to ``abs(float(x))`` for plain scalars.
        """
        if not tensors:
            raise ValueError("activity_of needs at least one tensor")
        vals: List[float] = []
        for t in tensors:
            abs_fn = getattr(t, "abs", None)
            mean_fn = getattr(t, "mean", None)
            if callable(abs_fn) and callable(mean_fn):
                vals.append(_to_float(t.abs().mean()))
            else:
                vals.append(abs(_to_float(t)))
        return sum(vals) / len(vals)
