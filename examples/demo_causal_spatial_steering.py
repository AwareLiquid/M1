"""
examples/demo_causal_spatial_steering.py -- L3 causal spatial steering demo.

What this shows (the roadshow money shot)
------------------------------------------
An agent moves along a smooth path through a 2-D arena. Its recurrent *belief
state* lives on a low-dimensional manifold -- a 2-D plane (the "legal causal
subspace") embedded in a high-D activation space. As long as the agent moves
consistently every state stays on that plane and stays "on-manifold".

Part-way through we inject a causal break: a one-step *teleport* hallucination
that shoves the belief state OFF the legal plane (a physically impossible jump
that smooth motion could never produce). In a recurrent system that error
*persists* -- once you believe an impossible thing, every later step is built on
top of it. We roll the trajectory forward twice and compare:

  * WITHOUT steering -- the off-plane error rides forward unchanged; the belief
    stays permanently far off the legal manifold. A consistency detector even
    goes quiet again once the jump becomes a steady offset -- yet the belief is
    now corrupt forever.
  * WITH steering -- :class:`~mt_lnn.causal_steering.CausalActivationSteerer`
    detects the jump and orthogonally projects the illegal off-plane component
    out the moment it happens, snapping the agent back onto its legal manifold
    before the error can propagate.

This is the STARS idea (Stiefel / orthonormal-frame activation steering) applied
to *spatial* causal consistency: steer along the legal manifold, never off it.
The headline number is the off-manifold drift -- how "physically impossible" the
agent's current belief is -- which steering returns to ~0 while the unsteered
baseline carries it forever.

Decoupling (zhu yi ou he)
-------------------------
* Uses ONLY the public ``causality`` + ``causal_steering`` APIs. It does NOT
  import or touch the MT-LNN backbone (``model.py``) -- the demo owns its own
  recurrent state, which is exactly where corrective re-feed belongs (see the
  note in ``spatial_reasoning.py``). Zero new model coupling.
* Fully deterministic given ``--seed`` so the printed numbers are reproducible.
* Runs on CPU in well under a second.

Run::

    python examples/demo_causal_spatial_steering.py
    python examples/demo_causal_spatial_steering.py --plot
    python examples/demo_causal_spatial_steering.py --seed 7 --break-step 18
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from mt_lnn.causality import CausalConsistencyChecker
from mt_lnn.causal_steering import CausalActivationSteerer


# ---------------------------------------------------------------------------
# 1. A spatial belief trajectory and the legal manifold it should live on
# ---------------------------------------------------------------------------

@dataclass
class Scene:
    """A legal spatial trajectory plus the off-manifold teleport to inject.

    Attributes
    ----------
    legal : ``(T, D)`` the agent's belief state on a clean, consistent path.
    coords : ``(T, 2)`` the legal arena position each step (``legal`` encodes it).
    basis : ``(2, D)`` orthonormal rows spanning the legal subspace -- the plane
        the consistent trajectory moves in. A point on the Stiefel manifold.
    mean : ``(D,)`` manifold offset (legal = mean + coords @ basis + noise).
    offdir : ``(D,)`` a unit direction ORTHOGONAL to the legal plane -- the
        illegal direction the hallucination jumps along.
    break_step : the step at which the one-time teleport is injected; the error
        then persists for the rest of the trajectory.
    teleport : magnitude of the off-plane teleport.
    """

    legal: torch.Tensor
    coords: torch.Tensor
    basis: torch.Tensor
    mean: torch.Tensor
    offdir: torch.Tensor
    break_step: int
    teleport: float

    def off_manifold(self, state: torch.Tensor) -> float:
        """L2 norm of the component of ``state`` OUTSIDE the legal plane."""
        centred = state - self.mean
        in_plane = self.basis.t() @ (self.basis @ centred)
        return float((centred - in_plane).norm())

    def decode(self, state: torch.Tensor) -> Tuple[float, float]:
        """Project ``state`` onto the legal plane -> (x, y) arena coordinates."""
        c = self.basis @ (state - self.mean)
        return float(c[0]), float(c[1])


def build_scene(
    *,
    n_steps: int = 32,
    d_model: int = 48,
    arena: float = 2.2,
    break_step: int = 18,
    teleport_scale: float = 6.0,
    noise: float = 0.003,
    seed: int = 0,
) -> Scene:
    """Generate a smooth arena path embedded on a random 2-D plane in D-space,
    and pick an orthogonal off-plane direction for the injected teleport."""
    g = torch.Generator().manual_seed(seed)

    # Legal subspace: 2 orthonormal directions in D-space (Stiefel point) + offset.
    rand = torch.randn(d_model, 2, generator=g)
    basis_cols, _ = torch.linalg.qr(rand)               # (D, 2) orthonormal columns
    basis = basis_cols.t().contiguous()                 # (2, D) orthonormal rows
    mean = torch.randn(d_model, generator=g) * 0.1      # manifold offset

    # Smooth arena path: an ellipse the agent walks (clearly exercises both legal
    # dims within any window, so the legal path itself reads as consistent).
    t = torch.linspace(0.0, 1.0, n_steps)
    radius = 0.32 * arena
    centre = 0.5 * arena
    coords = torch.stack([
        centre + radius * torch.cos(2.0 * torch.pi * t),
        centre + 0.9 * radius * torch.sin(2.0 * torch.pi * t),
    ], dim=1)                                            # (T, 2)

    legal = mean.unsqueeze(0) + coords @ basis           # (T, D)
    legal = legal + noise * torch.randn(legal.shape, generator=g)

    # Off-plane unit direction (orthogonal to the legal plane).
    u = torch.randn(d_model, generator=g)
    u = u - basis.t() @ (basis @ u)                      # strip in-plane part
    u = u / (u.norm() + 1e-8)

    return Scene(
        legal=legal, coords=coords, basis=basis, mean=mean, offdir=u,
        break_step=break_step, teleport=teleport_scale * radius,
    )


# ---------------------------------------------------------------------------
# 2. Recurrent rollout through the checker, optionally steering
# ---------------------------------------------------------------------------

@dataclass
class StepLog:
    step: int
    score: float
    applied: bool
    correction_norm: float
    off_manifold: float          # distance of the propagated belief from the legal plane
    decoded: Tuple[float, float]


def rollout(
    scene: Scene,
    steerer: Optional[CausalActivationSteerer],
    *,
    window: int = 8,
    floor: float = 0.5,
    ema_alpha: float = 0.6,
    energy_keep: float = 0.99,
) -> List[StepLog]:
    """Roll the trajectory forward one step at a time.

    The error is *recurrent*: the one-time off-plane teleport at ``break_step``
    is added to a carried ``drift`` that rides forward on every subsequent step
    -- exactly how a hallucination persists in a real recurrent state. The
    canonical STARS-style actuator loop is:

      1. inject the teleport (once, at ``break_step``) into the carried drift;
      2. ``checker.update(raw)`` scores the incoming state against the prior
         clean history (the score is computed *before* the raw point is appended,
         so it is an honest "does this step break the trajectory?" signal);
      3. if a break is detected, orthogonally project the illegal off-plane drift
         out -- but first pop the just-appended raw point so the projection
         subspace is the *clean* prior manifold, not one already contaminated by
         the break (otherwise the projection would remove nothing);
      4. ingest the (corrected) state instead, and subtract the removed component
         from the carried drift so it never propagates forward.

    With ``steerer=None`` this is a pure detector baseline: the teleport is never
    removed, so the belief stays permanently off-manifold.

    The pop / re-append touches ``checker._history`` directly: that is the demo
    owning its own recurrent state (ingest the corrected state, not the raw one).
    The detector / actuator themselves stay generic and model-free.
    """
    checker = CausalConsistencyChecker(
        window=window, method="subspace", threshold=floor,
        ema_alpha=ema_alpha, energy_keep=energy_keep,
    )
    logs: List[StepLog] = []
    drift = torch.zeros_like(scene.mean)                 # carried off-plane error
    for i in range(scene.legal.shape[0]):
        if i == scene.break_step:
            drift = drift + scene.teleport * scene.offdir  # one-time teleport
        raw = scene.legal[i] + drift

        score = checker.update(raw)                       # detect vs clean prior history

        applied = False
        corr_norm = 0.0
        propagated = raw
        if steerer is not None:
            checker._history.pop()                        # remove polluting raw point
            res = steerer.steer(raw, checker, score=score)
            if res.applied:
                applied = True
                corr_norm = res.correction_norm
                propagated = res.steered
                drift = drift - (raw - propagated)        # stop the error propagating
            checker._history.append(                      # ingest the kept state
                propagated.detach().float().reshape(-1).clone())

        logs.append(StepLog(
            step=i, score=float(score), applied=applied,
            correction_norm=float(corr_norm),
            off_manifold=scene.off_manifold(propagated),
            decoded=scene.decode(propagated),
        ))
    return logs


# ---------------------------------------------------------------------------
# 3. Report
# ---------------------------------------------------------------------------

def _segment_mean(logs: List[StepLog], lo: int, hi: int, key: str) -> float:
    seg = [getattr(l, key) for l in logs if lo <= l.step < hi]
    return sum(seg) / max(1, len(seg))


def print_report(base: List[StepLog], steered: List[StepLog], scene: Scene) -> dict:
    bs = scene.break_step
    T = scene.legal.shape[0]
    print("=" * 84)
    print("L3 causal spatial steering  (STARS-inspired activation steering)")
    print("=" * 84)
    print(f"[scene] {T} steps | belief dim D={scene.legal.shape[1]} | legal subspace k=2")
    print(f"[scene] one-time off-manifold teleport at step {bs} (then persists)")
    print()
    print(f"{'step':>4} | {'score base':>10} | {'score steer':>11} | {'steer':>5} | "
          f"{'off-manifold base':>17} | {'off-manifold steer':>18}")
    print("-" * 84)
    for b, s in zip(base, steered):
        tag = "TELEPORT" if b.step == bs else ("off" if b.step > bs else "")
        flag = "FIRE" if s.applied else "."
        print(f"{b.step:>4} | {b.score:>10.3f} | {s.score:>11.3f} | {flag:>5} | "
              f"{b.off_manifold:>17.3f} | {s.off_manifold:>16.3f}  {tag}")

    base_pre = _segment_mean(base, 0, bs, "off_manifold")
    steer_pre = _segment_mean(steered, 0, bs, "off_manifold")
    base_post = _segment_mean(base, bs, T, "off_manifold")
    steer_post = _segment_mean(steered, bs, T, "off_manifold")
    n_fired = sum(1 for s in steered if s.applied)

    print("-" * 84)
    print("off-manifold drift  (how physically impossible the belief is; lower = better)")
    print(f"  before teleport : base {base_pre:7.3f}   steered {steer_pre:7.3f}")
    print(f"  after  teleport : base {base_post:7.3f}   steered {steer_post:7.3f}  "
          f"(lasting damage if unsteered)")
    print(f"  steering fired on {n_fired} step(s)")

    recovered = steer_post < 0.2 * max(base_post, 1e-6)
    damaged = base_post > 0.5
    verdict = recovered and damaged and n_fired > 0
    print()
    print("VERDICT:", "[OK] steering snaps the agent back onto its legal causal "
          "manifold; the unsteered belief stays permanently corrupted"
          if verdict else "[WARN] no clear recovery -- tune break/floor/teleport")
    print("=" * 84)
    return {
        "base_pre": base_pre, "steer_pre": steer_pre,
        "base_post": base_post, "steer_post": steer_post,
        "n_fired": n_fired, "verdict": bool(verdict),
    }


def maybe_plot(base: List[StepLog], steered: List[StepLog], scene: Scene,
               path: str) -> None:
    """Save a 2-panel figure: consistency score and off-manifold drift vs step."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plot] matplotlib unavailable ({e}); skipping figure.")
        return

    bs = scene.break_step
    steps = [l.step for l in base]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax0.plot(steps, [l.score for l in base], "o-", label="no-steer", color="#c0392b")
    ax0.plot(steps, [l.score for l in steered], "s-", label="steered", color="#1f8a4c")
    ax0.axvline(bs, color="grey", ls="--", alpha=0.6, label="teleport")
    ax0.set(xlabel="step", ylabel="consistency score", ylim=(-0.05, 1.05),
            title="Consistency detector")
    ax0.legend(loc="lower left", fontsize=8)

    ax1.plot(steps, [l.off_manifold for l in base], "o-", label="no-steer",
             color="#c0392b")
    ax1.plot(steps, [l.off_manifold for l in steered], "s-", label="steered",
             color="#1f8a4c")
    ax1.axvline(bs, color="grey", ls="--", alpha=0.6)
    ax1.set(xlabel="step", ylabel="off-manifold drift",
            title="Illegal (off-manifold) belief drift")
    ax1.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"[plot] saved -> {path}")


# ---------------------------------------------------------------------------
# 4. Entry point
# ---------------------------------------------------------------------------

def run_demo(args: argparse.Namespace) -> dict:
    scene = build_scene(
        n_steps=args.steps, d_model=args.d_model, break_step=args.break_step,
        teleport_scale=args.teleport_scale, seed=args.seed,
    )
    # adaptive=False: on any detected break, fully project back onto the legal
    # manifold (a clean "snap back"). adaptive=True instead ramps the correction
    # with break depth -- useful in a live loop, less crisp for a one-shot demo.
    steerer = CausalActivationSteerer(strength=1.0, floor=args.floor, adaptive=False)
    base = rollout(scene, steerer=None, floor=args.floor)
    steered = rollout(scene, steerer=steerer, floor=args.floor)
    summary = print_report(base, steered, scene)
    if args.plot:
        maybe_plot(base, steered, scene, args.plot_path)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="L3 causal spatial steering demo")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--d-model", type=int, default=48, dest="d_model")
    p.add_argument("--break-step", type=int, default=18, dest="break_step")
    p.add_argument("--teleport-scale", type=float, default=6.0, dest="teleport_scale")
    p.add_argument("--floor", type=float, default=0.5,
                   help="consistency gate: steer fires when score < floor")
    p.add_argument("--plot", action="store_true", help="save a figure")
    p.add_argument("--plot-path", default="causal_spatial_steering.png", dest="plot_path")
    run_demo(p.parse_args())


if __name__ == "__main__":
    main()
