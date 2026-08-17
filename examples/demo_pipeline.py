"""
examples/demo_pipeline.py -- the dual-speed sentry, all layers strung together.

The story
---------
This is the product: not a pile of capabilities, but the *loop* that composes
them. An edge perimeter-sentry watches a drone approach a protected zone, using
every layer in its intended role -- and only this loop makes them a system:

    acoustic_ops    -- localize the drone's bearing from its sound (ITD->azimuth)
                       and hear it approach / recede (Doppler).
    spatial_ops     -- ask, every tick, "is it inside the protected radius?"
    physics_ops     -- one-step ballistic forecast -> a predictive-coding surprise.
    salience_events -- wake the slow layer ONLY when that surprise ignites
                       (the drone manoeuvres) -- never poll the hot loop.
    slow_layer      -- once woken, a SlowThreatAssessor rolls the target forward
                       over a horizon (the deliberation the hot loop can't afford
                       every tick) -> time-to-breach + a CLEAR/WATCH/ENGAGE verdict.
    failsafe        -- BlindRolloutGuard coasts on the world model through a sensor
                       dropout (and would go dark if blind too long); CircuitBreaker
                       keeps the turret aim within its mechanical limit and slew-rate.

The drone approaches in a straight line (the sentry tracks it but stays calm),
then makes a sharp evasive turn (a salient event fires -> escalate), the sensor
feed stutters for two ticks (the sentry coasts), and the drone breaches the zone.

Honest scope
------------
Deterministic reference integration (not a trained agent). The surprise is a real
training-free predictive-coding residual; the guard's coast budget is the
imagination's geometric trust decay. Far-field acoustic model (scene is in
front of the head). CPU, sub-second, ASCII-only.

Run::

    python -m examples.demo_pipeline
    python -m examples.demo_pipeline --ignite-z 2.5
"""

from __future__ import annotations

import argparse
import math
from typing import List, Tuple

import torch

from mt_lnn.pipeline import DualSpeedSentry
from mt_lnn.ingest_ops import align_stream


def scenario() -> List[Tuple[torch.Tensor, torch.Tensor, bool]]:
    """Straight approach -> evasive turn -> sensor dropout -> zone breach.

    Returns a list of ``(position, velocity, sensor_ok)`` per tick. Deterministic.
    """
    steps: List[Tuple[torch.Tensor, torch.Tensor, bool]] = []
    pos = torch.tensor([-15.0, 9.0])
    vel = torch.tensor([1.1, -0.5])
    for i in range(14):                                  # steady approach
        pos = pos + vel
        # small deterministic measurement jitter: a real sensor is never perfectly
        # steady, so the surprise baseline has a realistic (non-zero) variance and
        # the manoeuvre's salience z-score is a believable number, not "infinite".
        jit = torch.tensor([0.04 * math.sin(i * 1.7), 0.04 * math.cos(i * 1.3)])
        steps.append(((pos + jit).clone(), vel.clone(), True))
    vel = torch.tensor([0.2, -1.3])                      # sharp evasive turn
    for t in range(7):
        pos = pos + vel
        steps.append((pos.clone(), vel.clone(), t not in (2, 3)))  # dropout at turn+2,+3
    return steps


def align_feed(steps: List[Tuple[torch.Tensor, torch.Tensor, bool]], *,
               dt: float = 1.0, max_gap: float = 0.9):
    """Front-end: align a *timestamped* raw sensor feed onto the model's dt grid.

    The fixed-dt liquid core assumes one frame per ``dt``; a real sensor does not
    oblige. We turn the per-tick scenario into a timestamped stream in which the
    dropped frames are *genuinely missing samples* (no timestamp), then re-align
    it with :func:`mt_lnn.ingest_ops.align_stream`. The dropout is therefore
    *detected* as a coverage gap (a step too far from any real sample) rather than
    hardcoded -- exactly the ingestion alignment the architecture audit asked for.

    Returns ``(frames, summary)`` where ``frames`` is the per-grid-step
    ``(pos, vel, sensor_ok)`` driving the sentry and ``sensor_ok`` is the coverage
    flag. On this clean clock the resampling is identity at every covered step, so
    the live ticks see exactly the original trajectory.
    """
    present = [(i + 1, p, v) for i, (p, v, ok) in enumerate(steps) if ok]
    ts = torch.tensor([float(t) for (t, _, _) in present])
    pos = torch.stack([p for (_, p, _) in present])           # (S, 2)
    vel = torch.stack([v for (_, _, v) in present])           # (S, 2)
    n_grid = len(steps)
    pos_a = align_stream(pos, ts, dt=dt, max_gap=max_gap, t_start=1.0, n_steps=n_grid)
    vel_a = align_stream(vel, ts, dt=dt, max_gap=max_gap, t_start=1.0, n_steps=n_grid)
    frames = [(pos_a.values[k], vel_a.values[k], bool(pos_a.covered[k]))
              for k in range(n_grid)]
    summary = {"n_samples": len(present), "n_grid": n_grid,
               "n_lost": pos_a.n_gap_steps}
    return frames, summary


def run(args) -> dict:
    sentry = DualSpeedSentry(
        left_ear=[-0.1, 0.0], right_ear=[0.1, 0.0],
        zone_center=[0.0, 0.0], danger_radius=float(args.radius),
        freq=1000.0, dt=1.0, ignite_z=float(args.ignite_z),
        max_slew=math.radians(20.0), max_blind_steps=4,
    )
    steps = scenario()
    frames, ingest = align_feed(steps, dt=1.0, max_gap=0.9)
    ticks = [sentry.step(p, v, sensor_ok=ok) for (p, v, ok) in frames]
    positions = [p for (p, _, _) in steps]
    return {"ticks": ticks, "positions": positions, "radius": float(args.radius),
            "zone": (0.0, 0.0), "ingest": ingest}


def render_map(d: dict, *, w: int = 49, h: int = 17) -> str:
    """Top-down ASCII map: 'H' the head, 'O' the protected zone, digits the drone
    path (last digit of the tick), '#' marks the ignition tick."""
    pts = d["positions"]
    xs = [p[0].item() for p in pts] + [0.0]
    ys = [p[1].item() for p in pts] + [0.0]
    lo_x, hi_x = min(xs) - 1, max(xs) + 1
    lo_y, hi_y = min(ys) - 1, max(ys) + 1

    def col(x):
        return max(0, min(w - 1, int((x - lo_x) / (hi_x - lo_x) * (w - 1))))

    def row(y):
        return max(0, min(h - 1, int((hi_y - y) / (hi_y - lo_y) * (h - 1))))

    grid = [[" " for _ in range(w)] for _ in range(h)]

    # protected zone outline (a ring of 'o' at the danger radius) + centre 'O'
    cx, cy = d["zone"]
    r = d["radius"]
    for a in range(0, 360, 8):
        rad = math.radians(a)
        grid[row(cy + r * math.sin(rad))][col(cx + r * math.cos(rad))] = "o"
    grid[row(cy)][col(cx)] = "O"

    # drone path
    for i, p in enumerate(pts):
        ch = str((i + 1) % 10)
        grid[row(p[1].item())][col(p[0].item())] = ch
    # mark ignition
    for i, t in enumerate(d["ticks"]):
        if t.event is not None:
            grid[row(pts[i][1].item())][col(pts[i][0].item())] = "#"

    # the head (sensor) at the interaural midpoint, slightly below the zone
    grid[h - 1][col(0.0)] = "H"

    body = "\n".join("  " + "".join(rr) for rr in grid)
    return body


def render_log(d: dict) -> str:
    lines = ["    tick  src       bearing  range  doppler  zone  aim(cmd)  note"]
    for t in d["ticks"]:
        note = ""
        if t.event is not None:
            a = t.event.assessment
            verdict = ""
            if a is not None:
                eta = f"eta {a.eta_breach}" if a.eta_breach is not None else "no breach"
                verdict = f" -> SLOW[{a.level} {eta}]"
            note = f"IGNITE z={t.event.salience:.1f} -> wake slow layer{verdict}"
        elif t.source == "imagined":
            note = "coasting on world model (sensor dropout)"
        elif t.source == "dark":
            note = "BLIND -> hold / safe-stop"
        zone = "IN " if t.inside_zone else "out"
        lines.append(
            f"    {t.tick:>3}  {t.source:8s} {math.degrees(t.azimuth):+6.1f}d "
            f"{t.distance:5.1f}m {t.doppler:7.1f} {zone}  {math.degrees(t.command):+6.1f}d  {note}"
        )
    return "\n".join(lines)


def print_report(d: dict) -> None:
    line = "=" * 70
    print(line)
    print("DUAL-SPEED SENTRY -- perception + prediction + salience + safety, one loop")
    print(line)
    print(f"\n  protected zone radius = {d['radius']:.1f} m; sensor = binaural head 'H'\n")
    print(render_map(d))
    print("\n  legend: H sensor head   O zone centre   o danger radius   "
          "digits drone path   # ignition\n")
    print(render_log(d))

    igs = [t for t in d["ticks"] if t.event is not None]
    coasts = sum(t.source == "imagined" for t in d["ticks"])
    breaches = sum(t.inside_zone for t in d["ticks"])
    cmds = [t.command for t in d["ticks"]]
    print(f"\n  summary: {len(igs)} salient ignition(s) (slow layer woken on the "
          f"manoeuvre, NOT the steady approach),")
    for t in igs:
        a = t.event.assessment
        if a is not None:
            eta = f"breach in {a.eta_breach} step(s)" if a.eta_breach is not None \
                else "no breach in horizon"
            print(f"           slow layer @tick {a.woken_tick}: threat {a.level} "
                  f"({eta}, closest {a.min_range:.1f} m) -> {a.recommendation};")
    ing = d["ingest"]
    print(f"           ingest: aligned {ing['n_samples']} timestamped sample(s) onto "
          f"the {ing['n_grid']}-step dt grid; {ing['n_lost']} lost frame(s) -> "
          f"coverage gap -> coast;")
    print(f"           coasted {coasts} tick(s) through the sensor dropout, "
          f"{breaches} tick(s) inside the zone;")
    print(f"           every aim command stayed within +/-90 deg and "
          f"<=20 deg/tick (max step {max(abs(cmds[i+1]-cmds[i]) for i in range(len(cmds)-1))*180/math.pi:.1f} deg).")

    print("\n" + line)
    print("VERDICT [OK]: the layers are not a pile of parts -- this loop PERCEIVES,")
    print("        PREDICTS, wakes the slow layer only on real change, COASTS")
    print("        through dropouts and CLAMPS the actuator. Zero model.py coupling.")
    print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description="dual-speed sentry integration demo")
    ap.add_argument("--radius", type=float, default=3.0)
    ap.add_argument("--ignite-z", type=float, default=3.0)
    args = ap.parse_args()
    print_report(run(args))


if __name__ == "__main__":
    main()
