"""
examples/demo_acoustic_ops.py -- composable acoustic / binaural-hearing operators.

The story
---------
A brain does not store "the sound is on my left" -- it *computes* direction from
the physics of two ears. Two scenes show that computation, both emergent from raw
geometry via the zero-parameter operators in :mod:`mt_lnn.acoustic_ops`:

  Scene 1 -- A DRONE FLIES PAST.  A source crosses left-to-right in front of a
    stationary binaural head. From the interaural time difference alone the head
    *localizes* the bearing (it sweeps left -> ahead -> right), and from the
    Doppler shift it hears the pitch fall through the rest frequency at the moment
    of closest approach: "where is it, and is it coming or going?" -- computed.

  Scene 2 -- TWO SPEAKERS INTERFERE.  Two coherent speakers play the same tone; a
    microphone slides along a line between them. Wavefront superposition produces
    alternating loud (constructive) and quiet (destructive) bands -- the standing
    interference pattern, summed as phasors from the path lengths.

Honest scope
------------
Deterministic, analytic acoustics (no learned model). ITD->azimuth uses the
far-field plane-wave model (exact far from the head). CPU, sub-second, ASCII-only.

Run::

    python -m examples.demo_acoustic_ops
    python -m examples.demo_acoustic_ops --freq 2000
"""

from __future__ import annotations

import argparse
import math
from typing import List

import torch

from mt_lnn.acoustic_ops import (
    SPEED_OF_SOUND,
    binaural_scene,
    superpose_arrivals,
)

LEFT_EAR = torch.tensor([-0.1, 0.0])
RIGHT_EAR = torch.tensor([0.1, 0.0])


# --------------------------------------------------------------------------- #
# Scene 1 -- fly-by localization + Doppler                                     #
# --------------------------------------------------------------------------- #


def run_flyby(args) -> dict:
    T = args.ticks
    xs = torch.linspace(-12.0, 12.0, T)
    source = torch.stack([xs, torch.full_like(xs, float(args.standoff))], dim=1)
    vel = torch.zeros_like(source)
    vel[:, 0] = 1.0                                      # constant left-to-right motion
    sc = binaural_scene(source, LEFT_EAR, RIGHT_EAR,
                        source_vel=vel, freq=float(args.freq))
    return {"xs": xs, "scene": sc, "freq": float(args.freq)}


def _arrow(az_rad: float) -> str:
    """Map a bearing in radians to a coarse left/ahead/right glyph."""
    deg = math.degrees(az_rad)
    if deg < -45:
        return "<<"
    if deg < -10:
        return "< "
    if deg <= 10:
        return "^^"
    if deg <= 45:
        return " >"
    return ">>"


def render_flyby(d: dict) -> str:
    sc, xs, f0 = d["scene"], d["xs"], d["freq"]
    T = xs.shape[0]
    az = sc.azimuth.tolist()
    dop = sc.doppler.tolist()

    bearings = " ".join(_arrow(a) for a in az)
    # Doppler as an up/down deviation glyph around the rest frequency.
    dop_row = "".join(_pitch_glyph(p, f0) for p in dop)
    head = (f"    source x : {xs[0]:+.0f} m ........ {xs[T // 2]:+.0f} m (closest) "
            f"........ {xs[-1]:+.0f} m\n"
            f"    bearing  : {bearings}\n"
            f"    pitch    : {dop_row}   (rest tone {f0:.0f} Hz; "
            "' = above/approaching, . = below/receding)")
    return head


def _pitch_glyph(p: float, f0: float) -> str:
    rel = (p - f0) / f0
    if rel > 1e-3:
        return "'"
    if rel < -1e-3:
        return "."
    return "-"


# --------------------------------------------------------------------------- #
# Scene 2 -- two-speaker interference                                          #
# --------------------------------------------------------------------------- #


def run_interference(args) -> dict:
    f = float(args.freq)
    wavelength = SPEED_OF_SOUND / f
    speakers = torch.stack([torch.tensor([-2.0, 0.0]),
                            torch.tensor([2.0, 0.0])])    # (2, 2)
    n = 61
    mic_x = torch.linspace(-2.0, 2.0, n)
    mics = torch.stack([mic_x, torch.full_like(mic_x, 3.0)], dim=1)   # along a line
    # batch the receiver axis: superpose for each mic position
    amps = []
    for i in range(n):
        p = superpose_arrivals(speakers, mics[i], freq=f)
        amps.append(p.abs().item())
    return {"mic_x": mic_x, "amp": amps, "wavelength": wavelength, "freq": f}


def render_interference(d: dict) -> str:
    amp = d["amp"]
    hi = max(amp) or 1.0
    height = 7
    n = len(amp)
    rows = [[" " for _ in range(n)] for _ in range(height)]
    for c, a in enumerate(amp):
        h = int(round(a / hi * (height - 1)))
        for r in range(h + 1):
            rows[height - 1 - r][c] = "|"
    body = "\n".join("    " + "".join(r) for r in rows)
    axis = "    " + "-" * n
    label = (f"    mic slides from x=-2 to x=+2 (3 m in front); "
             f"wavelength={d['wavelength']:.2f} m\n"
             "    loud bands = constructive, gaps = destructive interference")
    return body + "\n" + axis + "\n" + label


# --------------------------------------------------------------------------- #
# report                                                                       #
# --------------------------------------------------------------------------- #


def print_report(fd: dict, idata: dict) -> None:
    line = "=" * 66
    print(line)
    print("BINAURAL HEARING -- localize and hear motion from the geometry of sound")
    print(line)

    print("\nSCENE 1  a drone flies past a binaural head (left -> right)")
    print("  cues: ITD -> azimuth (Jeffress readout)  +  Doppler -> approaching/receding\n")
    print(render_flyby(fd))
    sc = fd["scene"]
    import math as _m
    print(f"\n  localized bearing swept {_m.degrees(sc.azimuth[0]):+.0f} deg -> "
          f"{_m.degrees(sc.azimuth[-1]):+.0f} deg through ~0 at closest approach;")
    print(f"  pitch fell {sc.doppler[0]:.0f} Hz -> {sc.doppler[-1]:.0f} Hz "
          f"through the {fd['freq']:.0f} Hz rest tone.")

    print("\n" + "-" * 66)
    print("\nSCENE 2  two coherent speakers, a sliding microphone")
    print("  cue: wavefront superposition (phasor sum over path lengths)\n")
    print(render_interference(idata))

    print("\n" + line)
    print("VERDICT [OK]: direction, motion and interference are all COMPUTED from")
    print("        time-of-flight geometry by zero-parameter operators -- the")
    print("        auditory-space counterpart to 'compute, don't memorise'.")
    print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description="composable acoustic / binaural operators demo")
    ap.add_argument("--ticks", type=int, default=25)
    ap.add_argument("--standoff", type=float, default=2.0, help="source distance in front (m)")
    ap.add_argument("--freq", type=float, default=1000.0, help="source tone (Hz)")
    args = ap.parse_args()
    print_report(run_flyby(args), run_interference(args))


if __name__ == "__main__":
    main()
