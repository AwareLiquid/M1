"""
examples/demo_stdp_ops.py -- composable, backprop-free STDP on discrete event
streams (P0 learning): a synapse that learns purely from spike *timing*.

The story
---------
``plasticity.py`` runs Hebbian consolidation at the loss level for the smooth,
continuous-time LTC core -- and explains why biological STDP does NOT fit there
(no discrete spikes to time). But the salience layer DOES emit genuinely discrete,
timestamped ignition events, and that is exactly where spike-timing learning is
the right tool. This demo drives the STDP operator with two event streams and
shows a synapse strengthen or weaken with NO loss, NO gradient -- the learning
signal is the timing alone.

We build three scenarios on a shared spike grid:

    (1) causal      -- the pre-synaptic event consistently fires BEFORE the post
                       one (pre helped cause post). Net change is POTENTIATION,
                       and the depression lobe is exactly zero.
    (2) acausal     -- the roles are swapped (post leads). Net change is
                       DEPRESSION, and the potentiation lobe is exactly zero.
    (3) coincident  -- a single simultaneous event pair (dt = 0): no causal
                       order, so the weight does not move at all.

For the causal stream we also cross-check the two computations the module
provides -- the online eligibility-trace update (how STDP runs on hardware,
O(T)) and the all-to-all pairwise window sum (the definition) -- and confirm
they agree to machine precision.

Honest scope
------------
Pure timing operator, no model, zero trainable parameters. The spike trains here
are synthetic {0,1} grids standing in for salience ignitions / L2 place-code
writes; wiring this to gate real event streams is a separate integration step.
Deterministic, CPU, sub-second, ASCII-only.

Run::

    python examples/demo_stdp_ops.py
    python examples/demo_stdp_ops.py --tau 25
"""

from __future__ import annotations

import argparse

import torch

from mt_lnn.stdp_ops import (
    STDPParams,
    pairwise_stdp,
    stdp_trace_update,
    window_integral,
)

_f64 = torch.float64


def _spikes(T: int, idx) -> torch.Tensor:
    """A length-T {0,1} spike train with spikes at the given indices."""
    s = torch.zeros(T, dtype=_f64)
    s[list(idx)] = 1.0
    return s


def _ascii_train(label: str, train: torch.Tensor) -> str:
    """Render a {0,1} train as a tick raster, e.g. ``pre  | . | # | . |``."""
    cells = "".join("#" if float(v) > 0 else "." for v in train)
    return f"   {label:5s} |{cells}|"


def _ascii_window(p: STDPParams, span: int = 20) -> str:
    """A tiny ASCII profile of the STDP window W(dt) over dt in [-span, span]."""
    grid = torch.arange(-span, span + 1, dtype=_f64)
    from mt_lnn.stdp_ops import stdp_kernel
    w = stdp_kernel(grid, p)
    peak = float(w.abs().max()) or 1.0
    rows = []
    width = 24
    for dt, val in zip(grid.tolist(), w.tolist()):
        n = int(round(width * abs(val) / peak))
        if val > 0:
            bar = " " * width + "|" + "+" * n
        elif val < 0:
            bar = " " * (width - n) + "-" * n + "|"
        else:
            bar = " " * width + "|"
        rows.append(f"   dt={int(dt):+3d} {bar}")
    return "\n".join(rows)


def run_demo(args) -> dict:
    p = STDPParams(a_plus=1.0, a_minus=1.0, tau_plus=args.tau, tau_minus=args.tau, dt=1.0)
    T = 12

    # (1) causal: pre leads post -> potentiation, no depression
    pre_c = _spikes(T, [1, 2, 3])
    post_c = _spikes(T, [8, 9, 10])
    causal = stdp_trace_update(pre_c, post_c, p)

    # (2) acausal: post leads pre -> depression, no potentiation (roles swapped)
    acausal = stdp_trace_update(post_c, pre_c, p)

    # (3) coincident single pair: dt = 0 -> no change
    pre_s = _spikes(T, [5])
    post_s = _spikes(T, [5])
    coincident = stdp_trace_update(pre_s, post_s, p)

    # cross-check the trace form against the all-to-all pairwise definition
    grid = torch.arange(T, dtype=_f64)
    pw = pairwise_stdp(grid[pre_c > 0], grid[post_c > 0], p)

    ltp_area, ltd_area = window_integral(p)

    return {
        "params": p,
        "T": T,
        "pre_c": pre_c, "post_c": post_c,
        "causal": causal, "acausal": acausal, "coincident": coincident,
        "dw_causal": float(causal.delta_w),
        "dep_causal": float(causal.depression),
        "dw_acausal": float(acausal.delta_w),
        "pot_acausal": float(acausal.potentiation),
        "dw_coincident": float(coincident.delta_w),
        "pairwise_causal": float(pw),
        "trace_vs_pairwise_gap": abs(float(causal.delta_w) - float(pw)),
        "window_areas": (ltp_area, ltd_area),
    }


def print_report(s: dict) -> None:
    line = "=" * 64
    p = s["params"]
    print(line)
    print("SPIKE-TIMING-DEPENDENT PLASTICITY -- learning from timing, no backprop")
    print(line)

    print(f"\n  window: A+/A- = {p.a_plus:.1f}/{p.a_minus:.1f}, "
          f"tau+/tau- = {p.tau_plus:.1f}/{p.tau_minus:.1f}, dt = {p.dt:.1f}")
    print(f"  balanced lobe areas (A*tau) = "
          f"{s['window_areas'][0]:.2f} / {s['window_areas'][1]:.2f}\n")
    print("  W(dt): + = potentiation (pre before post), - = depression:")
    print(_ascii_window(p))

    print("\n  -- (1) causal stream: pre fires BEFORE post --")
    print(_ascii_train("pre", s["pre_c"]))
    print(_ascii_train("post", s["post_c"]))
    print(f"  delta_w = {s['dw_causal']:+.4f}   (POTENTIATION; "
          f"depression lobe = {s['dep_causal']:.1e})")

    print("\n  -- (2) acausal stream: roles swapped, post leads --")
    print(_ascii_train("pre", s["post_c"]))
    print(_ascii_train("post", s["pre_c"]))
    print(f"  delta_w = {s['dw_acausal']:+.4f}   (DEPRESSION; "
          f"potentiation lobe = {s['pot_acausal']:.1e})")

    print("\n  -- (3) coincident pair: a single dt=0 event --")
    print(f"  delta_w = {s['dw_coincident']:+.1e}   (no causal order -> no change)")

    print("\n  -- trace form == all-to-all pairwise definition --")
    print(f"  trace delta_w    = {s['dw_causal']:+.6f}")
    print(f"  pairwise delta_w = {s['pairwise_causal']:+.6f}")
    print(f"  |gap|            = {s['trace_vs_pairwise_gap']:.2e}")

    ok = (
        s["dw_causal"] > 0.0
        and s["dep_causal"] < 1e-9
        and s["dw_acausal"] < 0.0
        and s["pot_acausal"] < 1e-9
        and abs(s["dw_coincident"]) < 1e-9
        and s["trace_vs_pairwise_gap"] < 1e-9
    )
    print("\n" + line)
    if ok:
        print("VERDICT [OK]: timing ALONE moves the weight -- causal order")
        print("        potentiates, acausal depresses, coincidence does nothing,")
        print("        and the O(T) trace form equals the all-pairs definition.")
        print("        Local, backprop-free learning for discrete event streams.")
    else:
        print("VERDICT [FAIL]: STDP did not behave as expected -- see above.")
    print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description="composable backprop-free STDP demo")
    ap.add_argument("--tau", type=float, default=15.0,
                    help="time constant for both window lobes (default 15.0).")
    args = ap.parse_args()
    print_report(run_demo(args))


if __name__ == "__main__":
    main()
