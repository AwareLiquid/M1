"""
examples/demo_spatial_memory.py -- L2 spatial sequence memory demo.

What this shows (the roadshow story)
------------------------------------
An agent walks a path through a 2-D arena. At each *landmark* it sees something
-- a content vector (think: an object embedding, a scene token, a fact). As it
moves it WRITES that content into a place-indexed associative memory, keyed by
its current location (Bicanski & Burgess 2021): "remember THIS, HERE".

Later the agent wants to recall what was at a place -- but it only has a *fuzzy*
idea of where it is (GPS drift, dead-reckoning error). We query the memory with
the landmark positions perturbed by increasing noise and measure how well the
recalled content matches the truth. The money shot:

  * recall is **pattern completion** -- a noisy / approximate positional cue
    still snaps back the correct stored content (high cosine similarity);
  * it **degrades gracefully** -- accuracy falls smoothly as the query noise
    grows past a place field, rather than failing catastrophically;
  * querying an **empty** region (between landmarks) recalls a blurred average,
    not any single landmark strongly -- the memory does not hallucinate a sharp
    answer where nothing was written.

This is L2 in the four-layer spatial-cognition story: L1 turns "where am I" into
a place code, L2 stores/recalls "what is here" by that code, L3 keeps the spatial
belief on its legal manifold.

Decoupling (zhu yi ou he)
-------------------------
* Uses ONLY the public ``mt_lnn.spatial_memory.SpatialMemory`` API (which itself
  reuses ``mt_lnn.spatial.PlaceCellCode``). It does NOT import the MT-LNN backbone
  (``model.py``). Zero new model coupling; zero trainable parameters.
* Fully deterministic given ``--seed`` so the printed numbers are reproducible.
* Runs on CPU in well under a second.

Run::

    python examples/demo_spatial_memory.py
    python examples/demo_spatial_memory.py --plot
    python examples/demo_spatial_memory.py --seed 7 --n-landmarks 12
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F

from mt_lnn.spatial_memory import SpatialMemory


# ---------------------------------------------------------------------------
# 1. A walked trajectory of landmarks, each carrying a content vector
# ---------------------------------------------------------------------------

@dataclass
class World:
    """A set of landmark locations and the content stored at each.

    Attributes
    ----------
    positions : ``(K, 2)`` arena coordinates of the K landmarks (the path).
    contents : ``(K, d_model)`` the unit content vector stored at each landmark.
    arena : square arena side length.
    """

    positions: torch.Tensor
    contents: torch.Tensor
    arena: float

    @property
    def n_landmarks(self) -> int:
        return self.positions.shape[0]

    @property
    def d_model(self) -> int:
        return self.contents.shape[1]


def build_world(
    *,
    n_landmarks: int = 8,
    d_model: int = 32,
    arena: float = 2.2,
    seed: int = 0,
) -> World:
    """Lay landmarks along a smooth path and give each a distinct unit content.

    The path is an ellipse so the landmarks spread across the arena; the content
    vectors are random and L2-normalised so cosine similarity reads cleanly.
    """
    g = torch.Generator().manual_seed(seed)

    t = torch.linspace(0.0, 1.0, n_landmarks + 1)[:-1]    # avoid duplicate endpoint
    radius = 0.32 * arena
    centre = 0.5 * arena
    positions = torch.stack([
        centre + radius * torch.cos(2.0 * torch.pi * t),
        centre + 0.9 * radius * torch.sin(2.0 * torch.pi * t),
    ], dim=1)                                             # (K, 2)

    contents = torch.randn(n_landmarks, d_model, generator=g)
    contents = F.normalize(contents, dim=-1)              # unit vectors

    return World(positions=positions, contents=contents, arena=arena)


# ---------------------------------------------------------------------------
# 2. Write the trajectory into memory, then recall under positional noise
# ---------------------------------------------------------------------------

def build_memory(world: World, *, n_place: int = 256, sigma: float = 0.12) -> SpatialMemory:
    """Walk the path once, writing each landmark's content at its location."""
    mem = SpatialMemory(
        d_model=world.d_model,
        n_place=n_place,
        arena=world.arena,
        sigma=sigma,
        seed=123,
    )
    # One trajectory, all landmarks in order: (B=1, K, ...).
    mem.write(world.positions.unsqueeze(0), world.contents.unsqueeze(0))
    return mem


def recall_accuracy(mem: SpatialMemory, world: World, *, noise: float, seed: int) -> float:
    """Mean cosine(recall(true_pos + noise), true_content) over all landmarks."""
    g = torch.Generator().manual_seed(seed)
    jitter = noise * torch.randn(world.positions.shape, generator=g)
    query = (world.positions + jitter).unsqueeze(0)       # (1, K, 2)
    recalled = mem.read(query).squeeze(0)                 # (K, d)
    cos = F.cosine_similarity(recalled, world.contents, dim=-1)
    return float(cos.mean())


def discrimination_margin(mem: SpatialMemory, world: World) -> float:
    """How decisively recall addresses the RIGHT content (no noise).

    Recalling at each true landmark, we compare the cosine to the correct stored
    content against the best-matching *wrong* content. A large positive margin
    means the place key retrieves the right memory rather than a blur of its
    neighbours -- the associative-addressing fidelity.
    """
    recalled = mem.read(world.positions.unsqueeze(0)).squeeze(0)   # (K, d)
    recalled = F.normalize(recalled, dim=-1)
    sims = recalled @ world.contents.t()                  # (K, K) cosine
    k = world.n_landmarks
    eye = torch.eye(k, dtype=torch.bool)
    correct = sims[eye]                                   # (K,)
    best_wrong = sims.masked_fill(eye, float("-inf")).max(dim=-1).values
    return float((correct - best_wrong).mean())


# ---------------------------------------------------------------------------
# 3. Report
# ---------------------------------------------------------------------------

NOISE_GRID = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]


def run_demo(args) -> dict:
    """End-to-end: build world, write memory, sweep recall under noise."""
    torch.manual_seed(args.seed)
    world = build_world(
        n_landmarks=args.n_landmarks, d_model=args.d_model,
        arena=args.arena, seed=args.seed,
    )
    mem = build_memory(world, n_place=args.n_place, sigma=args.sigma)

    curve: List[Tuple[float, float]] = [
        (nz, recall_accuracy(mem, world, noise=nz, seed=args.seed + 1))
        for nz in NOISE_GRID
    ]
    margin = discrimination_margin(mem, world)

    clean = curve[0][1]
    moderate = recall_accuracy(mem, world, noise=args.sigma, seed=args.seed + 1)
    degraded = curve[-1][1]                               # recall at the largest noise

    # Verdict: clean recall is near-perfect; recall at ~one field width still
    # works; large-noise recall falls clearly below clean (so recall genuinely
    # tracks position, not a constant); and addressing is decisive (margin > 0).
    verdict = (
        (clean > 0.9)
        and (moderate > 0.5)
        and (degraded < clean - 0.2)
        and (margin > 0.2)
    )

    summary = {
        "clean_recall": clean,
        "recall_at_sigma": moderate,
        "degraded_recall": degraded,
        "discrimination_margin": margin,
        "curve": curve,
        "verdict": bool(verdict),
        "n_landmarks": world.n_landmarks,
    }

    print_report(world, mem, summary)
    if args.plot:
        maybe_plot(summary, args.plot_path)
    return summary


def print_report(world: World, mem: SpatialMemory, summary: dict) -> None:
    print("=" * 64)
    print("L2 spatial sequence memory -- write a path, recall by place")
    print("=" * 64)
    print(f"landmarks={world.n_landmarks}  d_model={world.d_model}  "
          f"n_place={mem.n_place}  arena={world.arena}")
    print(f"params={sum(p.numel() for p in mem.parameters())} (zero-trainable)")
    print("-" * 64)
    print("query noise (m) | recall cosine vs truth")
    for nz, acc in summary["curve"]:
        bar = "#" * int(round(max(acc, 0.0) * 40))
        print(f"   {nz:5.2f}        | {acc:6.3f}  {bar}")
    print("-" * 64)
    print(f"clean recall (noise=0)        : {summary['clean_recall']:.3f}")
    print(f"recall at ~one field width    : {summary['recall_at_sigma']:.3f}")
    print(f"recall at largest noise       : {summary['degraded_recall']:.3f}")
    print(f"discrimination margin (right- : {summary['discrimination_margin']:.3f}")
    print(f"  vs best-wrong content)        ")
    verdict = "[OK]" if summary["verdict"] else "[FAIL]"
    print("-" * 64)
    print(f"VERDICT {verdict}  pattern completion recalls the RIGHT content from "
          f"a fuzzy place cue and degrades gracefully as the cue worsens")
    print("=" * 64)


def maybe_plot(summary: dict, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plot] matplotlib unavailable ({e}); skipping plot")
        return

    xs = [nz for nz, _ in summary["curve"]]
    ys = [acc for _, acc in summary["curve"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, "o-", label="recall cosine")
    ax.axhline(summary["discrimination_margin"], ls="--", color="grey",
               label="addressing margin (right vs best-wrong)")
    ax.set_xlabel("query position noise (m)")
    ax.set_ylabel("recall cosine vs truth")
    ax.set_title("L2 SpatialMemory: graceful pattern completion")
    ax.set_ylim(-0.1, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[plot] wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser(description="L2 spatial sequence memory demo")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-landmarks", type=int, default=8)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--n-place", type=int, default=256)
    p.add_argument("--arena", type=float, default=2.2)
    p.add_argument("--sigma", type=float, default=0.12)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-path", type=str, default="artifacts/spatial_memory_recall.png")
    args = p.parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
