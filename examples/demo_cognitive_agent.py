"""
examples/demo_cognitive_agent.py -- the whole brain, one loop: an autonomous
agent perceives, remembers, is steered by a goal, imagines + physically validates
a plan, survives a sensor blackout, and clamps its command to a safe envelope.

The story
---------
This is the *system*, not a pile of capabilities. A small agent is dropped into a
2-D scene cluttered with obstacles and asked to reach a goal point. Every layer of
the MT-LNN stack plays its intended biological role, and ONLY their composition
makes an agent:

    PERCEIVE  (L1 grid cells)   -- mt_lnn.spatial.GridCellEncoding turns every
                                   object's (x, y) into an entorhinal-style
                                   multi-scale place code; distinct objects get
                                   near-orthogonal codes (a metric space sense).
    REMEMBER  (L2 SpatialMemory)-- mt_lnn.spatial_memory.SpatialMemory writes
                                   each object's content at its location, forming
                                   a place-indexed cognitive map; a fuzzy positional
                                   cue later recalls the right content (pattern
                                   completion, Bicanski & Burgess 2021).
    ATTEND    (top-down goal)   -- the goal vector is injected through the model's
                                   top_down pathway (P1 closed-loop step 2). With
                                   the per-block gate CLOSED a goal is a strict
                                   no-op; OPEN, the same scene is steered toward
                                   goal-relevant processing (measurable hidden shift).
    IMAGINE   (L4 + physics)    -- mt_lnn.imagination.LatentImagination rolls the
                                   world model forward in latent space (the mental
                                   simulation, with decaying confidence), while
                                   mt_lnn.physics_ops validates each *candidate 2-D
                                   path* for collisions (overlapping_pairs) and
                                   kinematic feasibility -- the plan must obey the
                                   laws of motion AND not hit anything.
    SURVIVE   (BlindRolloutGuard)- the perception feed is cut mid-mission; the
                                   guard coasts on the internal world model while
                                   its confidence holds, then goes DARK ("lost
                                   perception, stop") rather than hallucinate.
    ACT-SAFE  (CircuitBreaker)  -- the chosen path's steering command is hard-
                                   clamped to the actuator's physical red-lines, no
                                   matter what the planner emits.

Honest scope
------------
A deterministic *reference integration*, not a trained agent. The backbone and
spatial MLP are randomly initialised (seeded), so -- exactly as in the imagination
and pipeline demos -- the latent imagination proves the *machinery* (zero added
params, confidence decay), while the genuine quantitative discriminator for the
plan is the training-free physics check (collisions + kinematics are exact). The
top-down no-op/steer property is the real, tested behaviour of model.py. Every
actuator (guard, breaker) has ZERO trainable params and never imports model.py.
CPU, sub-second, ASCII-only, reproducible given --seed.

Run::

    python -m examples.demo_cognitive_agent
    python -m examples.demo_cognitive_agent --gate 1.5 --seed 3
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel
from mt_lnn.spatial import GridCellEncoding, SpatialCoordEncoder
from mt_lnn.spatial_memory import SpatialMemory
from mt_lnn.world_model import PredictiveStateHead  # noqa: F401  (head comes via model)
from mt_lnn.imagination import LatentImagination
from mt_lnn.failsafe import BlindRolloutGuard, CircuitBreaker
from mt_lnn import physics_ops

_D = 104           # d_model; divisible by n_heads=13 (protofilament count)
_ARENA = 1.0       # unit arena -> grid-cell codes live in their natural [0,1] range


# ---------------------------------------------------------------------------
# The scene: a start, a goal, and obstacles to avoid
# ---------------------------------------------------------------------------

@dataclass
class Scene:
    """A 2-D navigation problem in the unit arena.

    Attributes
    ----------
    start, goal : ``(2,)`` agent start and target positions.
    obstacles : ``(K, 2)`` obstacle centres.
    obstacle_radius : obstacle sphere radius (collision threshold).
    agent_radius : the agent's own collision radius.
    """

    start: torch.Tensor
    goal: torch.Tensor
    obstacles: torch.Tensor
    obstacle_radius: float
    agent_radius: float

    @property
    def objects(self) -> torch.Tensor:
        """All perceivable objects: obstacles then the goal -> (K+1, 2)."""
        return torch.cat([self.obstacles, self.goal.unsqueeze(0)], dim=0)


def build_scene() -> Scene:
    """A fixed scene whose straight-line solution is blocked (so planning matters)."""
    return Scene(
        start=torch.tensor([0.10, 0.10]),
        goal=torch.tensor([0.90, 0.90]),
        # a diagonal wall of obstacles straddling the straight start->goal line
        obstacles=torch.tensor([
            [0.50, 0.50],
            [0.42, 0.58],
            [0.58, 0.42],
            [0.35, 0.35],
        ]),
        obstacle_radius=0.09,
        agent_radius=0.03,
    )


# ---------------------------------------------------------------------------
# Stage 1 -- PERCEIVE: grid-cell place codes for every object
# ---------------------------------------------------------------------------

def perceive(scene: Scene) -> dict:
    """Encode all object positions with the entorhinal grid-cell population code.

    A good metric code makes *distinct* locations near-orthogonal: we report the
    mean off-diagonal cosine between object codes (want it well below 1 -- the
    code separates places) and confirm a place is identical to itself.
    """
    grid = GridCellEncoding(coord_dim=2, n_scales=6)          # 0 trainable params
    codes = grid(scene.objects.unsqueeze(0)).squeeze(0)        # (K+1, out_dim)
    codes_n = F.normalize(codes, dim=-1)
    sim = codes_n @ codes_n.t()                                # cosine matrix
    k = codes.shape[0]
    off = sim.masked_fill(torch.eye(k, dtype=torch.bool), 0.0)
    return {
        "code_dim": grid.out_dim,
        "n_objects": k,
        "mean_offdiag_cos": float(off.abs().sum() / (k * (k - 1))),
        "self_cos": float(sim.diagonal().mean()),
    }


# ---------------------------------------------------------------------------
# Stage 2 -- REMEMBER: write the scene into a place-indexed cognitive map
# ---------------------------------------------------------------------------

def remember(scene: Scene, *, seed: int) -> dict:
    """Write (position, content) for every object, then recall under positional noise.

    Each object carries a distinct unit "content" vector (its identity). Recall
    from a fuzzy positional cue should snap back the correct content (pattern
    completion) -- the agent has built an internal map of "what is where".
    """
    g = torch.Generator().manual_seed(seed)
    objs = scene.objects                                        # (K+1, 2)
    n = objs.shape[0]
    content = F.normalize(torch.randn(n, _D, generator=g), dim=-1)

    # Sharp place fields (sigma < object spacing) so the clustered central
    # obstacles are resolved as distinct memories rather than blurred together.
    mem = SpatialMemory(d_model=_D, n_place=256, arena=_ARENA, sigma=0.04, seed=123)
    mem.write(objs.unsqueeze(0), content.unsqueeze(0))

    def recall_cos(noise: float) -> float:
        jit = noise * torch.randn(objs.shape, generator=g)
        rec = mem.read((objs + jit).unsqueeze(0)).squeeze(0)
        return float(F.cosine_similarity(rec, content, dim=-1).mean())

    return {
        "n_place": mem.n_place,
        "clean_recall": recall_cos(0.0),
        "fuzzy_recall": recall_cos(0.05),
        "mem_params": sum(p.numel() for p in mem.parameters()),
    }


# ---------------------------------------------------------------------------
# Stage 3 -- ATTEND: a top-down goal steers the backbone (closed = no-op)
# ---------------------------------------------------------------------------

def _agent_model(*, seed: int) -> MTLNNModel:
    torch.manual_seed(seed)
    cfg = MTLNNConfig(
        vocab_size=64, d_model=_D, n_layers=2, n_heads=13, n_kv_heads=1,
        d_head=8, max_seq_len=64, gwtb_n_heads=1, dropout=0.0,
        attention_dropout=0.0, use_top_down=True, use_world_model=True,
    )
    return MTLNNModel(cfg).eval()


def _final_norm_hidden(model: MTLNNModel, *, inputs_embeds: torch.Tensor,
                       top_down: Optional[torch.Tensor]) -> torch.Tensor:
    """Run a forward on spatial tokens and tap the post-final_norm hidden state.

    The demo owns this hook; the model needs no modification. Returns (B, T, d).
    """
    captured: dict = {}

    def hook(_m, _i, out):
        captured["h"] = out.detach()

    handle = model.final_norm.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(inputs_embeds=inputs_embeds, top_down=top_down, use_cache=False)
    finally:
        handle.remove()
    return captured["h"]


def attend(scene: Scene, *, seed: int, gate: float) -> Tuple[dict, MTLNNModel, torch.Tensor]:
    """Inject the goal through the top-down pathway and measure that it steers.

    The scene's object positions become backbone tokens (inputs_embeds); the goal
    position becomes the top_down goal vector. With the per-block gate CLOSED a
    goal is bit-identical to no goal (strict no-op); OPEN, it shifts the hidden
    state -- the agent now attends to goal-relevant processing. Returns the
    goal-modulated hidden for the imagination stage.
    """
    model = _agent_model(seed=seed)
    enc = SpatialCoordEncoder(d_model=_D, coord_dim=2)         # grid-code -> tokens
    enc.eval()
    with torch.no_grad():
        tokens = enc(scene.objects.unsqueeze(0))               # (1, K+1, d)
        goal_vec = enc(scene.goal.view(1, 1, 2)).squeeze(1)    # (1, d)

    # 1. gate CLOSED (init): a goal must change nothing.
    none = _final_norm_hidden(model, inputs_embeds=tokens, top_down=None)
    closed = _final_norm_hidden(model, inputs_embeds=tokens, top_down=goal_vec)
    closed_diff = (none - closed).abs().max().item()

    # 2. open every block's gate, then the goal steers the representation.
    for blk in model.blocks:
        blk.top_down_gate.data.fill_(float(gate))
    steered = _final_norm_hidden(model, inputs_embeds=tokens, top_down=goal_vec)
    open_diff = (steered - none).abs().max().item()

    info = {
        "gate": float(gate),
        "closed_diff": closed_diff,           # want exactly 0.0 (strict no-op)
        "open_diff": open_diff,               # want > 0 (the goal steers)
        "hidden_finite": bool(torch.isfinite(steered).all()),
    }
    return info, model, steered


# ---------------------------------------------------------------------------
# Stage 4 -- IMAGINE: latent rollout + physics validation of candidate paths
# ---------------------------------------------------------------------------

def _waypoint_path(start: torch.Tensor, goal: torch.Tensor,
                   via: Optional[torch.Tensor], steps: int) -> torch.Tensor:
    """A polyline start -> (via) -> goal sampled at ``steps`` points -> (steps, 2)."""
    knots = [start] + ([via] if via is not None else []) + [goal]
    segs = len(knots) - 1
    per = max(1, steps // segs)
    pts: List[torch.Tensor] = []
    for s in range(segs):
        a, b = knots[s], knots[s + 1]
        ts = torch.linspace(0.0, 1.0, per + 1)[: -1 if s < segs - 1 else None]
        pts.extend(a + (b - a) * t for t in ts)
    return torch.stack(pts, dim=0)


def _validate_path(path: torch.Tensor, scene: Scene, *, dt: float,
                   v_max: float, a_max: float) -> dict:
    """Physics check: does this 2-D path collide, and does it obey motion limits?

    Collisions use the real ``physics_ops.overlapping_pairs`` (the agent sphere
    against every obstacle sphere at every step). Kinematic feasibility uses
    finite-difference velocity/acceleration against the actuator's limits -- the
    plan must be physically drivable, not just geometrically clear.
    """
    radius = torch.cat([
        torch.tensor([scene.agent_radius]),
        torch.full((scene.obstacles.shape[0],), scene.obstacle_radius),
    ])
    collisions = 0
    for p in path:
        bodies = torch.cat([p.unsqueeze(0), scene.obstacles], dim=0)   # (1+K, 2)
        over = physics_ops.overlapping_pairs(bodies, radius)           # (1+K, 1+K)
        if bool(over[0].any()):                                        # agent vs any
            collisions += 1

    vel = (path[1:] - path[:-1]) / dt
    acc = (vel[1:] - vel[:-1]) / dt
    max_speed = float(vel.norm(dim=-1).max()) if vel.numel() else 0.0
    max_accel = float(acc.norm(dim=-1).max()) if acc.numel() else 0.0
    length = float((path[1:] - path[:-1]).norm(dim=-1).sum())

    return {
        "collisions": collisions,
        "max_speed": max_speed,
        "max_accel": max_accel,
        "length": length,
        "collision_free": collisions == 0,
        "kinematically_ok": max_speed <= v_max and max_accel <= a_max,
    }


def imagine_and_plan(scene: Scene, model: MTLNNModel, hidden: torch.Tensor, *,
                     horizon: int, dt: float, v_max: float, a_max: float) -> dict:
    """L4 latent mental simulation + physics-validated path selection.

    Two complementary checks:
      * the world model imagines a latent trajectory off the goal-modulated hidden
        (confidence decays with the horizon -- the mental-simulation machinery);
      * three candidate 2-D paths (straight + two detours) are each physics-checked;
        the plan we COMMIT to is the cheapest path that is both collision-free and
        kinematically feasible.
    """
    head = model.world_model_head
    assert head is not None, "world model must be enabled"
    imag = LatentImagination(head, horizon=horizon, trust_decay=0.85)
    traj = imag.imagine(hidden if hidden.dim() == 3 else hidden.unsqueeze(0))

    candidates = {
        "straight": None,
        "detour-top-left": torch.tensor([0.15, 0.85]),
        "detour-bot-right": torch.tensor([0.85, 0.15]),
    }
    steps = max(horizon * 4, 16)
    reports = {}
    for name, via in candidates.items():
        path = _waypoint_path(scene.start, scene.goal, via, steps)
        reports[name] = (_validate_path(path, scene, dt=dt, v_max=v_max, a_max=a_max),
                         path)

    feasible = {n: r for n, (r, _) in reports.items()
                if r["collision_free"] and r["kinematically_ok"]}
    chosen = min(feasible, key=lambda n: reports[n][0]["length"]) if feasible else None

    return {
        "added_params": imag.n_parameters,
        "confidence": traj.confidence[0].tolist(),
        "conf_decays": bool(traj.confidence[0, -1] < traj.confidence[0, 0]),
        "candidates": {n: r for n, (r, _) in reports.items()},
        "chosen": chosen,
        "chosen_path": reports[chosen][1] if chosen else None,
        "straight_blocked": not reports["straight"][0]["collision_free"],
    }


# ---------------------------------------------------------------------------
# Stage 5 -- SURVIVE: cut the feed, coast on the world model, then go dark
# ---------------------------------------------------------------------------

def survive(model: MTLNNModel, hidden: torch.Tensor, *, horizon: int,
            floor: float, max_blind: int) -> dict:
    """Feed a few live ticks, then blackout: the guard coasts then stops.

    On input dropout the BlindRolloutGuard coasts on the world model's imagination
    while its confidence stays above the floor; the moment trust runs out it emits
    a DARK signal -- the agent's honest "I have lost perception, stop acting".
    """
    head = model.world_model_head
    imag = LatentImagination(head, horizon=horizon, trust_decay=0.8)
    guard = BlindRolloutGuard(imag, confidence_floor=floor, max_blind_steps=max_blind)

    base = hidden if hidden.dim() == 2 else hidden[:, -1]       # (1, d)
    present = [True, True, True] + [False] * (max_blind + horizon + 2)

    sources: List[str] = []
    confs: List[float] = []
    for t, live in enumerate(present):
        out = guard.step(base + 0.02 * t * torch.randn_like(base) if live else None)
        sources.append(out.source)
        confs.append(out.confidence)

    return {
        "present": present,
        "sources": sources,
        "confs": confs,
        "n_coast": sum(s == "imagined" for s in sources),
        "went_dark": "dark" in sources,
    }


# ---------------------------------------------------------------------------
# Stage 6 -- ACT-SAFE: hard-clamp the steering command to physical red-lines
# ---------------------------------------------------------------------------

def act_safe(scene: Scene, chosen_path: Optional[torch.Tensor], *, dt: float,
             max_turn: float) -> dict:
    """Clamp the plan's per-step heading change to the actuator's slew limit.

    The committed path yields a stream of heading commands; a model-external
    CircuitBreaker hard-clamps each to +/- max_turn rad and trips to a safe
    hold if the planner ever emits something out of range or non-finite. We
    deliberately splice in a garbage command to prove the clamp bites.
    """
    breaker = CircuitBreaker(
        lo=-max_turn, hi=max_turn, max_rate=max_turn,
        fallback=lambda last: 0.0, trip_after=2, reset_after=2,
    )

    if chosen_path is not None and chosen_path.shape[0] >= 3:
        seg = chosen_path[1:] - chosen_path[:-1]
        head = torch.atan2(seg[:, 1], seg[:, 0])
        cmds = [float(a) for a in (head[1:] - head[:-1])]      # per-step turn
    else:
        cmds = [0.0, 0.1, -0.1]

    cmds = cmds + [float("nan"), 9.0, 0.05]                    # garbage + spike + sane
    rows = [breaker.step(c) for c in cmds]
    return {
        "n_cmds": len(cmds),
        "n_tripped": sum(r.tripped for r in rows),
        "all_in_bounds": all(-max_turn - 1e-9 <= r.value <= max_turn + 1e-9
                             and math.isfinite(r.value) for r in rows),
        "values": [r.value for r in rows],
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_demo(args) -> dict:
    scene = build_scene()
    dt, v_max, a_max, max_turn = 0.1, 30.0, 600.0, math.radians(45.0)

    perc = perceive(scene)
    mem = remember(scene, seed=args.seed)
    att, model, hidden = attend(scene, seed=args.seed, gate=args.gate)
    plan = imagine_and_plan(scene, model, hidden, horizon=args.horizon,
                            dt=dt, v_max=v_max, a_max=a_max)
    surv = survive(model, hidden, horizon=args.horizon, floor=args.floor,
                   max_blind=args.max_blind)
    safe = act_safe(scene, plan["chosen_path"], dt=dt, max_turn=max_turn)

    ok = (
        perc["mean_offdiag_cos"] < 0.9 and perc["self_cos"] > 0.99      # codes separate
        and mem["clean_recall"] > 0.9 and mem["fuzzy_recall"] > 0.5     # map recalls
        and att["closed_diff"] == 0.0 and att["open_diff"] > 1e-4       # goal steers
        and att["hidden_finite"]
        and plan["added_params"] == 0 and plan["conf_decays"]           # imagination
        and plan["straight_blocked"] and plan["chosen"] is not None     # plan avoids
        and surv["n_coast"] > 0 and surv["went_dark"]                   # coast -> dark
        and safe["all_in_bounds"]                                       # safe output
    )
    return {
        "perceive": perc, "remember": mem, "attend": att,
        "plan": plan, "survive": surv, "act_safe": safe, "ok": bool(ok),
        "scene": scene,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _render_map(scene: Scene, chosen_path: Optional[torch.Tensor],
                w: int = 41, h: int = 19) -> str:
    grid = [[" " for _ in range(w)] for _ in range(h)]

    def col(x):
        return max(0, min(w - 1, int(x * (w - 1))))

    def row(y):
        return max(0, min(h - 1, int((1.0 - y) * (h - 1))))

    if chosen_path is not None:
        for p in chosen_path:
            grid[row(float(p[1]))][col(float(p[0]))] = "."
    for o in scene.obstacles:
        grid[row(float(o[1]))][col(float(o[0]))] = "#"
    grid[row(float(scene.start[1]))][col(float(scene.start[0]))] = "S"
    grid[row(float(scene.goal[1]))][col(float(scene.goal[0]))] = "G"
    return "\n".join("  " + "".join(rr) for rr in grid)


def print_report(s: dict) -> None:
    line = "=" * 70
    print(line)
    print("AUTONOMOUS COGNITIVE AGENT -- perceive, remember, attend, imagine,")
    print("                              survive, act-safe : one closed loop")
    print(line)

    scene = s["scene"]
    print("\n  scene map (S start, G goal, # obstacle, . committed path):\n")
    print(_render_map(scene, s["plan"]["chosen_path"]))

    p = s["perceive"]
    print(f"\n  [1] PERCEIVE  L1 grid cells -> {p['n_objects']} objects in a "
          f"{p['code_dim']}-d place code")
    print(f"        mean off-diagonal cosine = {p['mean_offdiag_cos']:.3f} "
          f"(want <1: places separate),  self-cosine = {p['self_cos']:.3f}")

    m = s["remember"]
    print(f"\n  [2] REMEMBER  L2 SpatialMemory ({m['n_place']} fields, "
          f"{m['mem_params']} trainable params)")
    print(f"        clean recall = {m['clean_recall']:.3f}, fuzzy-cue recall = "
          f"{m['fuzzy_recall']:.3f}  (pattern completion)")

    a = s["attend"]
    print(f"\n  [3] ATTEND    top-down goal injection (gate={a['gate']})")
    print(f"        gate CLOSED: max|hidden(goal)-hidden(none)| = "
          f"{a['closed_diff']:.2e}  (strict no-op)")
    print(f"        gate OPEN  : max|hidden(steered)-hidden(none)| = "
          f"{a['open_diff']:.3f}  (the goal steers)")

    pl = s["plan"]
    print(f"\n  [4] IMAGINE   L4 latent rollout (added params={pl['added_params']}, "
          f"confidence decays={pl['conf_decays']}) + physics validation")
    print("        candidate            collisions  max|v|  max|a|  length  verdict")
    for name, r in pl["candidates"].items():
        verdict = "FEASIBLE" if (r["collision_free"] and r["kinematically_ok"]) else \
                  ("HITS OBSTACLE" if not r["collision_free"] else "INFEASIBLE")
        mark = " <- COMMIT" if name == pl["chosen"] else ""
        print(f"        {name:18s}     {r['collisions']:>4d}   {r['max_speed']:6.2f}  "
              f"{r['max_accel']:6.1f}  {r['length']:5.2f}  {verdict}{mark}")

    sv = s["survive"]
    served = "".join({"live": "L", "imagined": "~", "dark": "X"}[x]
                     for x in sv["sources"])
    feed = "".join("#" if x else "." for x in sv["present"])
    print(f"\n  [5] SURVIVE   sensor blackout -> blind rollout on the world model")
    print(f"        feed   : {feed}   (# present, . dropped)")
    print(f"        served : {served}   (L live, ~ imagined, X dark)")
    print(f"        coasted {sv['n_coast']} tick(s) on imagination, then went DARK "
          f"= '{('lost perception, stop' if sv['went_dark'] else 'still trusting')}'")

    sf = s["act_safe"]
    print(f"\n  [6] ACT-SAFE  CircuitBreaker hard envelope (+/- 45 deg/step)")
    print(f"        {sf['n_cmds']} commands (incl. NaN + 9.0 rad spike); tripped "
          f"{sf['n_tripped']} tick(s); every output finite & in-bounds = "
          f"{sf['all_in_bounds']}")

    print("\n" + line)
    if s["ok"]:
        print("VERDICT [OK]: the layers are not a pile of parts -- this loop PERCEIVES")
        print("        the scene as a metric place code, REMEMBERS what-is-where,")
        print("        is STEERED by a top-down goal (closed=no-op, open=steers),")
        print("        IMAGINES + physically VALIDATES a collision-free plan,")
        print("        SURVIVES a blackout by coasting then honestly stopping, and")
        print("        CLAMPS its command to a safe envelope. Only top-down touches")
        print("        model.py; every other layer is zero-coupling.")
    else:
        print("VERDICT [FAIL]: the cognitive loop did not behave as specified -- see above.")
    print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description="autonomous cognitive agent closed-loop demo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate", type=float, default=1.0,
                    help="value to open each block's top-down gate to (default 1.0).")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--floor", type=float, default=0.35,
                    help="blind-rollout confidence floor (go dark below it).")
    ap.add_argument("--max-blind", type=int, default=4)
    args = ap.parse_args()
    print_report(run_demo(args))


if __name__ == "__main__":
    main()
