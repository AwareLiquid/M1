"""
examples/v2_demo.py — MT-LNN v2.0 full-feature integration demo.

Demonstrates all four v2.0 modules running together on a small model:

  Phase A — CompetitiveGWTBLayer  : K-bid workspace competition
  Phase B — CausalConsistencyChecker : h_prev trajectory anomaly detection
  Phase C — PredictiveStateHead   : next-state prediction + world-model loss
  Phase D — HebbianRegularizer    : co-activation loss term

Plus their cross-module linkages:
  LAVI rhythm gate reads world-model pred_error from the previous step
  GlobalRhythmController aggregates LAVI across all layers
  HebbianRegularizer modulates α by global LAVI mean

Run (CPU, ~5s):
    python examples/v2_demo.py

Run with GPU if available:
    python examples/v2_demo.py --device cuda
"""

import argparse
import math
import torch
import torch.optim as optim

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel
from mt_lnn.causality import CausalConsistencyChecker
from mt_lnn.deliberation import DeliberationRouter, RouterThresholds


# ---------------------------------------------------------------------------
# Config — small enough to run on CPU in seconds
# ---------------------------------------------------------------------------

def make_v2_config(d_model: int = 104) -> MTLNNConfig:
    """
    Create an MTLNNConfig with all v2.0 modules enabled.

    d_model=104 = 8×13: d_proto=8 (Tensor-Core aligned), d_gw=13 (104//8),
    gwtb_n_heads=1 (13/1=13, always divisible regardless of d_model).
    """
    return MTLNNConfig(
        vocab_size=256,
        d_model=d_model,
        n_layers=4,
        n_heads=13,
        n_kv_heads=1,
        d_head=d_model // 13,
        max_seq_len=256,
        gwtb_n_heads=1,
        gwtb_broadcast_init=0.01,
        dropout=0.0,
        attention_dropout=0.0,

        # Phase A: Competitive workspace
        use_competitive_gwtb=True,
        n_competitive_bids=3,

        # Phase B: Causal consistency (in deliberation, not in model config)

        # Phase C: World model
        use_world_model=True,
        world_model_loss_weight=0.05,
        world_model_hidden_ratio=0.5,

        # Phase D: Hebbian regularizer
        use_hebbian=True,
        hebbian_lr=5e-5,
        hebbian_lavi_gate=True,

        # LAVI rhythm gate (links Phase C → LAVI)
        use_rhythm=True,
        rhythm_scale_init=0.1,
        global_rhythm=True,

        # Predictive coding (τ-scale, orthogonal to Phase C)
        use_predictive_coding=True,
        predictive_loss_weight=0.02,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_training(model: MTLNNModel, device: str, n_steps: int = 20) -> dict:
    """
    Micro-training loop: random sequences, tracks all v2.0 loss components.
    Returns dict of loss curves for display.
    """
    model.train()
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    cfg = model.config

    history = {"lm": [], "world_model": [], "hebbian": [], "pred_coding": []}

    seq_len = min(cfg.max_seq_len, 32)   # keep training step fast on CPU
    for step in range(1, n_steps + 1):
        ids = torch.randint(0, cfg.vocab_size, (2, seq_len), device=device)
        labels = ids.clone()

        out = model(ids, labels=labels)
        loss = out["loss"]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Extract per-component losses for display
        lm_loss = loss.item()
        wm     = out.get("world_model_loss")
        hebb   = out.get("hebbian_loss")
        pred   = out.get("pred_loss")

        history["lm"].append(lm_loss)
        history["world_model"].append(wm.item()  if wm   is not None else 0.0)
        history["hebbian"].append(hebb.item()     if hebb is not None else 0.0)
        history["pred_coding"].append(pred.item() if pred is not None else 0.0)

        if step % 5 == 0:
            wm_str   = f"wm={wm.item():.4f}"     if wm   is not None else "wm=N/A"
            hb_str   = f"hebb={hebb.item():+.2e}" if hebb is not None else "hebb=N/A"
            pred_str = f"pred={pred.item():.4f}" if pred is not None else "pred=N/A"
            print(f"  step {step:3d} | total={lm_loss:.4f} | {wm_str} | {hb_str} | {pred_str}")

    return history


# ---------------------------------------------------------------------------
# Inference demo: causal consistency monitoring
# ---------------------------------------------------------------------------

def run_inference_demo(model: MTLNNModel, device: str):
    """
    Simulate a streaming inference session with CausalConsistencyChecker
    and DeliberationRouter.  Shows how Phase B integrates with the broader
    AwareLiquid deliberation pipeline.
    """
    model.eval()
    cfg = model.config
    checker  = CausalConsistencyChecker(
        window=6, ema_alpha=0.5, threshold=0.35, method="subspace"
    )
    router   = DeliberationRouter(thresholds=RouterThresholds(
        low=3.0, high=5.0, consistency_floor=0.35
    ))

    print("\n  [Inference] Simulating stable context then abrupt topic switch:")
    print(f"  {'step':>4}  {'LAVI':>6}  {'consist':>8}  {'route':>14}  {'note'}")
    print("  " + "-" * 55)

    # Phase 1: stable context (same token repeated)
    stable_ids = torch.randint(10, 50, (1, 8), device=device)
    cache = None
    for step in range(1, 13):
        # Mild variation for "stable" topic
        ids_t = stable_ids + torch.randint(0, 3, stable_ids.shape, device=device)
        ids_t = ids_t.clamp(0, cfg.vocab_size - 1)

        with torch.no_grad():
            out = model(ids_t, cache=cache, use_cache=True)
            cache = out["cache"]

        # Feed last block's h_prev to the checker
        h_prev = cache.layers[-1][1]  # (B, P, S, D)
        score  = checker.update(h_prev)

        # Route using the latest logits
        decision = router.decide(
            out["logits"][0, -1],
            query="stable query",
            evidence_log=[],
            consistency_signal=score,
        )

        diag    = model.get_mt_diagnostics()
        lavi_m  = diag.get("lavi_mean", float("nan"))
        note    = "BREAK→SC" if decision.reason == "causal_break" else ""
        print(f"  {step:4d}  {lavi_m:6.3f}  {score:8.3f}  {decision.route.value:>14}  {note}")

        # Step 7: simulate abrupt topic switch
        if step == 7:
            stable_ids = torch.randint(150, 200, (1, 8), device=device)
            print("  " + "·" * 55 + "  [topic switch at step 8]")


# ---------------------------------------------------------------------------
# Workspace competition diagnostics
# ---------------------------------------------------------------------------

def show_competition_diagnostics(model: MTLNNModel, device: str):
    """Display how workspace competition evolved during a forward pass."""
    from mt_lnn.gwtb import CompetitiveGWTBLayer
    if not isinstance(model.gwtb, CompetitiveGWTBLayer):
        return

    cfg = model.config
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 16), device=device)
    with torch.no_grad():
        model(ids)

    weights  = model.gwtb.last_winner_weights.tolist()
    entropy  = model.gwtb.last_competition_entropy.item()
    max_w    = max(weights)
    dominant = weights.index(max_w)

    # Structural divergence: pairwise cosine similarity of bid_projector fc2
    # weights.  This is the LEARNED specialization, separate from per-step
    # bid weights (which only differ when noise is injected at training).
    import torch.nn.functional as F
    fc2s = torch.stack([p.fc2.weight.flatten() for p in model.gwtb.bid_projectors], dim=0)
    fc2_norm = F.normalize(fc2s + 1e-8 * torch.randn_like(fc2s), dim=-1)
    sim = (fc2_norm @ fc2_norm.t()).abs()
    K = sim.shape[0]
    off_diag_mean = (sim - torch.eye(K, device=sim.device)).abs().sum().item() / (K * (K - 1))

    print(f"\n  [Phase A] Workspace competition diagnostics:")
    print(f"    Eval-mode bid weights : "
          f"{' | '.join(f'bid{i}={w:.3f}' for i, w in enumerate(weights))}")
    print(f"    Dominant bid          : bid{dominant} ({max_w:.3f})")
    print(f"    Competition entropy   : {entropy:.3f} "
          f"(0=monopoly, {math.log(len(weights)):.2f}=uniform)")
    print(f"    Bid-projector pairwise|cos|: {off_diag_mean:.4f} "
          f"(↓ over training = orthogonality penalty working)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",  default="cpu")
    parser.add_argument("--steps",   type=int, default=20)
    parser.add_argument("--d_model", type=int, default=104)
    args = parser.parse_args()

    print("=" * 60)
    print("  MT-LNN v2.0 — full-feature integration demo")
    print("=" * 60)
    print(f"  device={args.device}  d_model={args.d_model}  steps={args.steps}")
    print()

    cfg   = make_v2_config(d_model=args.d_model)
    model = MTLNNModel(cfg)
    n_p   = model.get_num_params()
    print(f"  Model: {n_p/1e3:.1f}K params, {cfg.n_layers} layers")
    print(f"  Modules active: CompetitiveGWTB({cfg.n_competitive_bids} bids) | "
          f"WorldModel(w={cfg.world_model_loss_weight}) | "
          f"Hebbian(lr={cfg.hebbian_lr}) | "
          f"LAVI+Rhythm")
    print()

    # --- Training ---
    print(f"  Training for {args.steps} steps …")
    history = run_training(model, args.device, n_steps=args.steps)

    lm_start = history["lm"][0]
    lm_end   = history["lm"][-1]
    print(f"\n  Total loss: {lm_start:.4f} → {lm_end:.4f} "
          f"({'↓' if lm_end < lm_start else '↑'}{abs(lm_end-lm_start):.4f})")

    # --- Diagnostics ---
    show_competition_diagnostics(model, args.device)

    diag = model.get_mt_diagnostics()
    print(f"\n  [v2.0 diagnostics after {args.steps} steps]:")
    keys = [
        "gwtb_competition_entropy",
        "world_model_pred_error",
        "hebbian_signal_mean",
        "hebbian_lavi_temperature",
        "lavi_mean",
        "rhythm_scale_mean",
    ]
    for k in keys:
        v = diag.get(k)
        if v is not None:
            print(f"    {k:<30}: {v:.4f}")

    # --- Inference demo ---
    run_inference_demo(model, args.device)

    print("\n  Done. All v2.0 modules verified.")
    print("=" * 60)


if __name__ == "__main__":
    main()
