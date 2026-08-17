"""
benchmarks/run_benchmark.py — End-to-end MT-LNN benchmark runner.

Runs three phases on a small TC-aligned MT-LNN:

  1. **Selective Copy** — Mamba's signature long-range selectivity task.
     Train ~1000 steps on randomly-generated batches.
  2. **Held-out evaluation** — greedy decoding token + sequence accuracy.
  3. **AVP / Φ̂ collapse** — anesthesia validation on the trained model.

Designed to run end-to-end in ~3-5 minutes on CPU.
"""

import os
import sys
import time
import warnings

import torch

# Allow `python benchmarks/run_benchmark.py` from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Tests use tiny non-TC-aligned dims for speed; suppress the alignment warning.
warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn import (
    MTLNNConfig, MTLNNModel,
    phi_hat_anesthesia_sweep, anesthesia_test_result,
)
from mt_lnn.utils import save_checkpoint
from benchmarks.selective_copy import (
    SelectiveCopyConfig,
    train_selective_copy,
    evaluate_selective_copy,
)


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 64)
    print(" MT-LNN end-to-end benchmark: Selective Copy + AVP")
    print("=" * 64)
    print(f" device: {device}")
    print(f" torch:  {torch.__version__}")
    print()

    # ------------------------------------------------------------------
    # Task config — small but non-trivial
    # ------------------------------------------------------------------
    task = SelectiveCopyConfig(
        K_mem=4,
        T_noise=32,
        vocab_size=16,           # 4 mem + 11 noise + 1 SEP
        batch=16,
        steps=1500,
        lr=3e-3,
        eval_batches=8,
        log_every=200,
    )
    print(f" Task: Selective Copy")
    print(f"   K_mem={task.K_mem}, T_noise={task.T_noise}, T_total={task.T_total}")
    print(f"   vocab_size={task.vocab_size}, SEP={task.SEP}, batch={task.batch}, steps={task.steps}")
    print()

    # ------------------------------------------------------------------
    # Model — TC-aligned tiny MT-LNN
    # ------------------------------------------------------------------
    cfg = MTLNNConfig(
        vocab_size=task.vocab_size,
        max_seq_len=task.T_total,
        d_model=104,             # 13 × 8, TC-aligned (d_proto = 8)
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_head=26,               # 104/4
        dropout=0.0,
        attention_dropout=0.0,
        gwtb_compression_ratio=4,
        gwtb_n_heads=2,
        coherence_heads=2,
    )
    model = MTLNNModel(cfg).to(device)
    n_params = model.get_num_params()
    print(f" Model: MT-LNN")
    print(f"   params={n_params/1e3:.1f}k  d_model={cfg.d_model}  n_layers={cfg.n_layers}")
    print(f"   d_proto={cfg.d_proto}  d_proto_total={cfg.d_proto_total}")
    print(f"   d_gw (GWTB) = {model.gwtb.d_gw}")
    print()

    # ------------------------------------------------------------------
    # Phase 1: Train
    # ------------------------------------------------------------------
    print("─" * 64)
    print(" PHASE 1 — Training")
    print("─" * 64)
    t0 = time.time()
    history = train_selective_copy(model, task, device=device, verbose=True)
    train_time = time.time() - t0
    print(f"\n training done in {train_time:.1f}s")
    print(f" final loss: {history[-1][1]:.4f}   final batch acc: {history[-1][2]:.3f}")
    print()

    # ------------------------------------------------------------------
    # Phase 2: Held-out greedy-decoding evaluation
    # ------------------------------------------------------------------
    print("─" * 64)
    print(" PHASE 2 — Held-out Selective Copy evaluation")
    print("─" * 64)
    t0 = time.time()
    eval_result = evaluate_selective_copy(model, task, device=device, n_batches=16)
    eval_time = time.time() - t0
    print(f"   token accuracy   : {eval_result['token_accuracy']:.3f}")
    print(f"   sequence exact   : {eval_result['sequence_exact']:.3f}")
    print(f"   n sequences      : {eval_result['n_sequences']}")
    print(f"   eval time        : {eval_time:.1f}s")
    print()

    # Random-guess baselines for reference
    # Each memorable token is uniform over K_mem values → 1/K_mem token accuracy
    # Sequence: (1/K_mem)^K_mem
    rand_tok = 1.0 / task.K_mem
    rand_seq = rand_tok ** task.K_mem
    print(f"   random baseline (token):    {rand_tok:.3f}")
    print(f"   random baseline (sequence): {rand_seq:.4f}")
    print()

    # ------------------------------------------------------------------
    # Phase 3: AVP — anesthesia Φ̂ collapse
    # ------------------------------------------------------------------
    print("─" * 64)
    print(" PHASE 3 — Anesthesia Validation Protocol (AVP)")
    print("─" * 64)
    # Use a real Selective-Copy batch as the activation source for Φ̂
    from benchmarks.selective_copy import make_selective_copy_batch
    ids, _ = make_selective_copy_batch(task, B=4, device=device)
    t0 = time.time()
    sweep = phi_hat_anesthesia_sweep(model, ids,
                                       kappas=[1.0, 2.0, 5.0, 10.0],
                                       K=4, k_nn=3)
    result = anesthesia_test_result(sweep, delta=0.7)
    avp_time = time.time() - t0

    print(f"   Phi_hat vs anesthesia level:")
    for kappa, phi in sweep.items():
        print(f"      kappa = {kappa:>4.1f}   Phi_hat = {phi:+.4f}")
    print(f"   Phi_hat(clean)     = {result['phi_clean']:+.4f}")
    print(f"   Phi_hat(full)      = {result['phi_full']:+.4f}")
    print(f"   absolute change    = {result['abs_change']:+.4f}")
    print(f"   signed rel change  = {result['signed_relative_change']*100:+.1f}%")
    print(f"   collapse_pct       = {result['collapse_pct']:.1f}%  "
          f"(threshold {result['delta_threshold']*100:.0f}%)")
    print(f"   monotone decrease  = {result['monotone_decrease']}")
    print(f"   AVP                = {'PASSED' if result['passed'] else 'FAILED'}")
    print(f"   eval time      : {avp_time:.1f}s")
    print()

    # MT diagnostics
    diag = model.get_mt_diagnostics()
    print("─" * 64)
    print(" Final MT diagnostics")
    print("─" * 64)
    for k, v in sorted(diag.items()):
        print(f"   {k:36s}: {v:+.4f}")
    print()

    # Save checkpoint (we keep only model + config; no optimiser state needed
    # to reproduce the benchmark numbers)
    os.makedirs("checkpoints", exist_ok=True)
    import dataclasses
    torch.save({
        "step": task.steps,
        "loss": history[-1][1],
        "model_state": model.state_dict(),
        "config": dataclasses.asdict(cfg),
        "task_config": dataclasses.asdict(task),
        "benchmark": {
            "selective_copy": eval_result,
            "avp": result,
            "phi_sweep": sweep,
            "diagnostics": diag,
        },
    }, "checkpoints/selective_copy.pt")
    print(f" Checkpoint saved → checkpoints/selective_copy.pt")
    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 64)
    print(" SUMMARY")
    print("=" * 64)
    print(f"   Selective Copy token accuracy:    {eval_result['token_accuracy']:.3f}  "
          f"(random {rand_tok:.3f})")
    print(f"   Selective Copy sequence exact:    {eval_result['sequence_exact']:.3f}  "
          f"(random {rand_seq:.4f})")
    print(f"   AVP Phi_hat collapse:             {result['collapse_pct']:.1f}%  "
          f"({'PASS' if result['passed'] else 'fail'})")
    print(f"   wall-clock (train + eval + AVP):  {train_time + eval_time + avp_time:.1f}s")
    print("=" * 64)


if __name__ == "__main__":
    main()
