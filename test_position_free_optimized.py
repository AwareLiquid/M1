"""
Optimized Position-Free Architecture Test

Tests the improved position-free architecture with:
1. Stronger h_prev position signal (0.01 -> 0.05)
2. Higher polarity rank (8 -> 16)
3. Better tau_weights initialization (1/sqrt(tau))
4. Longer training for convergence

Usage: python test_position_free_optimized.py
"""

import torch
import warnings
from mt_lnn import MTLNNConfig, MTLNNModel
from mt_lnn.utils import count_parameters
from benchmarks.selective_copy import (
    SelectiveCopyConfig,
    train_selective_copy,
    evaluate_selective_copy,
)

warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)


def optimized_config(use_position_free: bool = False):
    """
    Config with optimized position-free settings.

    Changes from baseline:
    - h_prev_position_weight: 0.01 -> 0.05 (5x stronger position signal)
    - polarity_rank: 8 -> 16 (2x content-based attention capacity)
    - tau_weights: Now initialized as 1/sqrt(tau) instead of uniform
    """
    return MTLNNConfig(
        vocab_size=16,
        max_seq_len=128,
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_head=32,
        dropout=0.1,
        attention_dropout=0.1,
        n_protofilaments=13,
        n_time_scales=5,
        # Position-free settings (now optimized)
        use_position_free_attention=use_position_free,
        h_prev_position_weight=0.05,        # Was 0.01
        keep_relative_bias=True,
        position_free_mode="hybrid",
        polarity_mode="low_rank" if use_position_free else "scalar",
        polarity_rank=16,                    # Was 8
    )


def run_comparison_test(
    training_steps: int = 1000,
    eval_batches: int = 16,
    device: str = "cpu",
):
    """
    Compare baseline vs optimized position-free with extended training.

    Args:
        training_steps: Number of training steps (1000 recommended)
        eval_batches: Number of evaluation batches
        device: "cpu" or "cuda"
    """
    print("=" * 70)
    print("OPTIMIZED POSITION-FREE ARCHITECTURE TEST")
    print("=" * 70)
    print(f"Training steps: {training_steps}")
    print(f"Device: {device}")
    print("\nOptimizations:")
    print("  [+] h_prev_position_weight: 0.01 -> 0.05 (5x stronger)")
    print("  [+] polarity_rank: 8 -> 16 (2x capacity)")
    print("  [+] tau_weights init: uniform -> 1/sqrt(tau) (prioritize fast scales)")
    print("=" * 70)

    # Task configuration - same as before
    task_cfg = SelectiveCopyConfig(
        K_mem=4,
        T_noise=32,
        vocab_size=16,
        batch=16,
        steps=training_steps,
        lr=3e-3,
        eval_batches=eval_batches,
        log_every=training_steps // 10,  # 10 progress updates
    )

    results = {}

    for use_pf in [False, True]:
        mode = "Position-Free (Optimized)" if use_pf else "Original (RoPE)"
        print(f"\n{'='*70}")
        print(f"{mode.upper()}")
        print("=" * 70)

        cfg = optimized_config(use_position_free=use_pf)
        model = MTLNNModel(cfg).to(device)

        param_info = count_parameters(model)
        print(f"Parameters: {param_info['total']:,}")

        # Show tau_weights initialization for position-free
        if use_pf:
            tau_vals = cfg.resonance_freqs
            tau_init = [1.0 / (t ** 0.5) for t in tau_vals]
            print(f"\nTau-weight initialization:")
            for i, (tau, w) in enumerate(zip(tau_vals, tau_init)):
                print(f"  tau={tau:5.2f} -> weight={w:.3f}")
            print(f"  (Fast scales get higher initial weight)")

        # Train
        print(f"\nTraining ({training_steps} steps):")
        history = train_selective_copy(
            model, task_cfg, device=device, verbose=True
        )

        # Evaluate
        print("\nEvaluating:")
        metrics = evaluate_selective_copy(
            model, task_cfg, device=device, n_batches=eval_batches
        )

        print(f"\nFinal Results:")
        print(f"  Token Accuracy:    {metrics['token_accuracy']:.4f}")
        print(f"  Sequence Exact:    {metrics['sequence_exact']:.4f}")

        results[mode] = {
            "metrics": metrics,
            "final_loss": history[-1][1],
            "final_acc": history[-1][2],
            "params": param_info['total'],
        }

    # Comparison
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    baseline_acc = results["Original (RoPE)"]["metrics"]["token_accuracy"]
    optimized_acc = results["Position-Free (Optimized)"]["metrics"]["token_accuracy"]
    ratio = optimized_acc / baseline_acc if baseline_acc > 0 else 0.0

    baseline_seq = results["Original (RoPE)"]["metrics"]["sequence_exact"]
    optimized_seq = results["Position-Free (Optimized)"]["metrics"]["sequence_exact"]

    print(f"\nToken Accuracy:")
    print(f"  Original (RoPE):           {baseline_acc:.4f}")
    print(f"  Position-Free (Optimized): {optimized_acc:.4f}")
    print(f"  Ratio:                     {ratio:.2%}")

    print(f"\nSequence Exact Match:")
    print(f"  Original (RoPE):           {baseline_seq:.4f}")
    print(f"  Position-Free (Optimized): {optimized_seq:.4f}")

    # Improvement over previous 87%
    prev_ratio = 0.8718
    improvement = ratio - prev_ratio
    print(f"\nImprovement over baseline (87.18%):")
    print(f"  Current ratio:   {ratio:.2%}")
    print(f"  Previous ratio:  {prev_ratio:.2%}")
    print(f"  Improvement:     {improvement:+.2%}")

    if ratio >= 0.95:
        verdict = "[PASS] Position-free reaches 95%+ of baseline"
    elif ratio >= 0.90:
        verdict = "[VERY GOOD] Position-free reaches 90-95% of baseline"
    elif ratio >= 0.85:
        verdict = "[GOOD] Position-free reaches 85-90% of baseline"
    elif ratio >= 0.80:
        verdict = "[ACCEPTABLE] Position-free reaches 80-85% of baseline"
    else:
        verdict = "[NEEDS MORE WORK] Position-free below 80% of baseline"

    print(f"\n{verdict}")

    print("\n" + "=" * 70)
    print("PARAMETER EFFICIENCY")
    print("=" * 70)
    baseline_params = results["Original (RoPE)"]["params"]
    optimized_params = results["Position-Free (Optimized)"]["params"]
    param_overhead = (optimized_params - baseline_params) / baseline_params

    print(f"  Original:       {baseline_params:,} params")
    print(f"  Position-Free:  {optimized_params:,} params")
    print(f"  Overhead:       {param_overhead:.1%}")
    print(f"  Performance/Param: {ratio / (1 + param_overhead):.2%}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000,
                        help="Training steps (default: 1000)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"], help="Device to use")
    parser.add_argument("--eval-batches", type=int, default=16,
                        help="Number of eval batches")

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    results = run_comparison_test(
        training_steps=args.steps,
        eval_batches=args.eval_batches,
        device=device,
    )

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
