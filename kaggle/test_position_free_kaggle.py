"""
Position-Free Architecture Test - Kaggle Version

This notebook validates the position-free architecture on Kaggle's GPU environment.
Extended training for more reliable performance comparison.

Usage:
1. Upload this file + mt_lnn/ + benchmarks/ to Kaggle
2. Run: python test_position_free_kaggle.py
3. GPU recommended for faster training
"""

import torch
import warnings
import json
from pathlib import Path

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn import MTLNNConfig, MTLNNModel
from mt_lnn.utils import count_parameters
from benchmarks.selective_copy import (
    SelectiveCopyConfig,
    train_selective_copy,
    evaluate_selective_copy,
)


def get_config(use_position_free: bool = False, size: str = "small"):
    """
    Create configs for different model sizes.

    Args:
        use_position_free: Enable position-free architecture
        size: "tiny" (~200K), "small" (~1M), "medium" (~5M)
    """
    configs = {
        "tiny": {
            "d_model": 128,
            "n_layers": 2,
            "n_heads": 4,
            "n_kv_heads": 2,
            "d_head": 32,
        },
        "small": {
            "d_model": 256,
            "n_layers": 4,
            "n_heads": 8,
            "n_kv_heads": 2,
            "d_head": 32,
        },
        "medium": {
            "d_model": 512,
            "n_layers": 6,
            "n_heads": 8,
            "n_kv_heads": 2,
            "d_head": 64,
        },
    }

    params = configs[size]

    return MTLNNConfig(
        vocab_size=32,  # Selective Copy needs small vocab
        max_seq_len=128,
        d_model=params["d_model"],
        n_layers=params["n_layers"],
        n_heads=params["n_heads"],
        n_kv_heads=params["n_kv_heads"],
        d_head=params["d_head"],
        dropout=0.1,
        attention_dropout=0.1,
        n_protofilaments=13,
        n_time_scales=5,
        # Position-free settings
        use_position_free_attention=use_position_free,
        h_prev_position_weight=0.05 if use_position_free else 0.01,  # Slightly stronger
        keep_relative_bias=True,
        position_free_mode="hybrid",
        polarity_mode="low_rank" if use_position_free else "scalar",
        polarity_rank=16,  # Larger rank for better content-based attention
        # GWTB for long-term memory
        gwtb_per_block=False,
        # Dynamic scale gates
        dynamic_scale_gates=True,
    )


def run_experiment(
    model_size: str = "small",
    training_steps: int = 2000,
    eval_batches: int = 32,
    save_checkpoints: bool = True,
):
    """
    Run full experiment comparing RoPE vs Position-Free.

    Args:
        model_size: "tiny", "small", or "medium"
        training_steps: Number of training steps
        eval_batches: Number of evaluation batches
        save_checkpoints: Save model checkpoints
    """
    print("\n" + "=" * 70)
    print(f"POSITION-FREE ARCHITECTURE EXPERIMENT")
    print(f"Model Size: {model_size} | Steps: {training_steps} | Device: {device}")
    print("=" * 70)

    # Task configuration
    task_cfg = SelectiveCopyConfig(
        K_mem=6,           # 6 tokens to remember (harder than default 4)
        T_noise=64,        # 64 noise tokens (longer context)
        vocab_size=32,
        batch=32,
        steps=training_steps,
        lr=3e-4,           # Lower LR for stability
        eval_batches=eval_batches,
        log_every=training_steps // 20,  # 20 progress updates
    )

    results = {}
    checkpoints = {}

    for use_pf in [False, True]:
        mode = "Position-Free" if use_pf else "Original (RoPE)"
        print(f"\n{'='*70}")
        print(f"{mode.upper()} MODE")
        print("=" * 70)

        # Create model
        cfg = get_config(use_position_free=use_pf, size=model_size)
        model = MTLNNModel(cfg).to(device)

        param_info = count_parameters(model)
        print(f"Parameters: {param_info['total']:,}")
        print(f"  - Trainable: {param_info['trainable']:,}")
        print(f"  - Position-free extra: {param_info['total'] - 200000 if use_pf else 0:,}")

        # Train
        print("\nTraining:")
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
            "history": history,
            "params": param_info,
            "config": {
                "model_size": model_size,
                "use_position_free": use_pf,
                "training_steps": training_steps,
                "h_prev_weight": cfg.h_prev_position_weight if use_pf else None,
                "polarity_mode": cfg.polarity_mode,
            }
        }

        # Save checkpoint
        if save_checkpoints:
            ckpt_name = f"selective_copy_{mode.lower().replace(' ', '_')}_{model_size}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "metrics": metrics,
                "history": history,
            }, ckpt_name)
            checkpoints[mode] = ckpt_name
            print(f"  Saved checkpoint: {ckpt_name}")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    baseline_acc = results["Original (RoPE)"]["metrics"]["token_accuracy"]
    posfree_acc = results["Position-Free"]["metrics"]["token_accuracy"]
    ratio = posfree_acc / baseline_acc if baseline_acc > 0 else 0.0

    baseline_seq = results["Original (RoPE)"]["metrics"]["sequence_exact"]
    posfree_seq = results["Position-Free"]["metrics"]["sequence_exact"]

    print(f"\nToken Accuracy:")
    print(f"  Original (RoPE):    {baseline_acc:.4f}")
    print(f"  Position-Free:      {posfree_acc:.4f}")
    print(f"  Ratio:              {ratio:.2%}")

    print(f"\nSequence Exact Match:")
    print(f"  Original (RoPE):    {baseline_seq:.4f}")
    print(f"  Position-Free:      {posfree_seq:.4f}")

    if ratio >= 0.95:
        verdict = "[PASS] Position-free reaches 95%+ of baseline"
    elif ratio >= 0.85:
        verdict = "[GOOD] Position-free reaches 85-95% of baseline"
    elif ratio >= 0.75:
        verdict = "[PARTIAL] Position-free reaches 75-85% of baseline"
    else:
        verdict = "[NEEDS WORK] Position-free below 75% of baseline"

    print(f"\n{verdict}")

    # Save results
    results_file = f"position_free_results_{model_size}.json"
    with open(results_file, "w") as f:
        # Convert history to serializable format
        for mode in results:
            results[mode]["history"] = [
                {"step": h[0], "loss": h[1], "acc": h[2]}
                for h in results[mode]["history"]
            ]
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return results, checkpoints


def quick_test():
    """Quick smoke test - runs in ~30 seconds."""
    print("\n" + "=" * 70)
    print("QUICK SMOKE TEST (30 seconds)")
    print("=" * 70)

    task_cfg = SelectiveCopyConfig(
        K_mem=4, T_noise=16, vocab_size=16,
        batch=8, steps=50, lr=3e-3, eval_batches=4, log_every=25,
    )

    for use_pf in [False, True]:
        mode = "Position-Free" if use_pf else "RoPE"
        cfg = get_config(use_position_free=use_pf, size="tiny")
        cfg.vocab_size = 16
        model = MTLNNModel(cfg).to(device)

        print(f"\n{mode}: Training...")
        train_selective_copy(model, task_cfg, device=device, verbose=False)
        metrics = evaluate_selective_copy(model, task_cfg, device=device)
        print(f"{mode}: {metrics['token_accuracy']:.3f} token acc")

    print("\n[OK] Both modes work")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "tiny", "small", "medium"],
                        default="small", help="Test mode")
    parser.add_argument("--steps", type=int, default=2000,
                        help="Training steps")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save checkpoints")

    args = parser.parse_args()

    if args.mode == "quick":
        quick_test()
    else:
        results, ckpts = run_experiment(
            model_size=args.mode,
            training_steps=args.steps,
            save_checkpoints=not args.no_save,
        )

        print("\n" + "=" * 70)
        print("EXPERIMENT COMPLETE")
        print("=" * 70)
        print(f"\nCheckpoints: {ckpts}")
