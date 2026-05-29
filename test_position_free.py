"""
Test script for Position-Free Architecture.

Verifies that:
1. Position-free mode runs without errors
2. Both RoPE and position-free paths work
3. Position-free mode preserves reasonable performance

Usage: python test_position_free.py
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


def small_200k_config(use_position_free: bool = False):
    """
    Create a ~200K parameter config for testing.

    Original (RoPE) parameters:
    - vocab: 200, d_model: 128, n_layers: 2, n_heads: 4
    - This gives ~200K parameters

    Position-free adds:
    - tau_weights: 5 floats
    - h_prev_position_proj: 13*10 * 128 = 16,640 params
    Total extra: ~17K params → still ~217K total
    """
    return MTLNNConfig(
        vocab_size=200,
        max_seq_len=64,
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,        # GQA: 2 KV heads, 4 Q heads
        d_head=32,
        dropout=0.1,
        attention_dropout=0.1,
        n_protofilaments=13,
        n_time_scales=5,
        # Position-free settings
        use_position_free_attention=use_position_free,
        h_prev_position_weight=0.01,  # Weak position signal
        keep_relative_bias=True,      # Keep GTP-cap
        position_free_mode="hybrid",  # KV cache + h_prev timing
        # Enable low-rank polarity for content-based attention
        polarity_mode="low_rank" if use_position_free else "scalar",
        polarity_rank=8,
    )


def test_basic_forward():
    """Test that both modes can run a forward pass."""
    print("=" * 60)
    print("TEST 1: Basic Forward Pass")
    print("=" * 60)

    for use_pf in [False, True]:
        mode = "Position-Free" if use_pf else "Original (RoPE)"
        print(f"\n{mode} mode:")

        cfg = small_200k_config(use_position_free=use_pf)
        model = MTLNNModel(cfg).eval()
        n_params = count_parameters(model)["total"]
        print(f"  Parameters: {n_params:,}")

        # Basic forward
        ids = torch.randint(0, cfg.vocab_size, (2, 16))
        with torch.no_grad():
            out = model(ids)

        print(f"  Output shape: {out['logits'].shape}")
        print(f"  [OK] Forward pass successful")


def test_incremental_decode():
    """Test that KV caching works in both modes."""
    print("\n" + "=" * 60)
    print("TEST 2: Incremental Decoding with KV Cache")
    print("=" * 60)

    for use_pf in [False, True]:
        mode = "Position-Free" if use_pf else "Original (RoPE)"
        print(f"\n{mode} mode:")

        cfg = small_200k_config(use_position_free=use_pf)
        model = MTLNNModel(cfg).eval()

        # Prefill
        prefix = torch.randint(0, cfg.vocab_size, (1, 8))
        with torch.no_grad():
            out = model(prefix, use_cache=True)
            cache = out["cache"]

        # Incremental decode 3 tokens
        for i in range(3):
            next_tok = torch.randint(0, cfg.vocab_size, (1, 1))
            with torch.no_grad():
                out = model(next_tok, cache=cache, use_cache=True)
                cache = out["cache"]

        print(f"  [OK] Incremental decode successful")
        print(f"  Final token count: {cache.token_count}")


def test_selective_copy_comparison():
    """
    Train both modes on Selective Copy and compare performance.

    This is the critical test: position-free mode should reach
    at least 95% of original baseline accuracy.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Selective Copy Benchmark")
    print("=" * 60)

    task_cfg = SelectiveCopyConfig(
        K_mem=4,
        T_noise=32,
        vocab_size=16,
        batch=8,
        steps=200,      # Quick test, not full training
        lr=3e-3,
        eval_batches=4,
        log_every=50,
    )

    results = {}

    for use_pf in [False, True]:
        mode = "Position-Free" if use_pf else "Original (RoPE)"
        print(f"\n{mode} mode:")
        print("-" * 60)

        cfg = small_200k_config(use_position_free=use_pf)
        # Override vocab_size to match task
        cfg.vocab_size = task_cfg.vocab_size
        model = MTLNNModel(cfg)

        n_params = count_parameters(model)["total"]
        print(f"Model parameters: {n_params:,}")

        # Train
        print("\nTraining:")
        history = train_selective_copy(model, task_cfg, device="cpu", verbose=True)

        # Evaluate
        print("\nEvaluating:")
        metrics = evaluate_selective_copy(model, task_cfg, device="cpu")

        print(f"\nFinal Results:")
        print(f"  Token Accuracy:    {metrics['token_accuracy']:.3f}")
        print(f"  Sequence Exact:    {metrics['sequence_exact']:.3f}")

        results[mode] = metrics

    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    baseline_acc = results["Original (RoPE)"]["token_accuracy"]
    posfree_acc = results["Position-Free"]["token_accuracy"]
    ratio = posfree_acc / baseline_acc if baseline_acc > 0 else 0.0

    print(f"Original (RoPE):    {baseline_acc:.3f}")
    print(f"Position-Free:      {posfree_acc:.3f}")
    print(f"Ratio:              {ratio:.2%}")

    if ratio >= 0.95:
        print("\n[PASS] Position-free reaches 95%+ of baseline")
    elif ratio >= 0.80:
        print("\n[PARTIAL] Position-free reaches 80-95% of baseline")
    else:
        print("\n[FAIL] Position-free below 80% of baseline")

    return results


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("POSITION-FREE ARCHITECTURE TEST SUITE")
    print("=" * 60)

    # Run tests
    test_basic_forward()
    test_incremental_decode()
    results = test_selective_copy_comparison()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
