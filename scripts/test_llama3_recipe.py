"""Test Phase 5b recipe application on Llama-3-8B (CPU sanity check).

Verifies that Phase 5b recipe attaches correctly to Llama-3-8B without training.
Use this to validate compatibility before launching expensive Kaggle training.

Usage:
    python scripts/test_llama3_recipe.py

Expected output:
- Model loads successfully
- MT adapters attach to every 4th layer
- LoRA applies to q/k/v/o projections
- ~0.1-0.2% trainable params
- Generation works (sanity check)
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mt_lnn.recipes import apply_phase5b_recipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Llama-3 model to test (8B or 8B-Instruct)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device (cpu or cuda)",
    )
    args = parser.parse_args()

    print(f"Testing Phase 5b recipe on {args.model}")
    print(f"Device: {args.device}")
    print()

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Loaded: {args.model}")
    print(f"Base params: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Apply Phase 5b recipe
    print("Applying Phase 5b recipe...")
    print("-" * 80)
    result = apply_phase5b_recipe(model, verbose=True)
    print("-" * 80)
    print()

    # Verify results
    print("VERIFICATION")
    print("=" * 80)
    print(f"✓ Wrapped layers: {result.wrapped_layer_indices}")
    print(f"✓ Trainable params: {result.trainable_params:,} ({result.trainable_percent:.3f}%)")
    print(f"✓ LoRA applied: {result.lora_applied}")

    # Check expected properties
    checks = []

    # Check 1: Trainable params should be < 0.3%
    if result.trainable_percent < 0.3:
        checks.append("✓ Trainable % < 0.3%")
    else:
        checks.append(f"✗ Trainable % too high: {result.trainable_percent:.3f}%")

    # Check 2: Should have wrapped layers (at least 8 for 32-layer model)
    if len(result.wrapped_layer_indices) >= 8:
        checks.append(f"✓ Wrapped {len(result.wrapped_layer_indices)} layers")
    else:
        checks.append(f"✗ Only wrapped {len(result.wrapped_layer_indices)} layers (expected ≥8)")

    # Check 3: LoRA should be applied
    if result.lora_applied:
        checks.append("✓ LoRA applied to attention projections")
    else:
        checks.append("✗ LoRA not applied")

    print()
    for check in checks:
        print(check)

    # Sanity check: generate a few tokens
    print()
    print("Testing generation...")
    model.to(args.device)
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(args.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")
    print()

    # Final verdict
    print("=" * 80)
    if all("✓" in check for check in checks):
        print("✓ SUCCESS: Phase 5b recipe ready for Llama-3-8B")
        print()
        print("Next steps:")
        print("1. Upload kaggle/awareliquid_train_llama3_phase5b.py to Kaggle")
        print("2. Run training (~4h on T4)")
        print("3. Expect PPL improvement in -25% to -35% range")
        print("4. If successful, validates recipe on third model family")
    else:
        print("✗ FAILED: Issues detected")
        print("Check errors above before running expensive training")

    print("=" * 80)


if __name__ == "__main__":
    main()
