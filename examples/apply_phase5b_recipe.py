"""Example: Apply Phase 5b recipe to any HuggingFace model.

Demonstrates the simplified API for applying the validated Phase 5b adapter recipe.

Usage:
    # Apply to Qwen-1.5B
    python examples/apply_phase5b_recipe.py --model Qwen/Qwen2.5-1.5B-Instruct

    # Apply to Llama-3-8B
    python examples/apply_phase5b_recipe.py --model meta-llama/Llama-3-8B

    # MT adapters only (no LoRA)
    python examples/apply_phase5b_recipe.py --model Qwen/Qwen2.5-3B-Instruct --no-lora

    # Custom LoRA rank
    python examples/apply_phase5b_recipe.py --model Qwen/Qwen2.5-1.5B-Instruct --lora-rank 16
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mt_lnn.recipes import apply_phase5b_recipe


def main():
    parser = argparse.ArgumentParser(description="Apply Phase 5b recipe to a HuggingFace model")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model identifier",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Skip LoRA (MT adapters only)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank (default: 8)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha (default: 16)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (default: 0.05)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    print(f"Device: {args.device}")
    print()

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Apply Phase 5b recipe
    print("Applying Phase 5b recipe...")
    print("-" * 60)
    result = apply_phase5b_recipe(
        model,
        lora_rank=0 if args.no_lora else args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        verbose=True,
    )
    print("-" * 60)
    print()

    # Summary
    print("Recipe Applied Successfully!")
    print(f"  Wrapped layers: {result.wrapped_layer_indices}")
    print(f"  Trainable params: {result.trainable_params:,} ({result.trainable_percent:.3f}%)")
    print(f"  LoRA applied: {result.lora_applied}")
    print()

    # Move to device and test generation
    model.to(args.device)
    print("Testing generation...")
    prompt = "The capital of Australia is"
    inputs = tokenizer(prompt, return_tensors="pt").to(args.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    print()
    print("Done! Model is ready for training or inference.")


if __name__ == "__main__":
    main()
