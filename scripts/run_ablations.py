"""Ablation study runner for MT-LNN adapter recipes.

Systematically tests different configurations to understand what drives performance.

Usage:
    # Run all ablations (requires GPU, ~8h on Kaggle T4)
    python scripts/run_ablations.py --model Qwen/Qwen2.5-1.5B-Instruct --device cuda

    # Run single ablation group
    python scripts/run_ablations.py --group layer_interval --device cuda

    # Dry run (show config without training)
    python scripts/run_ablations.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from mt_lnn.recipes import (
    RecipeResult,
    apply_lora_only_recipe,
    apply_mt_only_recipe,
    apply_phase5b_recipe,
)


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""

    name: str
    description: str
    recipe_fn: str  # 'phase5b' | 'mt_only' | 'lora_only'
    recipe_kwargs: dict
    expected_trainable_percent: Optional[float] = None


@dataclass
class AblationResult:
    """Result of running one ablation experiment."""

    name: str
    description: str
    recipe_result: RecipeResult
    final_loss: float
    final_ppl: float
    training_time_s: float
    tokens_per_sec: float
    steps_completed: int


# Ablation experiment groups
ABLATION_GROUPS = {
    "layer_interval": [
        AblationConfig(
            name="mt_every2",
            description="MT adapters every 2nd layer (more coverage, more params)",
            recipe_fn="mt_only",
            recipe_kwargs={"every": 2, "n_protofilaments": 13, "n_time_scales": 5},
        ),
        AblationConfig(
            name="mt_every4",
            description="MT adapters every 4th layer (Phase 5b default)",
            recipe_fn="mt_only",
            recipe_kwargs={"every": 4, "n_protofilaments": 13, "n_time_scales": 5},
        ),
        AblationConfig(
            name="mt_every8",
            description="MT adapters every 8th layer (sparse coverage, fewer params)",
            recipe_fn="mt_only",
            recipe_kwargs={"every": 8, "n_protofilaments": 13, "n_time_scales": 5},
        ),
    ],
    "lora_rank": [
        AblationConfig(
            name="phase5b_lora_r4",
            description="Phase 5b with LoRA rank 4 (fewer params)",
            recipe_fn="phase5b",
            recipe_kwargs={"lora_rank": 4, "lora_alpha": 16},
        ),
        AblationConfig(
            name="phase5b_lora_r8",
            description="Phase 5b with LoRA rank 8 (default)",
            recipe_fn="phase5b",
            recipe_kwargs={"lora_rank": 8, "lora_alpha": 16},
        ),
        AblationConfig(
            name="phase5b_lora_r16",
            description="Phase 5b with LoRA rank 16 (more capacity)",
            recipe_fn="phase5b",
            recipe_kwargs={"lora_rank": 16, "lora_alpha": 32},
        ),
    ],
    "adapter_type": [
        AblationConfig(
            name="mt_only",
            description="MT adapters only (no LoRA)",
            recipe_fn="mt_only",
            recipe_kwargs={"every": 4, "n_protofilaments": 13, "n_time_scales": 5},
        ),
        AblationConfig(
            name="lora_only",
            description="LoRA only (no MT, vanilla baseline)",
            recipe_fn="lora_only",
            recipe_kwargs={"lora_rank": 8, "lora_alpha": 16},
        ),
        AblationConfig(
            name="mt_plus_lora",
            description="MT + LoRA (Phase 5b full recipe)",
            recipe_fn="phase5b",
            recipe_kwargs={"lora_rank": 8, "lora_alpha": 16},
        ),
    ],
    "protofilaments": [
        AblationConfig(
            name="mt_proto8",
            description="MT adapters with 8 protofilaments",
            recipe_fn="mt_only",
            recipe_kwargs={"every": 4, "n_protofilaments": 8, "n_time_scales": 5},
        ),
        AblationConfig(
            name="mt_proto13",
            description="MT adapters with 13 protofilaments (default)",
            recipe_fn="mt_only",
            recipe_kwargs={"every": 4, "n_protofilaments": 13, "n_time_scales": 5},
        ),
        AblationConfig(
            name="mt_proto21",
            description="MT adapters with 21 protofilaments",
            recipe_fn="mt_only",
            recipe_kwargs={"every": 4, "n_protofilaments": 21, "n_time_scales": 5},
        ),
    ],
}


def apply_recipe(config: AblationConfig, model):
    """Apply the recipe specified in config."""
    recipe_map = {
        "phase5b": apply_phase5b_recipe,
        "mt_only": apply_mt_only_recipe,
        "lora_only": apply_lora_only_recipe,
    }
    recipe_fn = recipe_map[config.recipe_fn]
    return recipe_fn(model, **config.recipe_kwargs, verbose=True)


def build_dataloader(tokenizer, args):
    """Build WikiText-2 dataloader."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    def tokenize(batch):
        text = [t for t in batch["text"] if t]
        if not text:
            return {"input_ids": []}
        return tokenizer(text, add_special_tokens=False)

    tokenized = ds.map(
        tokenize, batched=True, remove_columns=ds.column_names, desc="tokenizing"
    )

    def group_texts(examples):
        ids = []
        for row in examples["input_ids"]:
            ids.extend(row + [tokenizer.eos_token_id])
        total = (len(ids) // args.seq_len) * args.seq_len
        ids = ids[:total]
        chunks = [ids[i : i + args.seq_len] for i in range(0, total, args.seq_len)]
        return {"input_ids": chunks, "labels": [c.copy() for c in chunks]}

    lm_ds = tokenized.map(
        group_texts, batched=True, remove_columns=tokenized.column_names, desc="chunking"
    )
    lm_ds.set_format(type="torch", columns=["input_ids", "labels"])
    return DataLoader(lm_ds, batch_size=args.batch, shuffle=True, drop_last=True)


def train_one_ablation(config: AblationConfig, args) -> AblationResult:
    """Train a single ablation configuration."""
    print(f"\n{'='*80}")
    print(f"Ablation: {config.name}")
    print(f"Description: {config.description}")
    print(f"{'='*80}\n")

    # Load fresh model for each ablation
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )
    model.config.use_cache = False
    if args.device == "cuda":
        model.gradient_checkpointing_enable()

    # Apply recipe
    recipe_result = apply_recipe(config, model)
    model.to(args.device)
    model.train()

    # Setup training
    loader = build_dataloader(tokenizer, args)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Training loop
    step = 0
    total_tokens = 0
    t0 = time.time()
    final_loss = 0.0

    while step < args.steps:
        for batch in loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            final_loss = loss.item() * args.grad_accum
            total_tokens += batch["input_ids"].numel()
            step += 1

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                ppl = torch.exp(torch.tensor(final_loss)).item()
                toks_per_s = total_tokens / max(elapsed, 1e-6)
                print(
                    f"  step {step:4d}/{args.steps} | loss {final_loss:.4f} | "
                    f"ppl {ppl:.2f} | {toks_per_s:.0f} tok/s"
                )

            if step >= args.steps:
                break

    wall = time.time() - t0
    final_ppl = torch.exp(torch.tensor(final_loss)).item()

    print(f"\nCompleted: {config.name}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Final PPL: {final_ppl:.2f}")
    print(f"  Training time: {wall:.1f}s")
    print(f"  Throughput: {total_tokens / wall:.0f} tok/s")

    return AblationResult(
        name=config.name,
        description=config.description,
        recipe_result=recipe_result,
        final_loss=final_loss,
        final_ppl=final_ppl,
        training_time_s=wall,
        tokens_per_sec=total_tokens / wall,
        steps_completed=step,
    )


def save_results(results: List[AblationResult], out_path: str):
    """Save ablation results to JSON."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    payload = [
        {
            "name": r.name,
            "description": r.description,
            "trainable_params": r.recipe_result.trainable_params,
            "trainable_percent": r.recipe_result.trainable_percent,
            "wrapped_layers": r.recipe_result.wrapped_layer_indices,
            "lora_applied": r.recipe_result.lora_applied,
            "final_loss": r.final_loss,
            "final_ppl": r.final_ppl,
            "training_time_s": r.training_time_s,
            "tokens_per_sec": r.tokens_per_sec,
            "steps_completed": r.steps_completed,
        }
        for r in results
    ]

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved results: {out_path}")


def print_summary_table(results: List[AblationResult]):
    """Print comparison table."""
    print("\n" + "=" * 100)
    print("ABLATION SUMMARY")
    print("=" * 100)
    print(
        f"{'Name':<20} {'Trainable %':>12} {'Final PPL':>12} {'Tok/s':>10} {'Time (s)':>10}"
    )
    print("-" * 100)
    for r in results:
        print(
            f"{r.name:<20} {r.recipe_result.trainable_percent:>11.3f}% "
            f"{r.final_ppl:>12.2f} {r.tokens_per_sec:>10.0f} {r.training_time_s:>10.1f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Run MT-LNN ablation studies")
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model to ablate"
    )
    parser.add_argument(
        "--group",
        choices=list(ABLATION_GROUPS.keys()) + ["all"],
        default="all",
        help="Which ablation group to run",
    )
    parser.add_argument("--steps", type=int, default=200, help="Training steps per ablation")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out_dir", default="artifacts/ablations")
    parser.add_argument("--dry_run", action="store_true", help="Print configs without training")
    args = parser.parse_args()

    # Select ablation configs
    if args.group == "all":
        configs = [c for group in ABLATION_GROUPS.values() for c in group]
    else:
        configs = ABLATION_GROUPS[args.group]

    print(f"Running {len(configs)} ablations on {args.model}")
    print(f"Device: {args.device}, Steps: {args.steps}, Batch: {args.batch}")

    if args.dry_run:
        print("\nDRY RUN - Configurations to test:")
        for i, cfg in enumerate(configs, 1):
            print(f"\n{i}. {cfg.name}")
            print(f"   {cfg.description}")
            print(f"   Recipe: {cfg.recipe_fn}({cfg.recipe_kwargs})")
        return

    # Run ablations
    results = []
    for cfg in configs:
        try:
            result = train_one_ablation(cfg, args)
            results.append(result)
        except Exception as e:
            print(f"\nERROR in {cfg.name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Save and display results
    if results:
        out_path = os.path.join(args.out_dir, f"ablation_{args.group}_results.json")
        save_results(results, out_path)
        print_summary_table(results)


if __name__ == "__main__":
    main()
