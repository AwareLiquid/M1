"""
Run a simple base-vs-adapter ablation for HuggingFace LMs.

The benchmark uses the same tokenized validation windows for every variant and
reports perplexity plus throughput. Adapter checkpoints may be MT-only or
LoRA+MT; the loader reconstructs the layout from the saved training args.

Examples:
    python bench_llama_mt_ablation.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

    python bench_llama_mt_ablation.py \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --adapters checkpoints/llama_mt_adapter/llama_mt_adapter_001000.pt
"""

import argparse
import gc
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader

from demo_llama_mt_adapter import maybe_apply_lora_for_checkpoint
from mt_lnn.llama_adapter import attach_adapters_from_checkpoint, load_adapter_state


@dataclass
class BenchResult:
    name: str
    adapter: Optional[str]
    tokens: int
    batches: int
    ppl: float
    seconds: float
    tok_per_sec: float
    trainable_params: int
    total_params: int


def build_loader(tokenizer, args):
    from datasets import load_dataset

    ds = load_dataset(args.dataset, args.dataset_config, split=args.split)

    def tokenize(batch):
        text = [t for t in batch[args.text_column] if t]
        if not text:
            return {"input_ids": []}
        return tokenizer(text, add_special_tokens=False)

    tokenized = ds.map(
        tokenize,
        batched=True,
        remove_columns=ds.column_names,
        desc="tokenizing benchmark",
    )

    def group_texts(examples):
        ids = []
        eos = tokenizer.eos_token_id
        for row in examples["input_ids"]:
            ids.extend(row + ([eos] if eos is not None else []))
        total = (len(ids) // args.seq_len) * args.seq_len
        ids = ids[:total]
        chunks = [ids[i : i + args.seq_len] for i in range(0, total, args.seq_len)]
        return {"input_ids": chunks, "labels": [c.copy() for c in chunks]}

    lm_ds = tokenized.map(
        group_texts,
        batched=True,
        remove_columns=tokenized.column_names,
        desc="chunking benchmark",
    )
    if args.max_batches > 0:
        max_rows = min(len(lm_ds), args.max_batches * args.batch)
        lm_ds = lm_ds.select(range(max_rows))
    lm_ds.set_format(type="torch", columns=["input_ids", "labels"])
    return DataLoader(lm_ds, batch_size=args.batch, shuffle=False)


def load_variant(args, adapter_path=None):
    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype if device == "cuda" else torch.float32,
        device_map=None,
    )
    model.config.use_cache = False

    if adapter_path is not None:
        checkpoint = torch.load(adapter_path, map_location="cpu")
        attach_adapters_from_checkpoint(model, checkpoint)
        model = maybe_apply_lora_for_checkpoint(model, checkpoint)
        info = load_adapter_state(model, adapter_path, strict=False)
        unexpected = [k for k in info["unexpected"] if "mt_adapter" in k or "lora_" in k]
        if unexpected:
            raise RuntimeError(f"Unexpected adapter keys in {adapter_path}: {unexpected[:8]}")

    model.to(device).eval()
    return model, device


@torch.no_grad()
def evaluate_variant(model, loader, device, name, adapter_path=None):
    total_nll, n_tokens, n_batches = 0.0, 0, 0
    start = time.time()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        n = batch["labels"].numel()
        total_nll += out.loss.item() * n
        n_tokens += n
        n_batches += 1
    seconds = max(time.time() - start, 1e-6)
    ppl = math.exp(total_nll / max(n_tokens, 1))
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return BenchResult(
        name=name,
        adapter=adapter_path,
        tokens=n_tokens,
        batches=n_batches,
        ppl=ppl,
        seconds=seconds,
        tok_per_sec=n_tokens / seconds,
        trainable_params=trainable_params,
        total_params=total_params,
    )


def adapter_name(path):
    if path is None:
        return "base"
    try:
        ckpt = torch.load(path, map_location="cpu")
        args = ckpt.get("args", {})
        tags = ["mt"]
        if args.get("lora", False):
            tags.append("lora")
        return f"{'+'.join(tags)}:{os.path.basename(path)}"
    except Exception:
        return os.path.basename(path)


def print_table(results):
    print()
    print("| Variant | PPL | Tokens | Tok/s | Eval seconds | Trainable params |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in results:
        print(
            f"| {r.name} | {r.ppl:.3f} | {r.tokens:,} | "
            f"{r.tok_per_sec:.0f} | {r.seconds:.1f} | {r.trainable_params:,} |"
        )


def release_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    loader = build_loader(tokenizer, args)

    variants = []
    if not args.skip_base:
        variants.append(None)
    variants.extend(args.adapters or [])

    results = []
    for adapter_path in variants:
        name = adapter_name(adapter_path)
        print(f"Evaluating {name}...")
        model, device = load_variant(args, adapter_path)
        try:
            results.append(evaluate_variant(model, loader, device, name, adapter_path))
        finally:
            release_model(model)

    print_table(results)
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nSaved JSON: {args.out_json}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--adapters", nargs="*", default=[])
    p.add_argument("--skip_base", action="store_true")
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--dataset_config", default="wikitext-2-raw-v1")
    p.add_argument("--split", default="validation")
    p.add_argument("--text_column", default="text")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--max_batches", type=int, default=50)
    p.add_argument("--out_json", default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
