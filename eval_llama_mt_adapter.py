"""
Evaluate perplexity for a base HuggingFace LM with optional MT-LNN adapter.

Examples:
    python eval_llama_mt_adapter.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python eval_llama_mt_adapter.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --adapter checkpoints/llama_mt_adapter/llama_mt_adapter_001000.pt
"""

import argparse
import math

import torch
from torch.utils.data import DataLoader

from demo_llama_mt_adapter import load_model


def build_eval_loader(tokenizer, args):
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
        desc="tokenizing eval",
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
        group_texts,
        batched=True,
        remove_columns=tokenized.column_names,
        desc="chunking eval",
    )
    if args.max_batches > 0:
        max_rows = min(len(lm_ds), args.max_batches * args.batch)
        lm_ds = lm_ds.select(range(max_rows))
    lm_ds.set_format(type="torch", columns=["input_ids", "labels"])
    return DataLoader(lm_ds, batch_size=args.batch, shuffle=False)


@torch.no_grad()
def evaluate(args):
    model, tokenizer, device = load_model(args)
    loader = build_eval_loader(tokenizer, args)
    total_nll, n_tokens = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        n = batch["labels"].numel()
        total_nll += out.loss.item() * n
        n_tokens += n
    ppl = math.exp(total_nll / max(n_tokens, 1))
    label = "base" if args.adapter is None else args.adapter
    print(f"{label} | tokens {n_tokens:,} | ppl {ppl:.3f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--adapter", default=None)
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--dataset_config", default="wikitext-2-raw-v1")
    p.add_argument("--split", default="validation")
    p.add_argument("--text_column", default="text")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--max_batches", type=int, default=50)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
