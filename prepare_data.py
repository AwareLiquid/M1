"""
prepare_data.py — pre-tokenise a HuggingFace text dataset to a flat .bin file
on disk, so training can use numpy.memmap and avoid loading everything to RAM.

Output:
    data/{split}.bin   uint16 token stream (each token < 65535)
    data/meta.json     {vocab_size, n_train_tokens, n_val_tokens, tokenizer}

Usage:
    python prepare_data.py                            # WikiText-103 default
    python prepare_data.py --dataset wikitext --config wikitext-2-raw-v1
    # Config-less datasets (e.g. TinyStories) + a per-split token cap:
    python prepare_data.py --dataset roneneldan/TinyStories --config none \
        --max_tokens 30000000 --out_dir data_tiny
"""

import argparse
import json
import os

import numpy as np
from tqdm import tqdm


def main(args):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    # Config-less datasets (TinyStories, etc.) pass --config none / "" -> None.
    cfg = None if args.config in ("", "none", "None", "null") else args.config
    print(f"Loading {args.dataset}/{cfg} …")
    ds = load_dataset(args.dataset, cfg)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    assert tok.vocab_size < 65535, "Use uint32 if vocab_size > 65535"

    os.makedirs(args.out_dir, exist_ok=True)
    meta = {"tokenizer": args.tokenizer, "vocab_size": tok.vocab_size}

    # Per-split token cap (0 = unlimited). Lets a huge corpus (e.g. TinyStories'
    # ~471M-token train split) be truncated to just what an experiment needs,
    # turning a ~30 min tokenisation into ~1-2 min. Backward compatible: the
    # default 0 reproduces the original full-corpus behaviour exactly.
    max_tokens = max(0, int(args.max_tokens))

    for split in ("train", "validation", "test"):
        if split not in ds:
            continue
        out_path = os.path.join(args.out_dir, f"{split}.bin")
        n_tokens = 0
        with open(out_path, "wb") as f:
            for item in tqdm(ds[split], desc=f"tokenising {split}"):
                text = item["text"]
                if not text:
                    continue
                ids = tok.encode(text)
                if not ids:
                    continue
                arr = np.asarray(ids, dtype=np.uint16)
                f.write(arr.tobytes())
                n_tokens += len(arr)
                if max_tokens and n_tokens >= max_tokens:
                    break
        print(f"  -> {out_path}: {n_tokens:,} tokens"
              + (f" (capped at {max_tokens:,})" if max_tokens else ""))
        meta[f"n_{split}_tokens"] = n_tokens

    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Done. Meta: {os.path.join(args.out_dir, 'meta.json')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",  default="wikitext")
    p.add_argument("--config",   default="wikitext-103-raw-v1")
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--out_dir",  default="data")
    p.add_argument("--max_tokens", type=int, default=0,
                   help="per-split token cap (0 = unlimited / full corpus)")
    main(p.parse_args())
