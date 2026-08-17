"""
Needle-in-a-haystack benchmark for base Llama vs MT adapter checkpoints.

This probes the claim MT adapters are meant to help with: a short fact appears
early in a long context, and the model must retrieve it near the end. The script
builds prompts at token level so context length and needle depth are controlled
across variants.

Examples:
    python bench_llama_mt_needle.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

    python bench_llama_mt_needle.py \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --adapters checkpoints/llama_mt_adapter/llama_mt_adapter_001000.pt \
        --context_lengths 1024 2048 4096 \
        --depths 0.1 0.5 0.9
"""

import argparse
import gc
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from typing import Optional

import torch

from bench_llama_mt_ablation import adapter_name
from demo_llama_mt_adapter import maybe_apply_lora_for_checkpoint
from mt_lnn.llama_adapter import attach_adapters_from_checkpoint, load_adapter_state


FILLER = (
    " The archive contains ordinary notes about schedules, weather, budgets, "
    "meeting rooms, project status, draft ideas, and routine observations."
)
QUESTION = (
    "\nQuestion: What is the secret passcode? "
    "Answer with only the digits.\nAnswer:"
)


@dataclass
class NeedleResult:
    variant: str
    adapter: Optional[str]
    context_len: int
    depth: float
    samples: int
    exact: int
    contains: int
    accuracy: float
    contains_rate: float
    seconds: float
    tok_per_sec: float


def load_variant(args, adapter_path=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.model)
    if not hasattr(config, "rope_theta") or config.rope_theta is None:
        config.rope_theta = 10000.0
    config.rope_scaling = {"type": "linear", "rope_type": "linear", "factor": 4.0}

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=dtype if device == "cuda" else torch.float32,
        device_map=None,
    )

    if adapter_path is not None:
        checkpoint = torch.load(adapter_path, map_location="cpu")
        attach_adapters_from_checkpoint(model, checkpoint)
        model = maybe_apply_lora_for_checkpoint(model, checkpoint)
        info = load_adapter_state(model, adapter_path, strict=False)
        unexpected = [k for k in info["unexpected"] if "mt_adapter" in k or "lora_" in k]
        if unexpected:
            raise RuntimeError(f"Unexpected adapter keys in {adapter_path}: {unexpected[:8]}")

    model.to(device).eval()
    return model, tokenizer, device


def repeat_to_length(ids, length):
    if length <= 0:
        return []
    reps = (length + len(ids) - 1) // len(ids)
    return (ids * reps)[:length]


def make_prompt_ids(tokenizer, context_len, depth, code):
    needle = f"\nImportant memory: the secret passcode is {code}. Remember this exact passcode.\n"
    filler_ids = tokenizer.encode(FILLER, add_special_tokens=False)
    needle_ids = tokenizer.encode(needle, add_special_tokens=False)
    question_ids = tokenizer.encode(QUESTION, add_special_tokens=False)
    fixed = len(needle_ids) + len(question_ids)
    if fixed >= context_len:
        raise ValueError(
            f"context_len={context_len} is too short for needle+question ({fixed} tokens)"
        )

    available = context_len - fixed
    before_len = int(available * depth)
    after_len = available - before_len
    ids = (
        repeat_to_length(filler_ids, before_len)
        + needle_ids
        + repeat_to_length(filler_ids, after_len)
        + question_ids
    )
    return torch.tensor([ids], dtype=torch.long)


def normalize_digits(text):
    return "".join(re.findall(r"\d", text))


@torch.no_grad()
def run_one_setting(model, tokenizer, device, variant, adapter_path, args, context_len, depth):
    exact, contains, total_tokens = 0, 0, 0
    start = time.time()
    for _ in range(args.samples):
        code = f"{random.randint(0, 999999):06d}"
        prompt_ids = make_prompt_ids(tokenizer, context_len, depth, code).to(device)
        attention_mask = torch.ones_like(prompt_ids)
        total_tokens += prompt_ids.numel()
        out = model.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        gen_ids = out[:, prompt_ids.shape[1]:]
        total_tokens += gen_ids.numel()
        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        digits = normalize_digits(text)
        if digits.startswith(code):
            exact += 1
        if code in digits:
            contains += 1

    seconds = max(time.time() - start, 1e-6)
    return NeedleResult(
        variant=variant,
        adapter=adapter_path,
        context_len=context_len,
        depth=depth,
        samples=args.samples,
        exact=exact,
        contains=contains,
        accuracy=exact / max(args.samples, 1),
        contains_rate=contains / max(args.samples, 1),
        seconds=seconds,
        tok_per_sec=total_tokens / seconds,
    )


def print_table(results):
    print()
    print("| Variant | Context | Depth | Exact | Contains | Tok/s | Seconds |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        print(
            f"| {r.variant} | {r.context_len} | {r.depth:.2f} | "
            f"{r.accuracy:.3f} | {r.contains_rate:.3f} | "
            f"{r.tok_per_sec:.0f} | {r.seconds:.1f} |"
        )


def release_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(args):
    random.seed(args.seed)
    variants = []
    if not args.skip_base:
        variants.append(None)
    variants.extend(args.adapters or [])

    results = []
    for adapter_path in variants:
        variant = adapter_name(adapter_path)
        print(f"Loading {variant}...")
        model, tokenizer, device = load_variant(args, adapter_path)
        try:
            for context_len in args.context_lengths:
                for depth in args.depths:
                    print(
                        f"Evaluating {variant}: context={context_len}, "
                        f"depth={depth:.2f}, samples={args.samples}"
                    )
                    results.append(
                        run_one_setting(
                            model, tokenizer, device, variant, adapter_path,
                            args, context_len, depth,
                        )
                    )
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
    p.add_argument("--context_lengths", nargs="+", type=int, default=[1024, 2048])
    p.add_argument("--depths", nargs="+", type=float, default=[0.1, 0.5, 0.9])
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--max_new_tokens", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_json", default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
