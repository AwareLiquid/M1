"""Needle-in-a-haystack benchmark with proper chat template formatting.

Fixes the issue where raw prompt format doesn't match instruct-tuning expectations,
causing all models to score 0.0. Uses tokenizer.apply_chat_template() to properly
format prompts for instruct-tuned models like Qwen-2.5-*-Instruct.

Usage:
    python bench_needle_chat_template.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --adapters benchmarks/kaggle_qwen_run/llama_mt_adapter_001000.pt \\
        --context_lengths 1024 2048 \\
        --depths 0.1 0.5 0.9 \\
        --samples 5 \\
        --out_json artifacts/needle_chat_results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass

import torch


@dataclass
class NeedleResult:
    variant: str
    adapter: str | None
    context_len: int
    depth: float
    samples: int
    exact: int
    contains: int
    accuracy: float
    contains_rate: float
    seconds: float
    tok_per_sec: float


# Filler text to pad the context
FILLER = (
    " The archive contains ordinary notes about various topics. "
    "Here are some random facts: cats sleep 16 hours a day, "
    "the Pacific Ocean is the largest ocean, "
    "Mount Everest is 8,849 meters tall, "
    "Tokyo is the capital of Japan, "
    "the speed of light is 299,792,458 m/s. "
)


def repeat_to_length(items, target_len):
    """Repeat a list of items to reach approximately target_len."""
    if not items:
        return []
    result = []
    while len(result) < target_len:
        result.extend(items)
    return result[:target_len]


def load_variant(args, adapter_path=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    if adapter_path:
        from mt_lnn.llama_adapter import attach_adapters_from_checkpoint, load_adapter_state

        ckpt = torch.load(adapter_path, map_location="cpu")
        attach_adapters_from_checkpoint(model, ckpt)
        load_adapter_state(model, adapter_path, strict=False)
        print(f"  loaded adapter: {adapter_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, tokenizer, device


def adapter_name(path):
    if path is None:
        return "base"
    return f"mt+{os.path.basename(path)}"


def make_prompt_with_chat_template(tokenizer, context_len, depth, code):
    """Construct needle prompt using proper chat template.

    The needle is embedded at `depth` fraction through the filler context,
    then followed by a question asking for the passcode.
    """
    needle = f"Important memory: the secret passcode is {code}. Remember this exact passcode."
    question = "Question: What is the secret passcode? Answer with only the digits."

    # Build the user message with needle embedded in filler
    # First, estimate how many tokens we need
    filler_ids = tokenizer.encode(FILLER, add_special_tokens=False)
    needle_ids = tokenizer.encode(needle, add_special_tokens=False)
    question_ids = tokenizer.encode(question, add_special_tokens=False)

    # Reserve space for chat template overhead (system, user, assistant tokens)
    # Empirically, chat template adds ~20-40 tokens depending on model
    template_overhead = 50
    fixed = len(needle_ids) + len(question_ids) + template_overhead

    if fixed >= context_len:
        raise ValueError(
            f"context_len={context_len} too short for needle+question+template ({fixed} tokens)"
        )

    available = context_len - fixed
    before_len = int(available * depth)
    after_len = available - before_len

    # Build filler text chunks
    filler_before = tokenizer.decode(repeat_to_length(filler_ids, before_len))
    filler_after = tokenizer.decode(repeat_to_length(filler_ids, after_len))

    # Construct user message with needle embedded
    user_message = filler_before + "\n\n" + needle + "\n\n" + filler_after + "\n\n" + question

    # Apply chat template
    messages = [{"role": "user", "content": user_message}]

    # Use tokenizer.apply_chat_template with tokenize=True
    # add_generation_prompt=True adds the assistant prompt at the end
    try:
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
    except Exception as e:
        # Fallback for models without chat template support
        print(f"  Warning: chat template failed ({e}), using raw format")
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_ids = tokenizer(text, return_tensors="pt").input_ids

    # Trim or pad to target context_len if needed
    if prompt_ids.shape[1] > context_len:
        prompt_ids = prompt_ids[:, :context_len]

    return prompt_ids


def normalize_digits(text):
    """Extract all digits from text."""
    return "".join(re.findall(r"\d", text))


@torch.no_grad()
def run_one_setting(model, tokenizer, device, variant, adapter_path, args, context_len, depth):
    exact, contains, total_tokens = 0, 0, 0
    start = time.time()

    for _ in range(args.samples):
        code = f"{random.randint(0, 999999):06d}"
        prompt_ids = make_prompt_with_chat_template(tokenizer, context_len, depth, code).to(device)
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
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
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
