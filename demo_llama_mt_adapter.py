"""
Generate text with a base HuggingFace causal LM plus optional MT-LNN adapter.

Examples:
    python demo_llama_mt_adapter.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "Hello"
    python demo_llama_mt_adapter.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --adapter checkpoints/llama_mt_adapter/llama_mt_adapter_001000.pt
"""

import argparse

import torch
import torch.nn.functional as F

from mt_lnn.llama_adapter import attach_adapters_from_checkpoint, load_adapter_state


def filter_top_k(logits, k):
    if k <= 0:
        return logits
    values, _ = torch.topk(logits, min(k, logits.shape[-1]))
    return logits.masked_fill(logits < values[:, [-1]], float("-inf"))


def filter_top_p(logits, p):
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    keep = probs.cumsum(dim=-1) <= p
    keep[..., 0] = True
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(-1, sorted_idx, keep)
    return logits.masked_fill(~mask, float("-inf"))


def maybe_apply_lora_for_checkpoint(model, checkpoint):
    saved_args = checkpoint.get("args", {})
    if not saved_args.get("lora", False):
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "This adapter checkpoint contains LoRA weights, but `peft` is not "
            "installed. Install with `pip install peft`."
        ) from exc

    config = LoraConfig(
        r=int(saved_args.get("lora_r", 8)),
        lora_alpha=int(saved_args.get("lora_alpha", 16)),
        lora_dropout=float(saved_args.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=str(saved_args.get("lora_targets", "q_proj,k_proj,v_proj,o_proj")).split(","),
    )
    return get_peft_model(model, config)


def load_model(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype if device == "cuda" else torch.float32,
        device_map=None,
    )

    if args.adapter:
        checkpoint = torch.load(args.adapter, map_location="cpu")
        attach_adapters_from_checkpoint(model, checkpoint)
        model = maybe_apply_lora_for_checkpoint(model, checkpoint)
        info = load_adapter_state(model, args.adapter, strict=False)
        unexpected = [k for k in info["unexpected"] if "mt_adapter" in k or "lora_" in k]
        if unexpected:
            raise RuntimeError(f"Unexpected adapter keys: {unexpected[:8]}")
        print(f"Loaded adapter checkpoint: {args.adapter}")

    model.to(device).eval()
    return model, tokenizer, device


@torch.no_grad()
def generate(args):
    model, tokenizer, device = load_model(args)
    ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    print(tokenizer.decode(ids[0], skip_special_tokens=True), end="", flush=True)

    for _ in range(args.max_tokens):
        out = model(input_ids=ids)
        logits = out.logits[:, -1, :] / max(args.temperature, 1e-6)
        logits = filter_top_k(logits, args.top_k)
        logits = filter_top_p(logits, args.top_p)
        next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        print(tokenizer.decode(next_id[0], skip_special_tokens=True), end="", flush=True)
        ids = torch.cat([ids, next_id], dim=1)
        if tokenizer.eos_token_id is not None and next_id.item() == tokenizer.eos_token_id:
            break
    print()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--adapter", default=None)
    p.add_argument("--prompt", default="The most useful small language model is")
    p.add_argument("--max_tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--top_p", type=float, default=0.9)
    return p.parse_args()


if __name__ == "__main__":
    generate(parse_args())
