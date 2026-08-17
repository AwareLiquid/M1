"""Faithful needle-in-a-haystack re-test for the deployed M1 adapter.

WHY THIS EXISTS: the old needle numbers on the site (0% across all cells) came
from bench_llama_mt_needle.py, which concatenates raw tokens WITHOUT applying the
instruct chat template -- a format docs/reviews/NEEDLE_FIX.md already identified as producing
0.0 on instruct-tuned bases. bench_needle_chat_template.py fixes the formatting,
but its adapter loader (attach_adapters_from_checkpoint) rebuilds ONLY the MT
adapters and never re-wraps PEFT LoRA, so a phase5b checkpoint (MT + LoRA) is
silently dropped -- testing the wrong model.

This harness rebuilds the model the SAME way serve/server_hf.py does for the live
/adapter route (attach_mt_adapters -> get_peft_model(LoRA) -> load_state_dict),
with the same HONEST GUARD: if any checkpoint tensor fails to map onto the graph,
we abort rather than report numbers for a model that isn't really the adapter.
Prompts use the chat template (the correct way to query an instruct model).

Run:
    python bench_needle_m1_faithful.py \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --adapter checkpoints/llama_mt_adapter/llama_mt_adapter_003000.pt \
        --context_lengths 1024 2048 4096 --depths 0.1 0.5 0.9 --samples 5 \
        --out_json benchmarks/needle_m1_chat_template.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from dataclasses import asdict
from types import SimpleNamespace

import torch

# Reuse the corrected (chat-template) prompt builder + eval loop verbatim.
from bench_needle_chat_template import (  # noqa: E402
    NeedleResult,
    print_table,
    run_one_setting,
)


def load_base(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    return model, tok


def load_m1_adapter(model_name, adapter_path):
    """Rebuild EXACTLY as serve/server_hf.py does, then load + honest-guard."""
    from mt_lnn.llama_adapter import attach_mt_adapters, count_trainable_parameters

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

    ck = torch.load(adapter_path, map_location="cpu", weights_only=False)
    cargs = ck.get("args", {}) or {}
    sd = ck.get("state_dict", {})

    # Match the predictive-coding graph if the checkpoint carries W_pred tensors.
    want_pc = any("W_pred" in k for k in sd)

    wrapped = attach_mt_adapters(
        model,
        every=int(cargs.get("mt_every", 4)),
        n_protofilaments=int(cargs.get("mt_proto", 13)),
        n_time_scales=int(cargs.get("mt_scales", 5)),
        map_hidden_dim=int(cargs.get("mt_map_hidden", 64)),
        dropout=float(cargs.get("mt_dropout", 0.0)),
        init_scale=float(cargs.get("mt_init_scale", 1e-3)),
        use_scan=not bool(cargs.get("mt_no_scan", False)),
        use_predictive_coding=want_pc,
    )
    print(f"[needle] attached MT adapters to layers {wrapped}")

    want_lora = bool(cargs.get("lora", False))
    if want_lora:
        from peft import LoraConfig, get_peft_model

        targets = cargs.get("lora_targets", "q_proj,k_proj,v_proj,o_proj")
        if isinstance(targets, str):
            targets = targets.split(",")
        lcfg = LoraConfig(
            r=int(cargs.get("lora_r", 8)),
            lora_alpha=int(cargs.get("lora_alpha", 16)),
            lora_dropout=float(cargs.get("lora_dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        )
        model = get_peft_model(model, lcfg)
        print(f"[needle] applied PEFT LoRA (r={lcfg.r}, alpha={lcfg.lora_alpha}, targets={targets})")

    trainable = count_trainable_parameters(model)
    total = sum(p.numel() for p in model.parameters())
    print(f"[needle] {trainable:,} adapter params / {total:,} total ({100*trainable/total:.3f}%)")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    adapter_keys = [k for k in sd if "mt_adapter" in k or "lora_" in k]
    matched = len(sd) - len(unexpected)
    print(f"[needle] loaded step {ck.get('step', '?')}: {matched}/{len(sd)} tensors "
          f"matched; missing(base)={len(missing)} unexpected={len(unexpected)}")

    # HONEST GUARD: refuse to report needle numbers for a model whose adapter
    # tensors did not actually map onto the graph (that would be the bare base).
    if not (len(adapter_keys) > 0 and len(unexpected) == 0):
        raise SystemExit(
            f"[needle] ABORT: adapter did NOT map cleanly (unexpected={len(unexpected)}). "
            f"This would test the bare base, not M1. Fix the recipe before reporting."
        )
    print(f"[needle] adapter ACTIVE: {len(adapter_keys)} MT/LoRA tensors loaded.")
    model.eval()
    return model, tok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--adapter", default="checkpoints/llama_mt_adapter/llama_mt_adapter_003000.pt")
    p.add_argument("--context_lengths", nargs="+", type=int, default=[1024, 2048, 4096])
    p.add_argument("--depths", nargs="+", type=float, default=[0.1, 0.5, 0.9])
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--max_new_tokens", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip_base", action="store_true")
    p.add_argument("--out_json", default="benchmarks/needle_m1_chat_template.json")
    a = p.parse_args()

    device = "cpu"
    eval_args = SimpleNamespace(samples=a.samples, max_new_tokens=a.max_new_tokens)
    results = []

    def sweep(model, tok, variant, adapter_path):
        for cl in a.context_lengths:
            for d in a.depths:
                random.seed(a.seed)  # same needle codes across variants -> comparable
                print(f"[needle] {variant}: context={cl}, depth={d:.2f}, samples={a.samples}", flush=True)
                results.append(run_one_setting(model, tok, device, variant, adapter_path, eval_args, cl, d))

    if not a.skip_base:
        print("[needle] loading BASE (no adapter)...", flush=True)
        m, tok = load_base(a.model)
        try:
            sweep(m, tok, "base", None)
        finally:
            del m; gc.collect()

    print(f"[needle] loading M1 (base + {os.path.basename(a.adapter)})...", flush=True)
    m, tok = load_m1_adapter(a.model, a.adapter)
    try:
        sweep(m, tok, f"mt+lora:{os.path.basename(a.adapter)}", a.adapter)
    finally:
        del m; gc.collect()

    print_table(results)
    os.makedirs(os.path.dirname(a.out_json) or ".", exist_ok=True)
    with open(a.out_json, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\n[needle] saved JSON: {a.out_json}")


if __name__ == "__main__":
    main()
