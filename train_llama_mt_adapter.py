"""
Train MT-LNN residual adapters on top of a frozen HuggingFace causal LM.

Example:
    python train_llama_mt_adapter.py \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --dataset wikitext --dataset_config wikitext-2-raw-v1 \
        --steps 200 --batch 1 --seq_len 512
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from mt_lnn.llama_adapter import (
    attach_mt_adapters,
    count_trainable_parameters,
    iter_mt_adapter_parameters,
)


def maybe_apply_lora(model, args):
    if not args.lora:
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "LoRA requested but `peft` is not installed. Install with "
            "`pip install peft` or run without --lora."
        ) from exc

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.lora_targets.split(","),
    )
    return get_peft_model(model, config)


def build_dataloader(tokenizer, args):
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
        desc="tokenizing",
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
        desc="chunking",
    )
    lm_ds.set_format(type="torch", columns=["input_ids", "labels"])
    return DataLoader(lm_ds, batch_size=args.batch, shuffle=True, drop_last=True)


def build_sft_dataloader(tokenizer, args):
    """Supervised fine-tuning loader for instruction data.

    Unlike the plain-LM loader above (which trains next-token prediction on a
    raw corpus like WikiText), this formats each (instruction, input, output)
    triple through the tokenizer's CHAT TEMPLATE and masks the loss so only the
    assistant completion (+EOS) contributes -- i.e. the model is taught to
    FOLLOW instructions, not just continue text. This is the post-training (SFT)
    stage the adapter previously skipped, which is why it could continue English
    facts but could not answer questions, reason, or respond in Chinese.

    --dataset may be a comma-separated list (e.g. an English + a Chinese Alpaca
    set) which are concatenated for a bilingual mix. All datasets are assumed to
    share the Alpaca schema (instruction / input / output columns, configurable).
    """
    from datasets import concatenate_datasets, load_dataset

    names = [n.strip() for n in args.dataset.split(",") if n.strip()]
    configs = [c.strip() or None for c in (args.dataset_config or "").split(",")]
    if len(configs) < len(names):
        configs += [None] * (len(names) - len(configs))

    parts = []
    for name, config in zip(names, configs):
        d = load_dataset(name, config, split=args.split) if config else load_dataset(name, split=args.split)
        cols = d.column_names

        def _normalize(ex, _cols=cols):
            return {
                "instruction": str(ex.get(args.instr_column, "") or "") if args.instr_column in _cols else "",
                "input": str(ex.get(args.input_column, "") or "") if args.input_column in _cols else "",
                "output": str(ex.get(args.output_column, "") or "") if args.output_column in _cols else "",
            }

        # Normalize every source to a fixed {instruction,input,output} schema so
        # datasets with different columns (e.g. a ZH set lacking `input`) still
        # concatenate cleanly.
        parts.append(d.map(_normalize, remove_columns=cols, desc=f"normalize:{name}"))
    ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)

    sys_prompt = args.system_prompt or None
    seq_len = args.seq_len
    eos_id = tokenizer.eos_token_id

    def encode(example):
        # Columns are normalized to fixed keys above.
        instr = (example.get("instruction") or "").strip()
        extra = (example.get("input") or "").strip()
        out = (example.get("output") or "").strip()
        if not instr or not out:
            return {"input_ids": [], "labels": []}
        user = instr if not extra else f"{instr}\n\n{extra}"
        msgs = []
        if sys_prompt:
            msgs.append({"role": "system", "content": sys_prompt})
        msgs.append({"role": "user", "content": user})
        # Render to strings then tokenize -- apply_chat_template(tokenize=True)
        # can return a BatchEncoding on some transformers versions, so we go via
        # text to guarantee plain int lists. prompt_text is a prefix of full_text
        # (same template, assistant turn appended), so the id prefix aligns and
        # we can mask exactly the prompt span.
        prompt_text = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )
        full_msgs = msgs + [{"role": "assistant", "content": out}]
        full_text = tokenizer.apply_chat_template(
            full_msgs, add_generation_prompt=False, tokenize=False
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        # End the target exactly at the assistant turn's closing EOS. Chat
        # templates often append a trailing separator (e.g. "</s>\n") after the
        # assistant content; without trimming, the last real token is the
        # newline and a naive `full_ids[-1] != eos` check appends a SECOND EOS,
        # training the model on a spurious "\n</s>". Truncating at the last EOS
        # yields a clean single-EOS completion; only synthesise one if the
        # template emitted none at all.
        if eos_id in full_ids:
            last_eos = len(full_ids) - 1 - full_ids[::-1].index(eos_id)
            full_ids = full_ids[: last_eos + 1]
        else:
            full_ids = full_ids + [eos_id]
        # Mask the prompt; train only on the completion + EOS.
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        full_ids = full_ids[:seq_len]
        labels = labels[:seq_len]
        return {"input_ids": full_ids, "labels": labels}

    enc = ds.map(encode, remove_columns=ds.column_names, desc="formatting-sft")
    enc = enc.filter(lambda e: len(e["input_ids"]) > 0)

    pad_id = tokenizer.pad_token_id

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            ids, lab = b["input_ids"], b["labels"]
            pad = maxlen - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

    return DataLoader(enc, batch_size=args.batch, shuffle=True, drop_last=True, collate_fn=collate)


def save_adapter_checkpoint(model, args, step):
    os.makedirs(args.out_dir, exist_ok=True)
    payload = {
        "step": step,
        "model": args.model,
        "state_dict": {
            k: v.cpu()
            for k, v in model.state_dict().items()
            if "mt_adapter" in k or "lora_" in k
        },
        "args": vars(args),
    }
    path = os.path.join(args.out_dir, f"llama_mt_adapter_{step:06d}.pt")
    torch.save(payload, path)
    print(f"saved {path}")


def train(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import random
    import numpy as np
    seed = getattr(args, "seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    if getattr(args, "no_mt", False):
        # 纯 LoRA 对照臂（无 MT adapter）——隔离 MT adapter 的贡献
        wrapped = []
    elif getattr(args, "adapter", "v1") == "v2":
        from mt_lnn.mt_lnn_v2 import attach_mt_v2_adapters
        wrapped = attach_mt_v2_adapters(
            model,
            every=args.mt_every,
            n_protofilaments=args.mt_proto,
            d_proto=args.v2_d_proto,
            n_time_scales=args.mt_scales,
            proj_rank=args.v2_rank,
            init_scale=args.mt_init_scale,
            dropout=args.mt_dropout,
            selective_decay=args.v2_selective,
            selective_decay_mode=args.sel_mode,
            use_fast_weight=not args.v2_no_fw,
            fast_weight_dim=args.v2_fw_dim,
            fast_weight_heads=args.v2_fw_heads,
        )
    else:
        wrapped = attach_mt_adapters(
            model,
            every=args.mt_every,
            n_protofilaments=args.mt_proto,
            n_time_scales=args.mt_scales,
            map_hidden_dim=args.mt_map_hidden,
            dropout=args.mt_dropout,
            init_scale=args.mt_init_scale,
            use_scan=not args.mt_no_scan,
        )
    model = maybe_apply_lora(model, args)

    # PEFT's get_peft_model() freezes every non-LoRA parameter (no
    # modules_to_save is set), which silently freezes the MT adapter's own
    # weights -- including the residual gate `scale`. With scale stuck at its
    # init (1e-3) the adapter only ever contributes 0.1% to the stream and can
    # never grow: the MT module becomes decorative and only LoRA trains. Re-arm
    # the MT adapter's parameters so the gate (and the rest of the adapter) can
    # actually learn. No-op when --lora is off (they were already trainable).
    from mt_lnn.llama_adapter import _iter_all_adapters
    n_rearmed = 0
    for adapter in _iter_all_adapters(model):     # covers v1 AND v2 adapters
        for p in adapter.parameters():
            if not p.requires_grad:
                p.requires_grad = True
            n_rearmed += p.numel()
    print(f"Re-armed MT adapter params (requires_grad=True): {n_rearmed:,}")

    model.to(device)
    model.train()

    trainable = count_trainable_parameters(model)
    total = sum(p.numel() for p in model.parameters())
    print(f"Wrapped decoder layers: {wrapped}")
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    if args.task == "sft":
        print(f"[train] SFT mode: instruction tuning with chat template + "
              f"completion-only loss on dataset(s): {args.dataset}")
        loader = build_sft_dataloader(tokenizer, args)
    else:
        print(f"[train] LM mode: plain causal-LM on {args.dataset}/{args.dataset_config}")
        loader = build_dataloader(tokenizer, args)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    use_amp = device == "cuda"
    # GradScaler does not support BFloat16 in some PyTorch versions
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if dtype != torch.bfloat16 else None
    step = 0
    t0 = time.time()
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype):
                out = model(**batch)
                loss = out.loss / args.grad_accum
                
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % args.grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip,
                )
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            if step % args.log_every == 0:
                elapsed = max(time.time() - t0, 1e-3)
                toks = args.log_every * args.batch * args.seq_len
                print(
                    f"step {step:6d} | loss {loss.item() * args.grad_accum:.4f} | "
                    f"{toks / elapsed:.0f} tok/s"
                )
                t0 = time.time()
            if step % args.save_every == 0:
                save_adapter_checkpoint(model, args, step)

    save_adapter_checkpoint(model, args, step)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--task", choices=["lm", "sft"], default="lm",
                   help="lm = plain causal-LM on a corpus; sft = instruction "
                        "tuning (chat template + completion-only loss)")
    p.add_argument("--seed", type=int, default=0,
                   help="训练随机种子（多种子决策门实验用）")
    p.add_argument("--dataset", default="wikitext",
                   help="dataset name, or comma-separated list for an SFT mix")
    p.add_argument("--dataset_config", default="wikitext-2-raw-v1",
                   help="config name(s); comma-separated to match --dataset list")
    p.add_argument("--split", default="train")
    p.add_argument("--text_column", default="text")
    # SFT (instruction) columns -- Alpaca schema by default.
    p.add_argument("--instr_column", default="instruction")
    p.add_argument("--input_column", default="input")
    p.add_argument("--output_column", default="output")
    p.add_argument("--system_prompt", default="",
                   help="optional system prompt prepended to every SFT example")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--out_dir", default="checkpoints/llama_mt_adapter")

    p.add_argument("--mt_every", type=int, default=4)
    p.add_argument("--mt_proto", type=int, default=13)
    p.add_argument("--mt_scales", type=int, default=5)
    p.add_argument("--mt_map_hidden", type=int, default=64)
    p.add_argument("--mt_dropout", type=float, default=0.0)
    p.add_argument("--mt_init_scale", type=float, default=1e-3)
    p.add_argument("--mt_no_scan", action="store_true")

    # Adapter generation. v2 (mt_lnn.mt_lnn_v2) = bottleneck factorized
    # projections + diagonal per-scale maps + fast-weight memory: ~8.4M
    # trainable vs v1's 62.8M on TinyLlama. The choice is recorded in the
    # checkpoint's args so serve/server_hf.py rebuilds the matching graph.
    p.add_argument("--adapter", choices=["v1", "v2"], default="v1")
    p.add_argument("--no_mt", action="store_true",
                   help="纯 LoRA 对照（不挂任何 MT adapter）")
    p.add_argument("--v2_d_proto", type=int, default=64)
    p.add_argument("--v2_rank", type=int, default=128)
    p.add_argument("--v2_selective", action="store_true",
                   help="v2: input-dependent (selective) decay")
    p.add_argument("--sel_mode", choices=["mamba", "exp"], default="mamba",
                   help="v2 selective transition parameterisation (E5e): "
                        "exp = 2*exp(-dt/tau)-1, signed ±1 (length extrapolation)")
    p.add_argument("--v2_no_fw", action="store_true",
                   help="v2: disable the fast-weight memory")
    p.add_argument("--v2_fw_dim", type=int, default=64,
                   help="v2: fast-weight memory width (d_mem)")
    p.add_argument("--v2_fw_heads", type=int, default=1)

    p.add_argument("--lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_targets", default="q_proj,k_proj,v_proj,o_proj")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
