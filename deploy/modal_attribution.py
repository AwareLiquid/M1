"""Attribution ablation: **LoRA-only vs MT-only vs MT+LoRA** at a matched budget.

This is roadmap item #1 -- the foundation for every later claim about what the
MT-LNN adapter actually contributes. It answers one question honestly:

    On the SAME data, SAME step budget, SAME optimizer, does the MT residual
    adapter improve perplexity beyond what plain LoRA already gives -- and is
    that improvement worth its trainable parameters?

METHOD (everything held identical except the adapter under test)
---------------------------------------------------------------
  * Base: frozen TinyLlama-1.1B-Chat (bf16, grad-checkpointed, use_cache=False).
  * Data: WikiText-2-raw-v1, tokenized + chunked ONCE into fixed seq_len windows
    so all configs literally see the same token tensors, in the same order.
  * Train split -> training. Test split -> held-out perplexity (never trained on).
  * Optimizer: AdamW, identical lr / wd / grad-clip / grad-accum / steps.
  * Configs:
      - baseline : no adapter, no training. Reference PPL for % improvement.
      - lora_only: LoRA r=8 on q,k,v,o  (peft freezes the base).
      - mt_only  : MT residual adapters every 4th layer (13 proto x 5 scales).
      - mt_lora  : both, with the MT params RE-ARMED after peft freezes them
                   (this is exactly what train_llama_mt_adapter.py ships -- the
                   003000 checkpoint's recipe).

Reports a single table: trainable params, final train loss, test PPL, % vs
baseline, and PPL-improvement-per-million-trainable-params (the matched-budget
efficiency metric the review asked for).

RUN
---
    modal run deploy/modal_attribution.py                 # defaults: 1000 steps
    modal run deploy/modal_attribution.py --steps 400     # cheaper smoke run
    modal run deploy/modal_attribution.py --configs baseline,lora_only,mt_only,mt_lora

Cost: one T4, ~20-30 min for the full 4-way 1000-step sweep (<$0.50). The
TinyLlama base is reused from the awareliquid-m1-hf Volume, so no re-download.
"""
from __future__ import annotations

import modal

APP_NAME = "awareliquid-attribution"
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
HF_DIR = "/hf"
OUT_DIR = "/out"

app = modal.App(APP_NAME)

# Reuse the M1 deployment's HF cache Volume so TinyLlama's ~2.2 GB is already
# present (no cold-start download). Falls back to downloading if absent.
hf_vol = modal.Volume.from_name("awareliquid-m1-hf", create_if_missing=True)
out_vol = modal.Volume.from_name(f"{APP_NAME}-out", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "numpy<2",
        "transformers>=4.44,<5",
        "tokenizers>=0.19",
        "peft>=0.11",
        "accelerate>=0.30",
        "datasets>=2.19,<3",
    )
    .env({"HF_HOME": HF_DIR, "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("mt_lnn", "/root/mt_lnn")
)


@app.function(
    image=image,
    gpu="T4",
    volumes={HF_DIR: hf_vol, OUT_DIR: out_vol},
    timeout=4 * 3600,
)
def run_attribution(
    steps: int = 1000,
    seq_len: int = 512,
    batch: int = 1,
    grad_accum: int = 8,
    lr: float = 2e-4,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    lora_r: int = 8,
    lora_alpha: int = 16,
    log_every: int = 100,
    max_eval_chunks: int = 0,   # 0 = use the whole test split
    seed: int = 0,
    configs: str = "baseline,lora_only,mt_only,mt_lora,mt_v2_only,mt_v2_lora",
):
    import json
    import math
    import sys
    import time

    import torch

    sys.path.insert(0, "/root")
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mt_lnn.llama_adapter import (
        attach_mt_adapters,
        count_trainable_parameters,
        iter_mt_adapter_parameters,
    )

    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    torch.manual_seed(seed)

    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- tokenize + chunk BOTH splits once; every config sees identical tensors.
    def build_chunks(split: str) -> torch.Tensor:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        ids: list[int] = []
        for t in ds["text"]:
            if t:
                ids.extend(tok(t, add_special_tokens=False)["input_ids"])
                ids.append(tok.eos_token_id)
        n = (len(ids) // seq_len) * seq_len
        chunks = [ids[i : i + seq_len] for i in range(0, n, seq_len)]
        return torch.tensor(chunks, dtype=torch.long)

    print("Tokenizing WikiText-2 (train + test) ...", flush=True)
    train_chunks = build_chunks("train")
    test_chunks = build_chunks("test")
    if max_eval_chunks and max_eval_chunks < len(test_chunks):
        test_chunks = test_chunks[:max_eval_chunks]
    print(
        f"train chunks: {len(train_chunks)}  test chunks: {len(test_chunks)}  "
        f"(seq_len={seq_len})",
        flush=True,
    )

    # Fixed shuffle order for training, identical across configs.
    g = torch.Generator().manual_seed(seed)
    train_order = torch.randperm(len(train_chunks), generator=g)

    def make_base():
        m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype)
        m.config.use_cache = False
        m.gradient_checkpointing_enable()
        # With a FROZEN base + gradient checkpointing, the checkpointed segments
        # see inputs that don't require grad, so autograd builds no graph and the
        # trainable adapter gets None gradients (silent no-learn). PEFT does this
        # for LoRA automatically; the pure mt_only path needs it explicitly.
        m.enable_input_require_grads()
        return m

    def setup(cfg: str, m):
        """Attach the adapter under test; return (model, n_rearmed)."""
        if cfg == "baseline":
            for p in m.parameters():
                p.requires_grad = False
            return m, 0
        if cfg == "mt_only":
            attach_mt_adapters(
                m, every=4, n_protofilaments=13, n_time_scales=5,
                map_hidden_dim=64, dropout=0.0, init_scale=1e-3, use_scan=True,
            )
            return m, 0
        if cfg == "lora_only":
            from peft import LoraConfig, get_peft_model
            m = get_peft_model(m, LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            ))
            return m, 0
        if cfg == "mt_v2_only":
            from mt_lnn.mt_lnn_v2 import attach_mt_v2_adapters
            attach_mt_v2_adapters(m, every=4)     # defaults: P13/d64/S5/r128/FW64
            return m, 0
        if cfg == "mt_v2_lora":
            from mt_lnn.mt_lnn_v2 import attach_mt_v2_adapters, iter_mt_v2_adapter_parameters
            attach_mt_v2_adapters(m, every=4)
            from peft import LoraConfig, get_peft_model
            m = get_peft_model(m, LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            ))
            n_re = 0
            for p in iter_mt_v2_adapter_parameters(m):
                if not p.requires_grad:
                    p.requires_grad = True
                n_re += p.numel()
            return m, n_re
        if cfg == "mt_lora":
            attach_mt_adapters(
                m, every=4, n_protofilaments=13, n_time_scales=5,
                map_hidden_dim=64, dropout=0.0, init_scale=1e-3, use_scan=True,
            )
            from peft import LoraConfig, get_peft_model
            m = get_peft_model(m, LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            ))
            # PEFT froze the MT adapter (no modules_to_save) -- re-arm it, exactly
            # as train_llama_mt_adapter.py does, so the gate can actually learn.
            n_re = 0
            for p in iter_mt_adapter_parameters(m):
                if not p.requires_grad:
                    p.requires_grad = True
                n_re += p.numel()
            return m, n_re
        raise ValueError(f"unknown config: {cfg}")

    @torch.no_grad()
    def eval_ppl(m) -> float:
        m.eval()
        total_nll = 0.0
        total_tok = 0
        for i in range(0, len(test_chunks), batch):
            ids = test_chunks[i : i + batch].to(device)
            with torch.amp.autocast("cuda", dtype=dtype):
                out = m(input_ids=ids, labels=ids)
            # HF returns mean loss over (seq_len-1)*batch shifted tokens.
            n = ids.shape[0] * (ids.shape[1] - 1)
            total_nll += out.loss.float().item() * n
            total_tok += n
        return math.exp(total_nll / total_tok)

    def train_one(cfg: str) -> dict:
        torch.manual_seed(seed)
        m = make_base()
        m, n_rearmed = setup(cfg, m)
        m.to(device)
        trainable = count_trainable_parameters(m)
        total = sum(p.numel() for p in m.parameters())
        print(
            f"\n=== {cfg} ===  trainable {trainable:,} / {total:,} "
            f"({100 * trainable / total:.3f}%)  re-armed MT {n_rearmed:,}",
            flush=True,
        )

        if cfg == "baseline" or trainable == 0:
            ppl = eval_ppl(m)
            print(f"[{cfg}] test PPL (no training): {ppl:.3f}", flush=True)
            del m
            torch.cuda.empty_cache()
            return {"config": cfg, "trainable": trainable, "total": total,
                    "final_loss": None, "test_ppl": ppl}

        m.train()
        opt = torch.optim.AdamW(
            [p for p in m.parameters() if p.requires_grad],
            lr=lr, weight_decay=weight_decay,
        )
        scaler = None  # bf16 needs no GradScaler
        step = 0
        last_loss = float("nan")
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        while step < steps:
            for idx in train_order:
                if step >= steps:
                    break
                i = int(idx)
                ids = train_chunks[i : i + 1].to(device)
                with torch.amp.autocast("cuda", dtype=dtype):
                    out = m(input_ids=ids, labels=ids)
                    loss = out.loss / grad_accum
                loss.backward()
                last_loss = out.loss.detach().float().item()
                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in m.parameters() if p.requires_grad], grad_clip)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                step += 1
                if step % log_every == 0:
                    dt = max(time.time() - t0, 1e-3)
                    print(f"[{cfg}] step {step:5d}/{steps} | loss {last_loss:.4f} "
                          f"| {log_every / dt:.2f} it/s", flush=True)
                    t0 = time.time()

        ppl = eval_ppl(m)
        print(f"[{cfg}] final train loss {last_loss:.4f} | test PPL {ppl:.3f}",
              flush=True)
        del m, opt
        torch.cuda.empty_cache()
        return {"config": cfg, "trainable": trainable, "total": total,
                "final_loss": last_loss, "test_ppl": ppl}

    wanted = [c.strip() for c in configs.split(",") if c.strip()]
    results = [train_one(c) for c in wanted]

    # Per-config JSON so parallel single-config containers don't clobber each
    # other; the local entrypoint (or a later reader) assembles the table.
    tag = "_".join(wanted) if len(wanted) <= 2 else "all"
    payload = {
        "model": MODEL, "dataset": "wikitext-2-raw-v1", "steps": steps,
        "seq_len": seq_len, "grad_accum": grad_accum, "lr": lr,
        "lora_r": lora_r, "results": results,
    }
    with open(f"{OUT_DIR}/attribution_{steps}steps_{tag}.json", "w") as f:
        json.dump(payload, f, indent=2)
    out_vol.commit()
    print(f"\nsaved {OUT_DIR}/attribution_{steps}steps_{tag}.json", flush=True)
    return payload


@app.local_entrypoint()
def main(
    steps: int = 1000,
    seq_len: int = 512,
    lr: float = 2e-4,
    lora_r: int = 8,
    max_eval_chunks: int = 0,
    configs: str = "baseline,lora_only,mt_only,mt_lora,mt_v2_only,mt_v2_lora",
):
    wanted = [c.strip() for c in configs.split(",") if c.strip()]
    # One container PER CONFIG, all in parallel: each stays far under the
    # timeout (the serial 4-config run blew the 3600s cap) and a straggler
    # can't take the others down with it.
    calls = [
        run_attribution.spawn(
            steps=steps, seq_len=seq_len, lr=lr, lora_r=lora_r,
            max_eval_chunks=max_eval_chunks, configs=cfg,
        )
        for cfg in wanted
    ]
    payloads = [c.get() for c in calls]

    results = [r for p in payloads for r in p["results"]]
    base_ppl = next((r["test_ppl"] for r in results if r["config"] == "baseline"), None)

    print("\n" + "=" * 78, flush=True)
    print(f"ATTRIBUTION ABLATION  |  {MODEL}  |  WikiText-2  |  {steps} steps", flush=True)
    print("=" * 78, flush=True)
    header = f"{'config':<11} {'trainable':>12} {'%params':>8} {'test PPL':>9} {'vs base':>8} {'dPPL/1M':>9}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for r in results:
        pct = 100 * r["trainable"] / r["total"]
        if base_ppl and r["trainable"] > 0:
            vs = f"{100 * (r['test_ppl'] - base_ppl) / base_ppl:+.1f}%"
            gpm = f"{(base_ppl - r['test_ppl']) / (r['trainable'] / 1e6):+.2f}"
        else:
            vs, gpm = "--", "--"
        print(f"{r['config']:<11} {r['trainable']:>12,} {pct:>7.3f}% "
              f"{r['test_ppl']:>9.3f} {vs:>8} {gpm:>9}", flush=True)
