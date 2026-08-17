"""Capability evals (roadmap #4): does the adapter change ABILITIES, not
just perplexity?

Runs lm-evaluation-harness on (a) the frozen base model and (b) the base +
a trained adapter checkpoint (v1 or v2 — rebuilt from the checkpoint's own
recorded args, same code path as serve/server_hf.py), on a small standard
suite: LAMBADA-openai (long-range cloze), ARC-easy, HellaSwag, PIQA.

Honest framing: the attribution result predicts the adapter should be
~neutral here (its PPL contribution beyond LoRA is nil); this eval checks
the SFT checkpoint didn't TRADE core abilities for chat formatting, and
gives the paper a capabilities table.

Usage:
    python benchmarks/capability_eval.py --ckpt checkpoints/llama_mt_adapter/llama_mt_adapter_v2s_003000.pt
    python benchmarks/capability_eval.py --ckpt "" --tag base   # base only
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# DLL-order guard (Windows): datasets/pyarrow before transformers' tokenizers.
import datasets as _datasets  # noqa: F401


def build_model(model_name: str, ckpt_path: str, device: str, dtype):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    m.config.use_cache = True
    if not ckpt_path:
        return m.to(device).eval(), "base"

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cargs = ck.get("args", {})
    kind = str(cargs.get("adapter", "v1")).lower()
    if kind == "v2":
        from mt_lnn.mt_lnn_v2 import attach_mt_v2_adapters
        attach_mt_v2_adapters(
            m, every=int(cargs.get("mt_every", 4)),
            n_protofilaments=int(cargs.get("mt_proto", 13)),
            d_proto=int(cargs.get("v2_d_proto", 64)),
            n_time_scales=int(cargs.get("mt_scales", 5)),
            proj_rank=int(cargs.get("v2_rank", 128)),
            init_scale=float(cargs.get("mt_init_scale", 1e-3)),
            selective_decay=bool(cargs.get("v2_selective", False)),
            use_fast_weight=not bool(cargs.get("v2_no_fw", False)),
            fast_weight_dim=int(cargs.get("v2_fw_dim", 64)),
            fast_weight_heads=int(cargs.get("v2_fw_heads", 1)),
        )
    else:
        from mt_lnn.llama_adapter import attach_adapters_from_checkpoint
        attach_adapters_from_checkpoint(m, ck)
    if cargs.get("lora"):
        from peft import LoraConfig, get_peft_model
        m = get_peft_model(m, LoraConfig(
            r=int(cargs.get("lora_r", 8)),
            lora_alpha=int(cargs.get("lora_alpha", 16)),
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            target_modules=str(cargs.get(
                "lora_targets", "q_proj,k_proj,v_proj,o_proj")).split(","),
        ))
    sd = ck.get("state_dict", {})
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert len(unexpected) == 0, (
        f"{len(unexpected)} checkpoint tensors did not map (recipe mismatch): "
        f"{unexpected[:4]}"
    )
    print(f"[ckpt] {len(sd)} tensors loaded, 0 unexpected ({kind})", flush=True)
    # Streaming state ON: same inference semantics the server uses.
    from mt_lnn.llama_adapter import set_adapter_streaming
    set_adapter_streaming(m, True)
    return m.to(device).eval(), f"adapter-{kind}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--ckpt", default="checkpoints/llama_mt_adapter/"
                                      "llama_mt_adapter_v2s_003000.pt")
    ap.add_argument("--tag", default="", help="label override for the output")
    ap.add_argument("--tasks", default="lambada_openai,arc_easy,hellaswag,piqa")
    ap.add_argument("--limit", type=int, default=0,
                    help="per-task example cap (0 = full task; use small "
                         "values for smoke tests)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out_dir", default="benchmarks/capability_out")
    args = ap.parse_args()

    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.float16 if device == "cuda" else torch.float32)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    model, tag = build_model(args.model, args.ckpt, device, dtype)
    tag = args.tag or tag
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=args.batch)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    res = lm_eval.simple_evaluate(
        model=lm, tasks=tasks,
        limit=(args.limit or None),
    )
    table = {}
    for task, metrics in res["results"].items():
        keep = {k: v for k, v in metrics.items()
                if k.split(",")[0] in ("acc", "acc_norm", "perplexity")}
        table[task] = keep
        print(f"{task:<16} " + "  ".join(f"{k}={v:.4f}" for k, v in keep.items()
                                         if isinstance(v, float)), flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"capability_{tag}.json")
    with open(path, "w") as f:
        json.dump({"tag": tag, "model": args.model, "ckpt": args.ckpt,
                   "limit": args.limit, "results": table}, f, indent=2)
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
