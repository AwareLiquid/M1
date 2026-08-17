"""Real-inference reasoning trace for AwareLiquid.

Loads a HuggingFace causal LM (optionally with the Phase 5b MT residual
adapter), runs greedy/temperature decoding for a single prompt, and emits a
per-token ReasoningTrace event stream. At each step we compute the Shannon
entropy of the next-token distribution and pick a route:

    LOCAL                — entropy below the LOCAL_THRESH
    SELF_CRITIQUE        — entropy above LOCAL_THRESH but below CLOUD_THRESH
    CLOUD                — entropy above CLOUD_THRESH (mock inject)

For CLOUD we optionally splice an "Absorbed fact" template into the running
context (same template as bench_cloud_inject_uplift.py) using either a static
fact map or an EchoBackend stub. The point is to produce a real *.trace.jsonl
viewable in trace_timeline.html, not to win a benchmark.

    python scripts/awareliquid_real_trace.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --adapter benchmarks/kaggle_qwen_run/llama_mt_adapter_001000.pt \\
        --prompt "What is the capital of Australia? Answer:" \\
        --out artifacts/real_trace.jsonl --max_new 80
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from mt_lnn.reasoning_trace import ReasoningTrace


LOCAL_THRESH = 2.0
CLOUD_THRESH = 4.0
INJECT_TEMPLATE = "\n[Absorbed fact] {fact}\nContinuing: "


def shannon_entropy(logits: torch.Tensor) -> float:
    probs = F.softmax(logits.float(), dim=-1)
    logp = torch.log(probs.clamp_min(1e-12))
    return float(-(probs * logp).sum().item())


def pick_route(entropy: float, already_injected: bool) -> str:
    if entropy >= CLOUD_THRESH and not already_injected:
        return "cloud"
    if entropy >= LOCAL_THRESH:
        return "self_critique"
    return "local"


def load_model(model_id: str, adapter_path: Optional[str], device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    if adapter_path:
        from mt_lnn.llama_adapter import (
            attach_adapters_from_checkpoint,
            load_adapter_state,
        )

        ckpt = torch.load(adapter_path, map_location="cpu")
        attach_adapters_from_checkpoint(model, ckpt)
        load_adapter_state(model, adapter_path, strict=False)
        print(f"loaded MT adapter from {adapter_path}")
    model.to(device).eval()
    return tok, model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", default=None, help="Phase 5b adapter .pt (optional)")
    p.add_argument("--prompt", default="The capital of Australia is")
    p.add_argument("--out", default="real_trace.jsonl")
    p.add_argument("--max_new", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument(
        "--cloud_fact",
        default="Canberra is the capital city of Australia, located in the ACT.",
        help="Static fact spliced when the CLOUD route triggers.",
    )
    p.add_argument("--session_id", default="real_demo")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out = Path(args.out)
    if out.exists():
        out.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)

    tok, model = load_model(args.model, args.adapter, args.device)
    input_ids = tok(args.prompt, return_tensors="pt").input_ids.to(args.device)

    trace = ReasoningTrace(str(out), session_id=args.session_id, phi_every=0)
    trace.writer.write(
        "meta",
        {
            "model": args.model,
            "adapter": args.adapter or "",
            "prompt": args.prompt,
            "device": args.device,
            "local_thresh": LOCAL_THRESH,
            "cloud_thresh": CLOUD_THRESH,
        },
    )

    injected = False
    t0 = time.time()
    generated_ids = []

    with torch.inference_mode():
        for step in range(args.max_new):
            out_logits = model(input_ids=input_ids).logits[:, -1, :].squeeze(0)
            ent = shannon_entropy(out_logits)
            route = pick_route(ent, injected)

            if route == "cloud":
                trace.record_route(
                    route="cloud",
                    reason=f"entropy={ent:.2f} >= {CLOUD_THRESH}",
                    extras={"entropy": ent},
                )
                fact_text = INJECT_TEMPLATE.format(fact=args.cloud_fact)
                trace.record_cloud_inject(
                    source="static_oracle",
                    query=args.prompt,
                    fact_len=len(args.cloud_fact),
                )
                inject_ids = tok(fact_text, return_tensors="pt").input_ids.to(args.device)
                input_ids = torch.cat([input_ids, inject_ids], dim=1)
                injected = True
                continue
            if route == "self_critique":
                trace.record_route(
                    route="self_critique",
                    reason=f"entropy={ent:.2f} in [{LOCAL_THRESH},{CLOUD_THRESH})",
                    extras={"entropy": ent},
                )

            if args.temperature > 0:
                probs = F.softmax(out_logits / args.temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = out_logits.argmax(dim=-1, keepdim=True)
            tok_id = int(next_id.item())
            generated_ids.append(tok_id)

            trace.record_token(token_id=tok_id, entropy=ent, route=route)

            input_ids = torch.cat([input_ids, next_id.view(1, 1)], dim=1)
            if tok_id == tok.eos_token_id:
                break

    wall = time.time() - t0
    text = tok.decode(generated_ids, skip_special_tokens=True)
    trace.writer.write(
        "summary",
        {
            "n_tokens": len(generated_ids),
            "wall_s": round(wall, 2),
            "tokens_per_s": round(len(generated_ids) / max(wall, 1e-6), 2),
            "injected": injected,
            "completion": text,
        },
    )
    trace.close()
    print(f"wrote {len(generated_ids)} tokens · wall {wall:.1f}s · -> {out.resolve()}")
    print(f"completion: {text!r}")


if __name__ == "__main__":
    main()
