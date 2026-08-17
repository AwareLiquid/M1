"""Real-inference reasoning trace for AwareLiquid (v3: proper KV cache).

v3 fixes v2 bugs:
- v2 used LogitsProcessor but couldn't interrupt generate() → infinite loop
- v3 uses manual token loop with past_key_values → O(N) with proper stopping

    python scripts/awareliquid_real_trace_v3.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --adapter benchmarks/kaggle_qwen_run/llama_mt_adapter_001000.pt \\
        --prompt "What is the capital of Australia?" \\
        --out artifacts/real_trace.jsonl --max_new 80
"""

from __future__ import annotations

import argparse
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
    p.add_argument("--out", default="real_trace_v3.jsonl")
    p.add_argument("--max_new", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument(
        "--cloud_fact",
        default="Canberra is the capital city of Australia, located in the ACT.",
        help="Static fact spliced when CLOUD route triggers.",
    )
    p.add_argument("--session_id", default="real_demo_v3")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out = Path(args.out)
    if out.exists():
        out.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)

    tok, model = load_model(args.model, args.adapter, args.device)

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

    input_ids = tok(args.prompt, return_tensors="pt").input_ids.to(args.device)
    injected = False
    t0 = time.time()
    generated_ids = []
    past_key_values = None

    with torch.inference_mode():
        for step in range(args.max_new):
            # Forward pass with KV cache
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token_logits = outputs.logits[:, -1, :].squeeze(0)
            past_key_values = outputs.past_key_values

            # Compute entropy and route
            ent = shannon_entropy(next_token_logits)
            route = pick_route(ent, injected)

            # Handle CLOUD route: inject fact and restart generation
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
                # Reconstruct input with fact injected
                current_text = args.prompt + tok.decode(generated_ids, skip_special_tokens=False)
                new_input = current_text + fact_text
                input_ids = tok(new_input, return_tensors="pt").input_ids.to(args.device)
                past_key_values = None  # Reset KV cache after inject
                injected = True
                continue

            # Record self_critique route
            if route == "self_critique":
                trace.record_route(
                    route="self_critique",
                    reason=f"entropy={ent:.2f} in [{LOCAL_THRESH},{CLOUD_THRESH})",
                    extras={"entropy": ent},
                )

            # Sample next token
            if args.temperature > 0:
                probs = F.softmax(next_token_logits / args.temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = next_token_logits.argmax(dim=-1, keepdim=True)

            tok_id = int(next_id.item())
            generated_ids.append(tok_id)

            # Record token in trace
            trace.record_token(token_id=tok_id, entropy=ent, route=route)

            # Stop if EOS
            if tok_id == tok.eos_token_id:
                break

            # Prepare for next iteration
            input_ids = next_id.view(1, 1)

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
