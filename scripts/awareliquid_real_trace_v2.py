"""Real-inference reasoning trace for AwareLiquid (optimized with KV cache).

Uses model.generate() + custom LogitsProcessor instead of manual token loop.
This enables automatic KV cache → O(N) instead of O(N²).

    python scripts/awareliquid_real_trace_v2.py \\
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
from transformers import LogitsProcessor

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


class ReasoningTraceProcessor(LogitsProcessor):
    """Records per-token entropy + route decisions in ReasoningTrace."""

    def __init__(
        self,
        trace: ReasoningTrace,
        injected: bool,
        local_thresh: float,
        cloud_thresh: float,
    ):
        self.trace = trace
        self.injected = injected
        self.local_thresh = local_thresh
        self.cloud_thresh = cloud_thresh
        self.should_stop = False
        self.token_metadata = []  # List of (entropy, route) per token

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # scores shape: (batch=1, vocab)
        ent = shannon_entropy(scores[0])
        route = pick_route(ent, self.injected)

        if route == "cloud":
            # Signal to stop generation so caller can inject fact
            self.should_stop = True
            self.trace.record_route(
                route="cloud",
                reason=f"entropy={ent:.2f} >= {self.cloud_thresh}",
                extras={"entropy": ent},
            )
            self.token_metadata.append((ent, route))
            return scores

        if route == "self_critique":
            self.trace.record_route(
                route="self_critique",
                reason=f"entropy={ent:.2f} in [{self.local_thresh},{self.cloud_thresh})",
                extras={"entropy": ent},
            )

        # Store (entropy, route) for this token
        self.token_metadata.append((ent, route))
        return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", default=None, help="Phase 5b adapter .pt (optional)")
    p.add_argument("--prompt", default="The capital of Australia is")
    p.add_argument("--out", default="real_trace_v2.jsonl")
    p.add_argument("--max_new", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument(
        "--cloud_fact",
        default="Canberra is the capital city of Australia, located in the ACT.",
        help="Static fact spliced when CLOUD route triggers.",
    )
    p.add_argument("--session_id", default="real_demo_v2")
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

    prompt_text = args.prompt
    injected = False
    t0 = time.time()
    all_generated_ids = []

    with torch.inference_mode():
        while len(all_generated_ids) < args.max_new:
            input_ids = tok(prompt_text, return_tensors="pt").input_ids.to(args.device)

            processor = ReasoningTraceProcessor(
                trace, injected, LOCAL_THRESH, CLOUD_THRESH
            )

            # Generate with KV cache enabled
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new - len(all_generated_ids),
                temperature=args.temperature if args.temperature > 0 else 1.0,
                do_sample=args.temperature > 0,
                logits_processor=[processor],
                use_cache=True,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

            new_ids = outputs.sequences[0, input_ids.shape[1] :].tolist()

            # Record tokens with stored entropy/route from processor
            for i, tok_id in enumerate(new_ids):
                if i < len(processor.token_metadata):
                    ent, route = processor.token_metadata[i]
                else:
                    ent, route = 0.0, "local"
                trace.record_token(token_id=tok_id, entropy=ent, route=route)
                all_generated_ids.append(tok_id)

                if tok_id == tok.eos_token_id:
                    break

                # Check if we hit CLOUD route
                if processor.should_stop and not injected:
                    # Inject fact and continue
                    fact_text = INJECT_TEMPLATE.format(fact=args.cloud_fact)
                    trace.record_cloud_inject(
                        source="static_oracle",
                        query=args.prompt,
                        fact_len=len(args.cloud_fact),
                    )
                    prompt_text = prompt_text + tok.decode(new_ids[: i + 1]) + fact_text
                    injected = True
                    break

            if tok.eos_token_id in new_ids or processor.should_stop:
                if not processor.should_stop:
                    break
            else:
                # Normal completion without CLOUD
                prompt_text = prompt_text + tok.decode(new_ids)
                break

    wall = time.time() - t0
    text = tok.decode(all_generated_ids, skip_special_tokens=True)
    trace.writer.write(
        "summary",
        {
            "n_tokens": len(all_generated_ids),
            "wall_s": round(wall, 2),
            "tokens_per_s": round(len(all_generated_ids) / max(wall, 1e-6), 2),
            "injected": injected,
            "completion": text,
        },
    )
    trace.close()
    print(f"wrote {len(all_generated_ids)} tokens · wall {wall:.1f}s · -> {out.resolve()}")
    print(f"completion: {text!r}")


if __name__ == "__main__":
    main()
