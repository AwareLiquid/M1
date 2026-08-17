"""
demo_awareliquid_v2.py — Phase 7 + Phase 6: real-LM AwareLiquid loop.

Drops the vocab=200 sandbox of ``demo_mvp_loop.py`` and runs the full
AwareLiquid loop on a real pretrained HuggingFace causal LM plus the
MT-LNN residual adapter trained in Phase 5:

    base LM (frozen)         — TinyLlama-1.1B-Chat-v1.0 by default
    + MT residual adapters   — 6 decoder layers
    + LoRA on q/k/v/o
    + DeliberationRouter     — 3-way local / self-critique / cloud
    + CloudOracleRouter      — env-driven (mock / gemini / openai)
    + Capsule v2             — open_questions + evidence_log persistence
    + ReasoningTrace         — JSONL audit trail (+ optional Φ̂)

What is "capsule" for an HF model?
----------------------------------
TinyLlama has no MT-LNN-style O(1) recurrent state we can serialise.
Capsule v2 here carries the *thinking context* only — open_questions +
evidence_log + recent conversation summary. The h_states field is left
empty (None). When we replace the backbone with a real MT-LNN at full
scale in Phase 9+, the same Capsule v2 schema picks up h_states without
any callsite change.

Cloud inject for HF models
--------------------------
Without prefill_state_only we cannot silently absorb a fact into hidden
state. Instead we prepend the fact to the conversation context as a
system note before continuing generation. This is the honest HF analogue
of "quiet inject" — provenance still tracked in evidence_log.

Run
---
    python demo_awareliquid_v2.py \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --adapter checkpoints/llama_mt_adapter/llama_mt_adapter_001000.pt \
        --prompt "Explain the origins of m-theory to me."

    # No adapter (base only):
    python demo_awareliquid_v2.py --prompt "Hello"

    # Enable Φ̂ sampling every 8 tokens (slow, only on small models):
    python demo_awareliquid_v2.py --phi_every 8
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

from mt_lnn.deliberation import DeliberationRouter, Route, RouterThresholds
from mt_lnn.reasoning_trace import ReasoningTrace
from mt_lnn.router import CloudOracleRouter
from mt_lnn.cloud_client import build_oracle_client
from mt_lnn.session_state import (
    HFSessionState,
    save_session,
    load_session,
)


# ---------------------------------------------------------------------- model

def _load_lora(model, checkpoint: dict):
    saved_args = checkpoint.get("args", {})
    if not saved_args.get("lora", False):
        return model
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=int(saved_args.get("lora_r", 8)),
        lora_alpha=int(saved_args.get("lora_alpha", 16)),
        lora_dropout=float(saved_args.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=str(
            saved_args.get("lora_targets", "q_proj,k_proj,v_proj,o_proj")
        ).split(","),
    )
    return get_peft_model(model, config)


def load_backbone(model_name: str, adapter_path: Optional[str]):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mt_lnn.llama_adapter import (
        attach_adapters_from_checkpoint,
        load_adapter_state,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype if device == "cuda" else torch.float32,
        device_map=None,
    )

    if adapter_path:
        ckpt = torch.load(adapter_path, map_location="cpu")
        attach_adapters_from_checkpoint(model, ckpt)
        model = _load_lora(model, ckpt)
        load_adapter_state(model, adapter_path, strict=False)
        print(f"[backbone] loaded adapter: {adapter_path}")
    else:
        print("[backbone] no adapter — running base only")

    model.to(device).eval()
    return model, tokenizer, device


# ----------------------------------------------------------- generation loop

@torch.no_grad()
def generate_with_router(
    *,
    model,
    tokenizer,
    device: str,
    prompt: str,
    session: HFSessionState,
    deliberation: DeliberationRouter,
    cloud: CloudOracleRouter,
    trace: ReasoningTrace,
    max_tokens: int,
    temperature: float,
    top_p: float,
    phi_every: int,
) -> str:
    """Generate one user-turn response with router + trace + capsule update."""

    # Build the prompt with any prior evidence as a system preface.
    preface = ""
    if session.evidence_log:
        bullets = "\n".join(
            f"- ({row.get('source','?')}) {row.get('query','?')}"
            for row in session.evidence_log[-5:]
        )
        preface = (
            "Earlier in this session you absorbed these facts:\n" + bullets + "\n\n"
        )
    full_prompt = preface + prompt
    ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)
    eos_id = tokenizer.eos_token_id

    print(f"\n[user] {prompt}")
    print("[awareliquid] ", end="", flush=True)
    generated = []
    cloud_inject_used = False

    for step in range(max_tokens):
        out = model(input_ids=ids)
        next_logits = out.logits[:, -1, :] / max(temperature, 1e-6)

        decision = deliberation.decide(
            next_logits,
            query=prompt,
            evidence_log=session.evidence_log,
        )

        # Φ̂ sample (Phase 6) — sparse, expensive
        phi_val: Optional[float] = None
        if phi_every > 0 and step > 0 and step % phi_every == 0:
            phi_val = _sample_phi_hat(model, ids[:, -min(16, ids.shape[1]):])

        if decision.route == Route.CLOUD and not cloud_inject_used:
            print(
                f"\n  [route] CLOUD (E={decision.entropy:.2f}, "
                f"reason={decision.reason})",
                flush=True,
            )
            trace.record_route(
                route=decision.route.value,
                reason=decision.reason,
                extras={"entropy": decision.entropy, "fact_gap": decision.fact_gap},
            )
            session.open_questions.append(prompt)

            result = cloud.query(prompt)
            fact = result["fact"]
            source = result["source"]
            trace.record_cloud_inject(source=source, query=prompt, fact_len=len(fact))
            session.evidence_log.append(
                {
                    "source": source,
                    "query": prompt,
                    "fact_len": len(fact),
                    "ts": time.time(),
                }
            )

            inject_text = f"\n[Absorbed fact] {fact}\nContinuing: "
            inject_ids = tokenizer(
                inject_text, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(device)
            ids = torch.cat([ids, inject_ids], dim=1)
            cloud_inject_used = True
            print(f"  [inject] +{len(fact)}B from {source}\n  ", end="", flush=True)
            continue

        if decision.route == Route.SELF_CRITIQUE:
            trace.record_route(
                route=decision.route.value,
                reason=decision.reason,
                extras={"entropy": decision.entropy},
            )
            # Future: N-sample re-decode here.

        # Sample next token (top-p nucleus)
        filtered = _top_p_filter(next_logits, top_p)
        probs = F.softmax(filtered, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        trace.record_token(
            token_id=int(next_id.item()),
            entropy=decision.entropy,
            route=decision.route.value,
            phi=phi_val,
        )

        text_piece = tokenizer.decode(next_id[0], skip_special_tokens=True)
        print(text_piece, end="", flush=True)
        generated.append(int(next_id.item()))

        ids = torch.cat([ids, next_id], dim=1)
        if eos_id is not None and int(next_id.item()) == eos_id:
            break

    print()
    response = tokenizer.decode(generated, skip_special_tokens=True)
    session.history.append({"role": "user", "text": prompt})
    session.history.append({"role": "assistant", "text": response})
    return response


def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    keep = probs.cumsum(dim=-1) <= p
    keep[..., 0] = True
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(-1, sorted_idx, keep)
    return logits.masked_fill(~mask, float("-inf"))


# ------------------------------------------------------------------ Phase 6: Φ̂

def _sample_phi_hat(model, probe_ids: torch.Tensor) -> Optional[float]:
    """Compute kNN Φ̂ on last-layer hidden states for the probe batch.

    Uses ``mt_lnn.phi_hat.compute_phi_hat`` directly so we don't need an
    MT-LNN-specific model. Returns ``None`` on any failure (Φ̂ is
    diagnostic, not load-bearing).
    """
    try:
        from mt_lnn.phi_hat import compute_phi_hat
    except Exception:
        return None

    captured: List[torch.Tensor] = []

    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured.append(h.detach().float().cpu())

    target = None
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        target = model.model.norm
    elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        target = model.transformer.ln_f
    if target is None:
        return None

    handle = target.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=probe_ids)
    finally:
        handle.remove()
    if not captured:
        return None

    hidden = captured[-1].reshape(-1, captured[-1].shape[-1])
    if hidden.shape[0] < 8:
        return None
    try:
        return float(compute_phi_hat(hidden, K=4, k_nn=3))
    except Exception:
        return None


# ------------------------------------------------------------------------ CLI

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--adapter", default=None)
    p.add_argument("--prompt", default="Explain the origins of m-theory to me.")
    p.add_argument("--session", default="session_v2")
    p.add_argument("--max_tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--entropy_low", type=float, default=3.0)
    p.add_argument("--entropy_high", type=float, default=5.0)
    p.add_argument(
        "--phi_every",
        type=int,
        default=0,
        help="Sample Φ̂ every N tokens (0 disables, slow when on)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    model, tokenizer, device = load_backbone(args.model, args.adapter)

    session_path = Path(f"{args.session}.json")
    trace_path = Path(f"{args.session}.trace.jsonl")
    session = (
        load_session(str(session_path))
        if session_path.exists()
        else HFSessionState(session_id=args.session)
    )

    deliberation = DeliberationRouter(
        RouterThresholds(low=args.entropy_low, high=args.entropy_high)
    )
    cloud = CloudOracleRouter(client=build_oracle_client())
    trace = ReasoningTrace(str(trace_path), session_id=args.session, phi_every=args.phi_every)

    try:
        generate_with_router(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=args.prompt,
            session=session,
            deliberation=deliberation,
            cloud=cloud,
            trace=trace,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            phi_every=args.phi_every,
        )
    finally:
        save_session(session, str(session_path))
        trace.close()
        print(f"\n[capsule] {session_path}  ({len(session.open_questions)} open Q, "
              f"{len(session.evidence_log)} evidence rows)")
        print(f"[trace]   {trace_path}")


if __name__ == "__main__":
    main()
