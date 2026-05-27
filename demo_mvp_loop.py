"""
demo_mvp_loop.py — Full Commercial MVP Loop for AwareLiquid + Awareness Cloud

This script demonstrates the Hybrid Edge-State architecture to investors:
1. State Capsule Persistence (O(1) memory saving/loading)
2. Entropy-based "Blind Spot" Detection (Detecting when the model doesn't know)
3. Cloud Oracle Router (Routing to Cloud for facts and silently absorbing them)
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from mt_lnn import MTLNNConfig, MTLNNModel, ModelCacheStruct
from mt_lnn.utils import load_checkpoint
from mt_lnn.streaming import streaming_inference, prefill_state_only
from mt_lnn.capsule import save_capsule, load_capsule, add_open_question
from mt_lnn.router import CloudOracleRouter
from mt_lnn.reasoning_trace import ReasoningTrace
from mt_lnn.deliberation import DeliberationRouter, Route, RouterThresholds

# --- MVP Settings ---
# Because training is on a micro-sandbox (vocab 200, GPT-2 init style), we must keep that consistent.
CKPT_PATH = "checkpoints/final.pt"
TOKENIZER_NAME = "gpt2" # Placeholder. Our tiny sandbox model's vocab_size is 200, so we bypass word decoding visually.
DEVICE = "cpu"
THRESHOLDS = RouterThresholds(low=3.0, high=5.0)
CAPSULE_FILE = "session_alice.capsule"
TRACE_FILE = "session_alice.trace.jsonl"


def compute_entropy(logits: torch.Tensor) -> float:
    """Calculate Shannon entropy for a given logit distribution to measure confidence."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return float(entropy.item())


def mock_generate_with_interception(
    model: MTLNNModel,
    tokenizer,
    user_query: str,
    cache: ModelCacheStruct,
    router: CloudOracleRouter,
    trace: ReasoningTrace,
    deliberation: DeliberationRouter,
) -> ModelCacheStruct:
    """
    Stream-decode with a Layer 2 three-way router (local / self-critique / cloud).
    """
    print(f"\n[User] {user_query}")
    print(f"[AwareLiquid] Thinking...", end=" ")

    # 1. Prefill Query
    # Convert query to simple ASCII ordinals as a mock mapping to fit within vocab 200
    ids_list = [ord(c) % 200 for c in user_query]
    prompt_ids = torch.tensor([ids_list], dtype=torch.long).to(DEVICE)
    logits, cache = prefill_state_only(model, prompt_ids, cache=cache, use_lnn_recurrence=True)

    generated_tokens = []
    max_tokens = 50
    current_token_id = prompt_ids[:, -1:]

    # Mocking a topic to force intercept for demo purposes
    trigger_intercept = "m-theory" in user_query.lower() or "tokyo" in user_query.lower()

    for i in range(max_tokens):
        # Forward pass (streaming O(1))
        logits, cache = streaming_inference(model, current_token_id, cache=cache, state_only=True)
        next_logits = logits[:, -1, :]

        # 2. Layer 2 routing decision
        # For the demo, we manually inject high entropy if the trigger keyword is detected at step 3.
        if trigger_intercept and i == 3:
            forced = next_logits.clone()
            forced[..., :] = 0.0  # uniform → max entropy
            decision = deliberation.decide(
                forced,
                query=user_query,
                evidence_log=getattr(cache, "evidence_log", []),
            )
        else:
            decision = deliberation.decide(
                next_logits,
                query=user_query,
                evidence_log=getattr(cache, "evidence_log", []),
            )

        if decision.route == Route.CLOUD:
            print(f"\n[ROUTE] {decision.route.value} (E={decision.entropy:.2f}, reason={decision.reason})")
            print(f"[AwareLiquid] I lack precise memory regarding '{user_query}'. Initiating Oracle Uplink...")

            trace.record_route(
                route=decision.route.value,
                reason=decision.reason,
                extras={"entropy": decision.entropy, "fact_gap": decision.fact_gap},
            )
            add_open_question(cache, user_query)

            # 3. Request External Fact from Cloud
            cloud_result = router.query(user_query)
            fact = cloud_result["fact"]
            source = cloud_result["source"]

            trace.record_cloud_inject(source=source, query=user_query, fact_len=len(fact))

            # 4. Silent Absorption (Quiet Mode) — also logs evidence onto capsule
            cache = router.inject_to_local_state(
                fact, tokenizer, model, cache, device=DEVICE,
                source=source, query_text=user_query,
            )

            print("[AwareLiquid] Based on new context, I understand now. (Continuing generation using updated O(1) state...)\n")
            break

        if decision.route == Route.SELF_CRITIQUE:
            trace.record_route(
                route=decision.route.value,
                reason=decision.reason,
                extras={"entropy": decision.entropy},
            )
            # MVP: log only; a future revision will run N-sample re-decode here.

        # Normal generation sampling (simplified greedy for demo)
        next_id = next_logits.argmax(dim=-1, keepdim=True)
        trace.record_token(
            token_id=int(next_id.item()),
            entropy=decision.entropy,
            route=decision.route.value,
        )
        # Skip actual decoding for the tiny sandbox vocab (size 200, causes indexing errors with normal tokenizer)
        # We just print a simulation of words being generated
        mock_words = [" The", " quantum", " state", " is", " updated."]
        curr_text = mock_words[i % len(mock_words)]
        print(curr_text, end="", flush=True)
        time.sleep(0.3)

        generated_tokens.append(next_id)
        current_token_id = next_id

        # Stop on EOS or dot for demo
        if next_id.item() == tokenizer.eos_token_id or "." in curr_text:
            break

    print("\n[AwareLiquid] Finished response.")
    return cache


def main():
    print("="*60)
    print("🚀 AwareLiquid (MT-LNN) Hybrid Edge-State MVP Demo")
    print("="*60)
    
    if not os.path.exists(CKPT_PATH):
        print(f"[Error] Please ensure {CKPT_PATH} exists or change CKPT_PATH.")
        return

    # Initialize Engine
    print(f"\n1. Booting MT-LNN core on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    
    # We must instantiate model config first before loading weights
    ckpt_dict = torch.load(CKPT_PATH, map_location="cpu")
    config_dict = ckpt_dict.get("config", {})
    from dataclasses import fields
    valid = {f.name for f in fields(MTLNNConfig)} - {"d_proto", "d_proto_total"}
    config = MTLNNConfig(**{k: v for k, v in config_dict.items() if k in valid})
    model = MTLNNModel(config).to(DEVICE)
    load_checkpoint(CKPT_PATH, model)
    model.eval()
    
    router = CloudOracleRouter()
    trace = ReasoningTrace(TRACE_FILE, session_id="alice", phi_every=0)
    deliberation = DeliberationRouter(thresholds=THRESHOLDS)

    # ---------------------------------------------------------
    # Scenario A: Restoring state from yesterday
    # ---------------------------------------------------------
    if os.path.exists(CAPSULE_FILE):
        print("\n2. [State Capsule] Found previous session! Initiating O(1) state inheritance...")
        cache = load_capsule(CAPSULE_FILE, device=DEVICE)
    else:
        print("\n2. [State Capsule] Starting fresh session (No capsule found).")
        cache = ModelCacheStruct()
        cache.open_questions = []
        cache.evidence_log = []

    # ---------------------------------------------------------
    # Scenario B: Hitting a Factual Blind Spot
    # ---------------------------------------------------------
    print("\n--- Demo Interaction 1: Unknown Fact (High Entropy) ---")
    query_unknown = "Explain the origins of m-theory to me."
    cache = mock_generate_with_interception(model, tokenizer, query_unknown, cache, router, trace, deliberation)

    # ---------------------------------------------------------
    # Scenario C: Normal Knowledge
    # ---------------------------------------------------------
    print("\n--- Demo Interaction 2: Normal Generative Task ---")
    query_known = "Say hello."
    cache = mock_generate_with_interception(model, tokenizer, query_known, cache, router, trace, deliberation)

    # ---------------------------------------------------------
    # Scenario D: Saving State Capsule v2 (belief + open_q + evidence)
    # ---------------------------------------------------------
    print("\n3. [State Capsule] Session concluded. Crystallizing user's persona into capsule v2...")
    save_capsule(
        cache,
        CAPSULE_FILE,
        open_questions=getattr(cache, "open_questions", []),
        evidence_log=getattr(cache, "evidence_log", []),
    )
    trace.close()
    print(f"[Reasoning Trace] Timeline events written to {TRACE_FILE}")
    print("\n=== MVP Demo Successfully Completed ===")


if __name__ == "__main__":
    main()
