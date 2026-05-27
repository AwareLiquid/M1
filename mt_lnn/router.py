"""
router.py — Cloud Oracle Router for Awareness Network.

This module acts as the "API Handshake" for the Hybrid Edge-State architecture,
allowing the local MT-LNN (AwareLiquid) to securely query a larger cloud model
(like Awareness Cloud, Gemini, or ChatGPT) strictly for factual knowledge when
it hits an internal cognitive blind spot.

The router itself stays thin: backend selection lives in ``cloud_client.py``
(env-driven, falls back to a Mock client when no API key is set), and the
three-way routing decision lives in ``deliberation.py``. This keeps the
router file backwards-compatible with the v1 MVP demo while letting the new
``DeliberationRouter`` plug in for Layer 2 work.
"""

import time
from typing import Optional, Dict

import torch

from .cloud_client import MockOracleClient, OracleClient, build_oracle_client


class CloudOracleRouter:
    """
    Manages asymmetric queries to external Cloud LLMs.

    Parameters
    ----------
    endpoint, api_key : legacy v1 args (kept for backward compatibility with
        the original investor demo). When neither side is configured the
        router uses the env-driven ``build_oracle_client()`` factory, which
        returns a real Gemini/OpenAI client iff the API key env var is set,
        otherwise a ``MockOracleClient``.
    client : explicit override. Useful in tests and when callers want full
        control over backend selection.
    """

    def __init__(
        self,
        endpoint: str = "mock",
        api_key: str = "mock_key",
        *,
        client: Optional[OracleClient] = None,
    ):
        self.endpoint = endpoint
        self.api_key = api_key
        if client is not None:
            self.client: OracleClient = client
        elif endpoint == "mock" and api_key == "mock_key":
            # Honour env-based backend if user configured one, else mock.
            self.client = build_oracle_client()
        else:
            # Explicit legacy args — preserve old mock-only behaviour.
            self.client = MockOracleClient()

    def query(self, topic: str) -> Dict[str, str]:
        """Send a surgical query to the Cloud Oracle and retrieve factual context.

        Returns a dict ``{"source": str, "fact": str}`` so callers can log
        provenance into the capsule.evidence_log.
        """
        print(f"\n[Cloud Oracle Router] Dispatching surgical query to cloud API: '{topic}'...")
        time.sleep(0.1)  # token gesture toward async-ness; real client adds its own latency
        result = self.client.query(topic)
        print(f"[Cloud Oracle Router] {len(result.fact)} bytes from {result.source}.")
        return {"source": result.source, "fact": result.fact}

    def inject_to_local_state(
        self,
        fact: str,
        tokenizer,
        model,
        cache,
        device: str = "cpu",
        *,
        source: Optional[str] = None,
        query_text: Optional[str] = None,
    ):
        """[Quiet Mode] Feed factual text into the engine without emitting tokens.

        Updates h_prev silently. If ``source`` and ``query_text`` are provided,
        appends a row to ``cache.evidence_log`` for audit purposes.
        """
        print("[Local Brain] Silently absorbing cloud facts into local awareness state...")

        from .streaming import prefill_state_only

        quiet_prompt = f"<|oracle|>\n{fact}\n<|endoracle|>\n"
        ids_list = [ord(c) % 200 for c in quiet_prompt]
        input_ids = torch.tensor([ids_list], dtype=torch.long).to(device)

        _, updated_cache = prefill_state_only(
            model,
            input_ids,
            cache=cache,
            use_lnn_recurrence=True,
        )

        if source is not None and query_text is not None:
            from .capsule import add_evidence
            add_evidence(updated_cache, source=source, query=query_text, fact=fact)

        print("[Local Brain] Absorption complete. O(1) state matrix updated.")
        return updated_cache
