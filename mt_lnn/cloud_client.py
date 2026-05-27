"""
cloud_client.py — Pluggable cloud-oracle clients for Layer 2.

The legacy ``CloudOracleRouter.query()`` returns a mock dict. This module
defines a small client protocol so we can swap in a real Gemini/OpenAI
adapter without touching the deliberation logic.

Activation is keyed on environment variables so the demo continues to
work offline:

    AWARELIQUID_CLOUD_BACKEND = "mock" | "gemini" | "openai"  (default mock)
    AWARELIQUID_CLOUD_API_KEY = secret token (required for non-mock)
    AWARELIQUID_CLOUD_MODEL   = model id (optional override)

When an HTTP-backed client is selected but the SDK / key is missing we
fall back to ``MockOracleClient`` and log a route event so the caller can
notice the degradation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


@dataclass
class OracleResult:
    """Uniform shape returned by every oracle backend."""

    source: str
    fact: str


@runtime_checkable
class OracleClient(Protocol):
    """Minimal client contract used by ``DeliberationRouter``."""

    name: str

    def query(self, topic: str) -> OracleResult: ...


# ---------------------------------------------------------------------------
# Built-in backends
# ---------------------------------------------------------------------------

class MockOracleClient:
    """Offline default. Returns canned facts from a tiny dictionary."""

    name = "mock"

    def __init__(self) -> None:
        self._db = {
            "m-theory": "M-theory unifies all consistent versions of superstring theory; Witten 1995.",
            "awareliquid": "AwareLiquid is an O(1) recurrent state engine inspired by microtubule dynamics.",
            "tokyo": "Tokyo is Japan's capital; Greater Tokyo is the world's most populous metro area.",
            "quantum": "Quantum mechanics describes nature at atomic and subatomic scales.",
            "mamba": "Mamba is a selective state-space model; struggles on extreme long context vs MT-LNN.",
        }

    def query(self, topic: str) -> OracleResult:
        for key, value in self._db.items():
            if key in topic.lower():
                return OracleResult(source=f"mock:{key}", fact=value)
        return OracleResult(
            source="mock:fallback",
            fact=f"No cached fact for '{topic}'. (mock fallback)",
        )


class HttpOracleClient:
    """HTTP backend stub.

    The real Gemini/OpenAI call is deliberately deferred to runtime: we
    do not want to take a hard dependency on either SDK at import time.
    If the SDK or API key is missing, ``query()`` raises so callers can
    fall back to ``MockOracleClient`` via ``build_oracle_client()``.
    """

    def __init__(self, backend: str, api_key: str, model: Optional[str] = None):
        self.name = backend
        self._api_key = api_key
        self._model = model or self._default_model(backend)

    @staticmethod
    def _default_model(backend: str) -> str:
        return {
            "gemini": "gemini-2.5-flash",
            "openai": "gpt-4o-mini",
        }.get(backend, "unknown")

    def query(self, topic: str) -> OracleResult:
        if self.name == "gemini":
            return self._query_gemini(topic)
        if self.name == "openai":
            return self._query_openai(topic)
        raise RuntimeError(f"Unsupported backend: {self.name}")

    # NB: the SDK calls below are imported lazily so this module stays
    # import-clean in offline environments and in unit tests.
    def _query_gemini(self, topic: str) -> OracleResult:
        try:
            from google import genai  # type: ignore
        except ImportError as exc:
            raise RuntimeError("google-genai SDK not installed") from exc
        client = genai.Client(api_key=self._api_key)
        resp = client.models.generate_content(
            model=self._model,
            contents=f"Give one concise factual paragraph about: {topic}",
        )
        text = getattr(resp, "text", None) or str(resp)
        return OracleResult(source=f"gemini:{self._model}", fact=text.strip())

    def _query_openai(self, topic: str) -> OracleResult:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError("openai SDK not installed") from exc
        client = OpenAI(api_key=self._api_key)
        resp = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": f"Give one concise factual paragraph about: {topic}"}],
        )
        text = resp.choices[0].message.content or ""
        return OracleResult(source=f"openai:{self._model}", fact=text.strip())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_oracle_client() -> OracleClient:
    """Return a client based on environment configuration.

    Always returns *something* — never raises. If the requested backend
    cannot be constructed, falls back to ``MockOracleClient`` so the
    pipeline keeps running (invariant I4: offline must work).
    """
    backend = os.environ.get("AWARELIQUID_CLOUD_BACKEND", "mock").lower()
    if backend == "mock":
        return MockOracleClient()

    api_key = os.environ.get("AWARELIQUID_CLOUD_API_KEY", "")
    if not api_key:
        return MockOracleClient()

    model = os.environ.get("AWARELIQUID_CLOUD_MODEL")
    try:
        return HttpOracleClient(backend=backend, api_key=api_key, model=model)
    except Exception:
        return MockOracleClient()
