"""Tests for the Layer 2 DeliberationRouter and cloud client abstraction."""

import os
from unittest import mock

import math
import torch

from mt_lnn.deliberation import (
    DeliberationRouter,
    Route,
    RouterThresholds,
    lexical_fact_gap,
    semantic_entropy,
    token_entropy,
)
from mt_lnn.cloud_client import (
    HttpOracleClient,
    MockOracleClient,
    OracleResult,
    build_oracle_client,
)


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------

def test_token_entropy_zero_for_one_hot():
    logits = torch.tensor([10.0, -10.0, -10.0])
    assert token_entropy(logits) < 1e-3


def test_token_entropy_max_for_uniform():
    n = 8
    logits = torch.zeros(n)
    h = token_entropy(logits)
    # Uniform distribution → entropy ≈ log(n).
    assert abs(h - math.log(n)) < 1e-4


def test_semantic_entropy_zero_when_all_agree():
    samples = [[1, 2, 3]] * 5
    assert semantic_entropy(samples) == 0.0


def test_semantic_entropy_positive_when_diverge():
    samples = [[1, 2], [3, 4], [5, 6]]
    h = semantic_entropy(samples)
    assert h > 0.9  # 3 equally likely buckets → ≈ log(3) ≈ 1.098


def test_semantic_entropy_single_sample():
    assert semantic_entropy([[1, 2, 3]]) == 0.0


# ---------------------------------------------------------------------------
# Fact-gap
# ---------------------------------------------------------------------------

def test_fact_gap_true_when_no_evidence():
    assert lexical_fact_gap("What is m-theory?", []) is True


def test_fact_gap_false_when_evidence_overlaps():
    log = [{"query": "Explain m-theory origins"}]
    assert lexical_fact_gap("What is m-theory?", log) is False


def test_fact_gap_true_when_evidence_unrelated():
    log = [{"query": "what is the capital of france"}]
    assert lexical_fact_gap("describe quantum tunneling", log) is True


# ---------------------------------------------------------------------------
# Router decisions
# ---------------------------------------------------------------------------

def _logits_with_entropy(target_h: float, vocab: int = 64) -> torch.Tensor:
    """Produce logits whose Shannon entropy is approximately ``target_h``."""
    # For a logit vector [a, 0, 0, ..., 0] the softmax entropy is monotone
    # decreasing in a. We bisect.
    lo, hi = -10.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        logits = torch.zeros(vocab)
        logits[0] = mid
        h = token_entropy(logits)
        if h > target_h:
            lo = mid
        else:
            hi = mid
    logits = torch.zeros(vocab)
    logits[0] = (lo + hi) / 2
    return logits


def test_router_low_entropy_local():
    router = DeliberationRouter(RouterThresholds(low=3.0, high=5.0))
    logits = _logits_with_entropy(1.0)
    d = router.decide(logits, query="q", evidence_log=[])
    assert d.route == Route.LOCAL
    assert d.entropy < 3.0


def test_router_high_entropy_with_gap_goes_cloud():
    router = DeliberationRouter(RouterThresholds(low=3.0, high=4.0))
    logits = _logits_with_entropy(4.1)
    d = router.decide(logits, query="describe tokyo", evidence_log=[])
    assert d.route == Route.CLOUD
    assert d.fact_gap is True


def test_router_high_entropy_without_gap_self_critiques():
    router = DeliberationRouter(RouterThresholds(low=3.0, high=4.0))
    logits = _logits_with_entropy(4.1)
    log = [{"query": "describe tokyo"}]
    d = router.decide(logits, query="describe tokyo", evidence_log=log)
    assert d.route == Route.SELF_CRITIQUE
    assert d.fact_gap is False


def test_router_mid_entropy_without_sampler_self_critiques():
    router = DeliberationRouter(RouterThresholds(low=3.0, high=5.0))
    logits = _logits_with_entropy(4.0)
    d = router.decide(logits, query="anything", evidence_log=[])
    assert d.route == Route.SELF_CRITIQUE


def test_router_mid_entropy_with_divergent_sampler_and_gap_goes_cloud():
    samples = [[1, 2], [3, 4], [5, 6]]  # high semantic entropy
    router = DeliberationRouter(
        RouterThresholds(low=3.0, high=5.0),
        sampler=lambda: samples,
    )
    logits = _logits_with_entropy(4.0)
    d = router.decide(logits, query="brand new topic", evidence_log=[])
    assert d.route == Route.CLOUD
    assert d.semantic_entropy is not None and d.semantic_entropy > 0


def test_router_mid_entropy_with_convergent_sampler_self_critiques():
    samples = [[1, 2]] * 4  # tight cluster
    router = DeliberationRouter(
        RouterThresholds(low=3.0, high=5.0),
        sampler=lambda: samples,
    )
    logits = _logits_with_entropy(4.0)
    d = router.decide(logits, query="anything", evidence_log=[])
    assert d.route == Route.SELF_CRITIQUE
    assert d.semantic_entropy == 0.0


# ---------------------------------------------------------------------------
# Cloud client factory
# ---------------------------------------------------------------------------

def test_mock_client_returns_canned_fact():
    client = MockOracleClient()
    r = client.query("Tell me about m-theory")
    assert isinstance(r, OracleResult)
    assert "m-theory" in r.source
    assert len(r.fact) > 10


def test_mock_client_fallback_for_unknown_topic():
    client = MockOracleClient()
    r = client.query("the migration patterns of antarctic krill")
    assert r.source == "mock:fallback"


def test_build_client_defaults_to_mock(monkeypatch):
    monkeypatch.delenv("AWARELIQUID_CLOUD_BACKEND", raising=False)
    monkeypatch.delenv("AWARELIQUID_CLOUD_API_KEY", raising=False)
    client = build_oracle_client()
    assert isinstance(client, MockOracleClient)


def test_build_client_falls_back_to_mock_when_key_missing(monkeypatch):
    monkeypatch.setenv("AWARELIQUID_CLOUD_BACKEND", "gemini")
    monkeypatch.delenv("AWARELIQUID_CLOUD_API_KEY", raising=False)
    client = build_oracle_client()
    # Missing API key → graceful degrade.
    assert isinstance(client, MockOracleClient)


def test_build_client_returns_http_when_configured(monkeypatch):
    monkeypatch.setenv("AWARELIQUID_CLOUD_BACKEND", "gemini")
    monkeypatch.setenv("AWARELIQUID_CLOUD_API_KEY", "fake-key-for-test")
    client = build_oracle_client()
    assert isinstance(client, HttpOracleClient)
    assert client.name == "gemini"


def test_http_client_raises_when_sdk_missing():
    # We don't have google-genai installed in CI; query() must raise so the
    # caller can fall back, rather than silently producing nothing.
    client = HttpOracleClient(backend="gemini", api_key="x")
    raised = False
    try:
        client.query("anything")
    except RuntimeError:
        raised = True
    assert raised
