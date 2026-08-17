"""
tests/test_server_hf.py -- offline contract tests for the HF-adapter server
(serve/server_hf.py).

The server's startup loads a HuggingFace causal LM (a one-time download), so it
is deliberately NOT exercised here. Instead we pin the parts that must hold
*without* a model:

  * the FastAPI app + the routes the frontend depends on exist;
  * endpoints guard on readiness (503 / "starting" before startup ran);
  * CompletionRequest validation (max_new_tokens >= 1, temperature > 0);
  * the pure next-token selection (greedy == argmax & deterministic; sampling
    stays within the top-k support) -- the logic that previously hid a NameError
    in the greedy streaming branch.

Skipped automatically if FastAPI is unavailable.
"""
import os
import sys

import pytest
import torch

pytest.importorskip("fastapi")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fastapi import HTTPException  # noqa: E402

from serve import server_hf  # noqa: E402


# --- HTTP surface ----------------------------------------------------------


def test_app_exposes_expected_routes():
    paths = {getattr(r, "path", None) for r in server_hf.app.routes}
    assert {"/", "/health", "/v1/model",
            "/v1/completions", "/v1/completions/stream"} <= paths


def test_health_reports_starting_before_model_loaded():
    # No test triggers @app.on_event("startup"), so the model never loads.
    assert server_hf.health()["status"] == "starting"


def test_endpoints_503_before_ready():
    with pytest.raises(HTTPException) as ei:
        server_hf.model_info()
    assert ei.value.status_code == 503
    req = server_hf.CompletionRequest(prompt="hi", max_new_tokens=4)
    with pytest.raises(HTTPException) as ej:
        server_hf.completions(req)
    assert ej.value.status_code == 503


# --- request schema --------------------------------------------------------


def test_request_schema_validates():
    server_hf.CompletionRequest(prompt="x", max_new_tokens=1)        # ok
    with pytest.raises(Exception):
        server_hf.CompletionRequest(max_new_tokens=0)                # ge=1
    with pytest.raises(Exception):
        server_hf.CompletionRequest(temperature=0.0)                 # gt=0


# --- pure next-token selection (the bug that used to hide here) -------------


def test_greedy_is_argmax_and_deterministic():
    logits = torch.tensor([[0.1, 2.0, -1.0, 0.5]])
    req = server_hf.CompletionRequest(do_sample=False)
    a = server_hf._sample_next_token(logits, req)
    b = server_hf._sample_next_token(logits, req)
    assert a == b == 1                                               # argmax index


def test_sampling_stays_within_top_k():
    torch.manual_seed(0)
    # only indices 0 and 1 carry meaningful mass; top_k=2 must exclude the rest
    logits = torch.tensor([[5.0, 4.0, -10.0, -10.0, -10.0]])
    req = server_hf.CompletionRequest(do_sample=True, top_k=2, temperature=1.0, top_p=1.0)
    picks = {server_hf._sample_next_token(logits, req) for _ in range(64)}
    assert picks <= {0, 1}


def test_nucleus_keeps_at_least_the_top_token():
    # a very peaked distribution with tiny top_p must still return the peak
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    req = server_hf.CompletionRequest(do_sample=True, top_p=0.01, top_k=0, temperature=1.0)
    picks = {server_hf._sample_next_token(logits, req) for _ in range(32)}
    assert picks == {0}
