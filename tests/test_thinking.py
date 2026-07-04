"""Tests for the self-thinking serve path (``mt_lnn.thinking``).

The module is deliberately model-agnostic, so these tests drive it with a
tiny fake causal-LM + fake tokenizer (no weights, fully deterministic). We
verify: the self-consistency vote, route assignment as a function of
entropy, trace bookkeeping, graceful CLOUD degradation, and the renderers.
"""

import math

import torch

from mt_lnn.thinking import (
    StepTrace,
    ThinkingTrace,
    self_consistency_vote,
    generate_with_thinking,
    render_trace_markdown,
    render_trace_html,
)
from mt_lnn.deliberation import Route, RouterThresholds


# ---------------------------------------------------------------------------
# Fakes: a constant-logit LM and a char-level tokenizer
# ---------------------------------------------------------------------------

VOCAB = 16
EOS = VOCAB - 1


class _Out:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    """Returns a fixed next-token logit vector regardless of input.

    ``peak`` controls entropy: a large peak → near one-hot (low entropy →
    LOCAL route); ``peak=0`` → uniform (max entropy → CLOUD/critique route).
    """

    def __init__(self, peak: float, peak_token: int = 3):
        self.peak = peak
        self.peak_token = peak_token

    def parameters(self):
        # generate_with_thinking calls next(model.parameters()).device
        yield torch.zeros(1)

    def __call__(self, input_ids=None, past_key_values=None, use_cache=False,
                 output_hidden_states=False):
        # KV-cache kwargs accepted since generate_with_thinking decodes with
        # use_cache=True; the fake has no real cache — its logits depend only
        # on the newest position, which is all the router reads.
        b, t = input_ids.shape
        logits = torch.zeros(b, t, VOCAB)
        logits[:, -1, self.peak_token] = self.peak
        return _Out(logits)


class FakeTok:
    eos_token_id = EOS

    class _Enc:
        def __init__(self, ids):
            self.input_ids = ids

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        # Map each character to id (mod vocab); empty → single token 0.
        ids = [ord(c) % (VOCAB - 1) for c in text] or [0]
        return FakeTok._Enc(torch.tensor([ids]))

    def decode(self, ids, skip_special_tokens=False):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(chr(65 + (int(i) % 26)) for i in ids)


# ---------------------------------------------------------------------------
# self_consistency_vote
# ---------------------------------------------------------------------------

def test_vote_unanimous_when_one_hot():
    logits = torch.full((1, VOCAB), -10.0)
    logits[0, 5] = 20.0
    tok, sem_h, n = self_consistency_vote(logits, n_samples=5)
    assert tok == 5
    assert sem_h == 0.0          # all candidates identical
    assert n == 5


def test_vote_returns_valid_token_when_uniform():
    logits = torch.zeros(1, VOCAB)
    tok, sem_h, n = self_consistency_vote(logits, n_samples=7)
    assert 0 <= tok < VOCAB
    assert sem_h >= 0.0          # diverging candidates → entropy > 0 likely


def test_vote_handles_1d_logits():
    logits = torch.full((VOCAB,), -5.0)
    logits[2] = 10.0
    tok, _, _ = self_consistency_vote(logits, n_samples=3)
    assert tok == 2


# ---------------------------------------------------------------------------
# Routing as a function of entropy
# ---------------------------------------------------------------------------

def test_low_entropy_routes_local():
    # Sharp peak → entropy well below RouterThresholds.low (3.0).
    model = FakeModel(peak=30.0, peak_token=3)
    text, trace = generate_with_thinking(
        model, FakeTok(), "hi", max_new_tokens=5, device="cpu",
    )
    assert all(s.route == Route.LOCAL.value for s in trace.steps)
    assert trace.n_self_critique == 0


def test_high_entropy_routes_cloud_flag_and_degrades():
    # Uniform logits → entropy = log(16) ≈ 2.77 ... that's < low(3.0)!
    # Lower the low threshold so uniform counts as high-entropy here.
    model = FakeModel(peak=0.0)
    th = RouterThresholds(low=0.5, high=1.0)
    text, trace = generate_with_thinking(
        model, FakeTok(), "hi", max_new_tokens=4, thresholds=th, device="cpu",
    )
    # No cloud_fn wired → CLOUD steps must degrade to self-critique fallback.
    assert trace.n_cloud_flagged >= 1
    for s in trace.steps:
        if s.route == Route.CLOUD.value:
            assert "fallback" in s.reason
            assert s.n_resamples > 0


def test_cloud_fn_injection_used_once():
    model = FakeModel(peak=0.0)
    th = RouterThresholds(low=0.5, high=1.0)
    calls = []

    def cloud_fn(q):
        calls.append(q)
        return "FACT"

    text, trace = generate_with_thinking(
        model, FakeTok(), "hi", max_new_tokens=6, thresholds=th,
        cloud_fn=cloud_fn, device="cpu",
    )
    # cloud_fn should fire at most once (cloud_used latch).
    assert len(calls) == 1
    assert any(s.reason == "cloud_inject" for s in trace.steps)


# ---------------------------------------------------------------------------
# Trace bookkeeping + summary
# ---------------------------------------------------------------------------

def test_trace_summary_counts():
    trace = ThinkingTrace(steps=[
        StepTrace(0, 1, "A", 0.1, Route.LOCAL.value, "low_entropy"),
        StepTrace(1, 2, "B", 4.0, Route.SELF_CRITIQUE.value, "mid",
                  n_resamples=3, revised=True),
        StepTrace(2, 3, "C", 6.0, Route.CLOUD.value, "high"),
    ])
    s = trace.summary()
    assert s["n_tokens"] == 3
    assert s["route_counts"]["local"] == 1
    assert s["n_self_critique"] == 1
    assert s["n_cloud_flagged"] == 1
    assert s["n_revised"] == 1
    assert abs(trace.mean_entropy - (0.1 + 4.0 + 6.0) / 3) < 1e-6


def test_generation_stops_on_eos():
    # Peak the EOS token so the very first sampled token ends generation.
    model = FakeModel(peak=30.0, peak_token=EOS)
    text, trace = generate_with_thinking(
        model, FakeTok(), "hi", max_new_tokens=50, device="cpu",
    )
    assert len(trace.steps) == 1      # stopped immediately on EOS


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def test_render_markdown_contains_summary():
    trace = ThinkingTrace(steps=[
        StepTrace(0, 1, "A", 0.1, Route.LOCAL.value, "low_entropy"),
    ])
    md = render_trace_markdown(trace)
    assert "Self-thinking summary" in md
    assert "Tokens:" in md


def test_render_html_escapes_and_colours():
    trace = ThinkingTrace(steps=[
        StepTrace(0, 1, "<b>", 0.1, Route.LOCAL.value, "low_entropy"),
        StepTrace(1, 2, "x", 4.0, Route.SELF_CRITIQUE.value, "mid",
                  revised=True),
    ])
    html = render_trace_html(trace)
    assert "&lt;b&gt;" in html        # HTML-escaped
    assert "underline" in html        # revised token underlined
    assert "self-critique" in html    # legend present


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc!r}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
