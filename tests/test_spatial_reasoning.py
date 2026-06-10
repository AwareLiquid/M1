"""
tests/test_spatial_reasoning.py — 空间思考 (mt_lnn.spatial_reasoning).

Pins the convergence contract: SpatialReasoner perceives a spatial scene with
SpatialCoordEncoder, runs the MT-LNN backbone over the (optionally text-fused)
tokens, and deliberates per *spatial position* with the entropy router.

  • reason() returns one StepTrace per spatial token and backbone logits whose
    leading N positions correspond to those tokens.
  • Router thresholds steer which positions are LOCAL vs. "stopped to think":
    a huge `low` makes every position confident (uncertain_positions == []);
    a `low`=0 forces deliberation on all of them.
  • Fusing text_embeds prepends spatial tokens but only the spatial positions
    are scored.
  • The spatial encoder trains jointly with the backbone (gradient reaches its
    weights through a normal labelled forward — reason() itself is no_grad).

Tiny dims, fast under pytest.
"""
import warnings

import pytest
import torch

warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel
from mt_lnn.deliberation import Route, RouterThresholds
from mt_lnn.spatial_reasoning import SpatialReasoner, SpatialThinkingResult

D = 104


def _model(max_seq_len=64):
    cfg = MTLNNConfig(
        vocab_size=64, d_model=D, n_layers=2, n_heads=13, n_kv_heads=1,
        d_head=8, max_seq_len=max_seq_len, gwtb_n_heads=1, dropout=0.0,
        attention_dropout=0.0,
    )
    return MTLNNModel(cfg).eval()


def _reasoner(low=3.0, high=5.0):
    torch.manual_seed(0)
    m = _model()
    return SpatialReasoner(m, coord_dim=2, thresholds=RouterThresholds(low=low, high=high))


# --- construction ---------------------------------------------------------

def test_builds_encoder_from_model_d_model():
    m = _model()
    r = SpatialReasoner(m, coord_dim=2)
    # encoder emits backbone-width tokens
    tok = r.perceive(torch.rand(2, 4, 2))
    assert tok.shape == (2, 4, D)


# --- reason() basic shape contract ----------------------------------------

def test_reason_shapes_and_trace_length():
    r = _reasoner()
    coords = torch.rand(1, 5, 2)
    res = r.reason(coords)
    assert isinstance(res, SpatialThinkingResult)
    assert res.n_spatial == 5
    assert len(res.trace.steps) == 5
    assert res.logits.shape == (1, 5, 64)
    # every step is a spatial position (not a vocab token)
    assert all(s.token_id == -1 and s.token_text == "" for s in res.trace.steps)
    assert [s.index for s in res.trace.steps] == [0, 1, 2, 3, 4]


# --- routing is threshold-driven ------------------------------------------

def test_high_low_threshold_makes_all_local():
    r = _reasoner(low=1e9, high=2e9)          # everything below `low` → LOCAL
    res = r.reason(torch.rand(1, 6, 2))
    assert all(s.route == Route.LOCAL.value for s in res.trace.steps)
    assert res.uncertain_positions() == []


def test_zero_low_threshold_makes_all_deliberate():
    r = _reasoner(low=0.0, high=1e9)          # nothing below 0 → all SELF_CRITIQUE
    res = r.reason(torch.rand(1, 6, 2))
    assert all(s.route == Route.SELF_CRITIQUE.value for s in res.trace.steps)
    assert res.uncertain_positions() == [0, 1, 2, 3, 4, 5]
    # deliberation ran a self-consistency vote at each uncertain position
    assert all(s.n_resamples > 0 for s in res.trace.steps)
    assert all(s.sem_entropy is not None for s in res.trace.steps)


# --- text fusion: only spatial positions are scored -----------------------

def test_text_fusion_scores_only_spatial_positions():
    r = _reasoner()
    m = r.model
    ids = torch.randint(0, 64, (1, 7))
    text_embeds = m.embed_tokens(ids)
    coords = torch.rand(1, 4, 2)
    res = r.reason(coords, text_embeds=text_embeds)
    assert res.n_spatial == 4
    assert len(res.trace.steps) == 4                 # text positions not scored
    assert res.logits.shape == (1, 4 + 7, 64)        # but logits cover whole seq


# --- joint trainability of the encoder ------------------------------------

def test_encoder_trains_jointly_with_backbone():
    torch.manual_seed(1)
    m = _model()
    m.train()
    r = SpatialReasoner(m, coord_dim=2)
    tokens = r.perceive(torch.rand(1, 4, 2))         # (1, 4, D)
    labels = torch.randint(0, 64, (1, tokens.shape[1]))
    out = m(inputs_embeds=tokens, labels=labels)
    out["loss"].backward()
    g = r.encoder.mlp[0].weight.grad
    assert g is not None and g.abs().sum() > 0
    assert r.encoder.type_embed.grad is not None


# --- result helper --------------------------------------------------------

def test_uncertain_positions_subset_of_indices():
    r = _reasoner(low=3.0, high=5.0)
    res = r.reason(torch.rand(1, 5, 2))
    unc = res.uncertain_positions()
    assert set(unc).issubset(set(range(res.n_spatial)))
    # consistency: uncertain == positions whose route != LOCAL
    expected = [s.index for s in res.trace.steps if s.route != Route.LOCAL.value]
    assert unc == expected


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        try:
            fn()
            print(f"[ok] {fn.__name__}")
        except Exception:
            print(f"[FAIL] {fn.__name__}")
            traceback.print_exc()
            raise
