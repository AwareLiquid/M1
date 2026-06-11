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
from mt_lnn.spatial_memory import SpatialMemory
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


def _reasoner_with_memory(low=3.0, high=5.0, n_place=256, arena=2.2):
    torch.manual_seed(0)
    m = _model()
    mem = SpatialMemory(d_model=D, n_place=n_place, arena=arena)
    return SpatialReasoner(
        m, coord_dim=2,
        thresholds=RouterThresholds(low=low, high=high),
        memory=mem,
    )


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


# --- L2 memory side-channel -----------------------------------------------

def test_memory_defaults_off_and_familiarity_is_none():
    # No memory attached → backward-compatible: no familiarity reported.
    r = _reasoner()
    assert r.memory is None
    res = r.reason(torch.rand(1, 5, 2))
    assert res.memory_familiarity is None
    # ... and the trace carries no familiarity note.
    assert all("mem:familiarity" not in s.reason for s in res.trace.steps)


def test_memory_adds_zero_parameters():
    # SpatialMemory is zero-parameter; attaching it must not grow the reasoner.
    r_plain = _reasoner()
    r_mem = _reasoner_with_memory()
    assert sum(p.numel() for p in r_mem.parameters()) == \
        sum(p.numel() for p in r_plain.parameters())


def test_remember_then_reason_reports_high_familiarity():
    r = _reasoner_with_memory()
    coords = torch.rand(1, 5, 2) * 2.0          # inside the arena
    r.remember(coords)
    res = r.reason(coords)
    assert res.memory_familiarity is not None
    assert len(res.memory_familiarity) == res.n_spatial
    # Each remembered place reads back its own content strongly.
    assert all(f > 0.8 for f in res.memory_familiarity)
    # ... and the per-position trace note surfaces it.
    assert all("mem:familiarity" in s.reason for s in res.trace.steps)


def test_novel_locations_have_low_familiarity():
    r = _reasoner_with_memory()
    written = torch.rand(1, 5, 2) * 2.0
    r.remember(written)
    # Query places far from anything written → no content recalled → ~0 cosine.
    novel = torch.full((1, 5, 2), -50.0)
    res = r.reason(novel)
    assert res.memory_familiarity is not None
    assert all(abs(f) < 0.2 for f in res.memory_familiarity)


def test_reason_is_read_only_occupancy_unchanged():
    r = _reasoner_with_memory()
    coords = torch.rand(1, 6, 2) * 2.0
    r.remember(coords)
    before = float(r.memory.occupancy.sum())
    r.reason(coords)
    r.reason(torch.rand(1, 6, 2) * 2.0)
    after = float(r.memory.occupancy.sum())
    assert after == pytest.approx(before)


def test_remember_and_recall_require_memory():
    r = _reasoner()                              # no memory attached
    with pytest.raises(ValueError):
        r.remember(torch.rand(1, 3, 2))
    with pytest.raises(ValueError):
        r.recall(torch.rand(1, 3, 2))


def test_recall_matches_remembered_content():
    r = _reasoner_with_memory()
    coords = torch.rand(1, 4, 2) * 2.0
    written = r.remember(coords)                 # returns perceived tokens
    recalled = r.recall(coords)
    assert recalled.shape == written.shape
    cos = torch.nn.functional.cosine_similarity(recalled, written, dim=-1)
    assert float(cos.mean()) > 0.9


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
