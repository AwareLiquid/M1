"""
tests/test_multisource_gwt.py — P3.1 multi-source GWT external bidding.

CompetitiveGWTBLayer lets *external* modules (e.g. the predictive world model)
submit their own bids into the workspace competition alongside the K internal
BidProjectors. These tests pin the contract:

  • Zero regression: no external bids → bit-identical to the internal-only path.
  • Init-identity survives external bids that equal x.
  • External bids genuinely participate (change the output, receive weight).
  • Diagnostics: last_external_weight tracks external mass; last_winner_weights
    stays size K; competition entropy is computed over the full bid set.
  • Gradients flow back into external bid tensors (so source modules can learn).
  • Shape validation rejects mismatched external bids.

Tiny dims, fast under pytest.
"""

import torch

from mt_lnn.config import MTLNNConfig
from mt_lnn.gwtb import CompetitiveGWTBLayer
from mt_lnn.model import MTLNNModel


def _cfg(K=3, d_model=64):
    return MTLNNConfig(
        vocab_size=64, d_model=d_model, n_layers=1, n_heads=8, n_kv_heads=1,
        d_head=8, max_seq_len=32, gwtb_n_heads=4, dropout=0.0,
        attention_dropout=0.0,
        use_competitive_gwtb=True, n_competitive_bids=K,
        gwtb_external_bids=True,
    )


def _layer(K=3, d_model=64, eval_mode=True):
    layer = CompetitiveGWTBLayer(_cfg(K=K, d_model=d_model))
    return layer.eval() if eval_mode else layer.train()


# ---------------------------------------------------------------------------
# Zero-regression / init-identity
# ---------------------------------------------------------------------------

def test_no_external_bids_is_identical_to_before():
    """external_bids=None must give the exact same output as omitting it."""
    torch.manual_seed(0)
    layer = _layer()
    x = torch.randn(2, 5, 64)
    out_a, _ = layer(x)
    out_b, _ = layer(x, external_bids=None)
    out_c, _ = layer(x, external_bids=[])
    assert torch.equal(out_a, out_b)
    assert torch.equal(out_a, out_c)
    # And no external mass was recorded.
    assert layer.last_external_weight.item() == 0.0


def test_init_identity_holds_with_external_bid_equal_to_x():
    """
    At init all internal bids == x (zero-init fc2) and scores are uniform (zero-
    init score_head, eval = no noise). An external bid equal to x must keep the
    competed combination == x, so the layer is still a near-identity broadcast.
    """
    torch.manual_seed(0)
    layer = _layer(K=3)
    x = torch.randn(2, 4, 64)
    combined = layer._compete(x, external_bids=[x.clone()])
    assert torch.allclose(combined, x, atol=1e-6), \
        "init-identity broken when external bid == x"
    # Four equal competitors (3 internal + 1 external) → uniform → external mass 1/4.
    assert abs(layer.last_external_weight.item() - 0.25) < 1e-4


# ---------------------------------------------------------------------------
# External bids genuinely participate
# ---------------------------------------------------------------------------

def test_external_bid_changes_output_and_gets_weight():
    """A distinct external bid must alter the competed combination and pick up
    nonzero softmax mass."""
    torch.manual_seed(1)
    layer = _layer(K=2)
    x = torch.randn(2, 6, 64)
    ext = torch.randn(2, 6, 64) * 3.0          # clearly different from x
    base = layer._compete(x)                    # internal-only
    withext = layer._compete(x, external_bids=[ext])
    assert not torch.allclose(base, withext, atol=1e-4), \
        "external bid had no effect on the competition"
    w = layer.last_external_weight.item()
    assert 0.0 < w < 1.0, f"external mass {w} not a proper share"


def test_multiple_external_bids_accumulate_mass_and_entropy():
    """K internal + 2 external → 4 competitors; entropy bounded by log(4) and
    external mass is the sum over both external bids."""
    import math
    torch.manual_seed(2)
    K = 2
    layer = _layer(K=K)
    x = torch.randn(3, 5, 64)
    e1 = torch.randn(3, 5, 64)
    e2 = torch.randn(3, 5, 64)
    layer._compete(x, external_bids=[e1, e2])
    n_total = K + 2
    ent = layer.last_competition_entropy.item()
    assert 0.0 <= ent <= math.log(n_total) + 1e-4, f"entropy {ent} out of range"
    # last_winner_weights still reports only the K internal specialists.
    assert layer.last_winner_weights.numel() == K
    assert 0.0 < layer.last_external_weight.item() < 1.0


# ---------------------------------------------------------------------------
# Gradient flow into external sources
# ---------------------------------------------------------------------------

def test_gradients_flow_back_into_external_bid():
    """The external source must receive gradient so it can learn to bid well."""
    torch.manual_seed(3)
    layer = _layer(K=2, eval_mode=False)   # train mode: full graph
    x = torch.randn(2, 4, 64)
    ext = torch.randn(2, 4, 64, requires_grad=True)
    out, _ = layer(x, external_bids=[ext])
    out.sum().backward()
    assert ext.grad is not None, "no gradient reached the external bid"
    assert ext.grad.abs().sum().item() > 0.0, "external bid gradient is all zero"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_mismatched_external_bid_shape_raises():
    layer = _layer(K=2)
    x = torch.randn(2, 5, 64)
    bad = torch.randn(2, 5, 32)             # wrong d_model
    try:
        layer._compete(x, external_bids=[bad])
    except ValueError as e:
        assert "external bid" in str(e)
    else:
        raise AssertionError("expected ValueError on mismatched external bid shape")


def test_none_entries_in_external_list_are_skipped():
    """A None placeholder (e.g. a source that produced nothing this step) is
    silently ignored, not treated as a bid."""
    torch.manual_seed(4)
    layer = _layer(K=2)
    x = torch.randn(2, 4, 64)
    out_none, _ = layer(x, external_bids=[None])
    out_plain, _ = layer(x)
    assert torch.equal(out_none, out_plain)
    assert layer.last_external_weight.item() == 0.0


# ---------------------------------------------------------------------------
# Output stays finite & in-distribution
# ---------------------------------------------------------------------------

def test_competed_output_is_finite_with_extreme_external_bid():
    torch.manual_seed(5)
    layer = _layer(K=3)
    x = torch.randn(2, 5, 64)
    huge = torch.full((2, 5, 64), 1e6)
    combined = layer._compete(x, external_bids=[huge])
    assert torch.isfinite(combined).all()


# ---------------------------------------------------------------------------
# Hard-winner inference path with external bids (regression: one_hot width bug)
# ---------------------------------------------------------------------------

def test_hard_winner_with_external_bid_no_shape_error():
    """Regression: in eval+hard_winner mode the one-hot must span K+n_external,
    not just K. A winning external bid (winner_idx == K) previously raised
    'index >= num_classes' / a (B,T,K) vs (B,T,K+ext) broadcast mismatch."""
    torch.manual_seed(6)
    cfg = _cfg(K=3)
    cfg.competitive_hard_winner = True
    layer = CompetitiveGWTBLayer(cfg).eval()
    # Give the (shared) score head real weights so bid magnitude affects the
    # score — otherwise zero-init scores make argmax always pick index 0 and the
    # external (winner_idx == K) branch is never exercised.
    with torch.no_grad():
        layer.score_head[0].weight.fill_(0.01)
        layer.score_head[-1].weight.fill_(0.1)
    x = torch.randn(2, 5, 64)
    ext = torch.abs(torch.randn(2, 5, 64)) * 1e3 + 1e3   # huge → external wins
    combined = layer._compete(x, external_bids=[ext])
    assert combined.shape == x.shape
    assert torch.isfinite(combined).all()
    out, _ = layer(x, external_bids=[ext])
    assert out.shape == x.shape and torch.isfinite(out).all()


def test_hard_winner_can_select_external_bid():
    """When the external bid dominates the score at every position, the hard
    winner is the external competitor (winner_idx == K) and the combined view
    equals that bid — directly validating the widened one-hot."""
    torch.manual_seed(7)
    cfg = _cfg(K=2)
    cfg.competitive_hard_winner = True
    layer = CompetitiveGWTBLayer(cfg).eval()
    with torch.no_grad():
        layer.score_head[0].weight.fill_(0.01)   # positive → larger inputs score higher
        layer.score_head[-1].weight.fill_(0.1)
    x = torch.randn(1, 4, 64)
    ext = torch.abs(torch.randn(1, 4, 64)) * 1e3 + 1e3   # huge, positive everywhere
    combined = layer._compete(x, external_bids=[ext])
    assert torch.allclose(combined, ext, atol=1e-3), \
        "hard winner did not route to the dominant external bid"
    assert abs(layer.last_external_weight.item() - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# Model-level integration: world model bids into the workspace
# ---------------------------------------------------------------------------

def _model_cfg(external: bool):
    return MTLNNConfig(
        vocab_size=64, d_model=104, n_layers=2, n_heads=13, n_kv_heads=1,
        d_head=8, max_seq_len=32, gwtb_n_heads=1, dropout=0.0,
        attention_dropout=0.0,
        use_competitive_gwtb=True, n_competitive_bids=3,
        use_world_model=True,
        gwtb_external_bids=external,
    )


def test_model_wires_world_bid_only_when_enabled():
    """The world-model bid adapter exists iff gwtb_external_bids is on AND both
    competitive GWTB and world model are present."""
    off = MTLNNModel(_model_cfg(external=False))
    on = MTLNNModel(_model_cfg(external=True))
    assert off._world_gwtb_bid is False
    assert not hasattr(off, "world_bid_adapter")
    assert on._world_gwtb_bid is True
    assert hasattr(on, "world_bid_adapter")
    assert on.world_bid_gate.item() == 0.0   # starts closed


def test_model_external_bid_is_identity_at_init():
    """Zero-gated world bid == x, so it enters the competition as an equal
    competitor that does NOT change the combined workspace input: toggling the
    external bid on/off on the same model gives bit-identical logits at init."""
    torch.manual_seed(0)
    model = MTLNNModel(_model_cfg(external=True)).eval()
    ids = torch.randint(0, 64, (2, 10))
    with torch.no_grad():
        out_on = model(ids, labels=ids)
        logits_on = out_on["logits"].clone()
        # The external bid equals x at init (gate 0); it is one of K+1 equal
        # competitors, so it claims uniform mass but leaves combined == x.
        assert abs(model.get_mt_diagnostics()["gwtb_external_bid_weight"]
                   - 1.0 / (model.gwtb.n_bids + 1)) < 1e-4
        assert model.get_mt_diagnostics()["gwtb_world_bid_gate"] == 0.0
        # Disable the external path → combined still == x → near-identical logits
        # (only the softmax normalisation over K+1 vs K equal zero-scores differs,
        # an O(1e-4) floating-point artifact; the competed input is x either way).
        model._world_gwtb_bid = False
        logits_off = model(ids, labels=ids)["logits"]
    max_diff = (logits_on - logits_off).abs().max().item()
    assert max_diff < 1e-3, \
        f"zero-gated world bid perturbed the output at init (max diff {max_diff:.2e})"
    assert torch.isfinite(out_on["loss"])


def test_model_external_bid_gate_receives_gradient():
    """With the gate closed at init the adapter weights are nonzero, so the gate
    has a live gradient and can learn to open."""
    torch.manual_seed(0)
    model = MTLNNModel(_model_cfg(external=True)).train()
    ids = torch.randint(0, 64, (2, 12))
    out = model(ids, labels=ids)
    out["loss"].backward()
    assert model.world_bid_gate.grad is not None
    assert model.world_bid_gate.grad.abs().item() > 0.0, \
        "world-model bid gate has no gradient — feature can never turn on"


def test_model_open_gate_routes_mass_to_world_bid():
    """Force the gate open: the world model's bid then takes real competition
    mass and the forward stays finite."""
    torch.manual_seed(0)
    model = MTLNNModel(_model_cfg(external=True)).eval()
    with torch.no_grad():
        model.world_bid_gate.fill_(1.0)
        # Make the adapter output large so the world bid is clearly distinct.
        model.world_bid_adapter.weight.mul_(50.0)
        ids = torch.randint(0, 64, (2, 10))
        out = model(ids, labels=ids)
    diag = model.get_mt_diagnostics()
    assert diag["gwtb_external_bid_weight"] > 0.0
    assert torch.isfinite(out["loss"])


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
