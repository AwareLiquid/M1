"""
tests/test_memory_gwt_bid.py -- Gap 1: synaptic memory -> global workspace.

Closes the "spatial/Hebbian associative memory -> global workspace" chain by
letting a content-addressed FastWeightMemory recall stream bid into the top-level
CompetitiveGWTBLayer competition, JOINTLY trained with the main CE. This is the
end-to-end wiring step (mechanism Built + jointly trainable); whether it MOVES
held-out perplexity is a separate effectiveness question (the GPU ablation).

The bid reuses, without duplication:
  * the FastWeightMemory primitive built for the served adapter
    (mt_lnn.llama_adapter.FastWeightMemory), and
  * the same zero-init-gated residual-bid machinery as the world-model bid.

Contract pinned here (mirrors tests/test_multisource_gwt.py for the world bid):
  1. wiring  -- the memory-bid path is built iff enabled AND the workspace is a
     CompetitiveGWTBLayer that accepts external bids; the gate starts closed.
  2. zero regression at init -- gate 0 => bid == x => toggling it on/off leaves
     the logits near-identical (only the K+1 softmax-normalisation artifact).
  3. joint trainability -- with the gate closed at init the recall projection is
     nonzero, so the gate AND every FastWeightMemory projection receive gradient
     from the main CE (the chain is genuinely trained end-to-end, not a dead path).
  4. an opened gate routes real competition mass to the memory bid, finitely.
  5. memory and world bids coexist as two independent external competitors.
"""

import torch

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel


def _model_cfg(memory_bid: bool, world_model: bool = False):
    return MTLNNConfig(
        vocab_size=64, d_model=104, n_layers=2, n_heads=13, n_kv_heads=1,
        d_head=8, max_seq_len=32, gwtb_n_heads=1, dropout=0.0,
        attention_dropout=0.0,
        use_competitive_gwtb=True, n_competitive_bids=3,
        gwtb_external_bids=True,
        use_world_model=world_model,
        gwtb_memory_bid=memory_bid,
        gwtb_memory_bid_dim=16, gwtb_memory_bid_heads=2,
    )


# ---------------------------------------------------------------------------
# 1. Wiring
# ---------------------------------------------------------------------------

def test_model_wires_memory_bid_only_when_enabled():
    """The memory-bid path exists iff gwtb_memory_bid is on AND the workspace is
    a CompetitiveGWTBLayer that accepts external bids."""
    off = MTLNNModel(_model_cfg(memory_bid=False))
    on = MTLNNModel(_model_cfg(memory_bid=True))
    assert off._memory_gwtb_bid is False
    assert not hasattr(off, "memory_fast_weight")
    assert on._memory_gwtb_bid is True
    assert hasattr(on, "memory_fast_weight")
    assert on.memory_bid_gate.item() == 0.0   # starts closed


def test_memory_bid_requires_external_bids_enabled():
    """Without gwtb_external_bids the competitive workspace does not accept
    external bids, so the memory bid must NOT wire (no silent dead path)."""
    cfg = _model_cfg(memory_bid=True)
    cfg.gwtb_external_bids = False
    model = MTLNNModel(cfg)
    assert model._memory_gwtb_bid is False
    assert not hasattr(model, "memory_fast_weight")


# ---------------------------------------------------------------------------
# 2. Zero regression at init
# ---------------------------------------------------------------------------

def test_memory_bid_is_identity_at_init():
    """Zero-gated memory bid == x, so it competes as an equal competitor that does
    NOT change the combined workspace input: toggling it on/off at init gives
    near-identical logits (only the K+1 vs K softmax artifact)."""
    torch.manual_seed(0)
    model = MTLNNModel(_model_cfg(memory_bid=True)).eval()
    ids = torch.randint(0, 64, (2, 10))
    with torch.no_grad():
        out_on = model(ids, labels=ids)
        logits_on = out_on["logits"].clone()
        assert model.get_mt_diagnostics()["gwtb_memory_bid_gate"] == 0.0
        # Disable the memory path -> combined still == x -> near-identical logits.
        model._memory_gwtb_bid = False
        logits_off = model(ids, labels=ids)["logits"]
    max_diff = (logits_on - logits_off).abs().max().item()
    assert max_diff < 1e-3, \
        f"zero-gated memory bid perturbed the output at init (max diff {max_diff:.2e})"
    assert torch.isfinite(out_on["loss"])


# ---------------------------------------------------------------------------
# 3. Joint trainability (the Gap-1 claim: end-to-end trained, not a dead path)
# ---------------------------------------------------------------------------

def test_memory_bid_gate_receives_gradient_at_init():
    """With the gate closed at init the recall projection is still nonzero, so the
    gate itself has a live gradient from the main CE -- the path can LEARN to open
    (exactly the zero-gated-residual warmup the world-model bid uses). The inner
    memory weights are multiplied by gate==0 at init, so they only start training
    once the gate cracks open (verified separately under an open gate)."""
    torch.manual_seed(0)
    model = MTLNNModel(_model_cfg(memory_bid=True)).train()
    ids = torch.randint(0, 64, (2, 12))
    out = model(ids, labels=ids)
    out["loss"].backward()
    assert model.memory_bid_gate.grad is not None
    assert model.memory_bid_gate.grad.abs().item() > 0.0, \
        "memory bid gate has no gradient -- the memory can never turn on"


def test_open_gate_trains_every_memory_projection():
    """Once the gate is open the spatial/synaptic-memory -> workspace chain is
    genuinely jointly trained end-to-end: every FastWeightMemory projection
    (write/read/value/out + the learnable decay) receives gradient from the main
    CE -- the memory is not a dead path, it learns through the bid competition."""
    torch.manual_seed(0)
    model = MTLNNModel(_model_cfg(memory_bid=True)).train()
    with torch.no_grad():
        model.memory_bid_gate.fill_(1.0)   # open the gate so the memory trains
    ids = torch.randint(0, 64, (2, 12))
    out = model(ids, labels=ids)
    out["loss"].backward()
    for name, p in model.memory_fast_weight.named_parameters():
        assert p.grad is not None and p.grad.abs().sum().item() > 0.0, \
            f"FastWeightMemory.{name} got no gradient under an open gate -- dead path"


# ---------------------------------------------------------------------------
# 4. Opened gate routes real mass to the memory bid
# ---------------------------------------------------------------------------

def test_open_gate_routes_mass_to_memory_bid():
    """Force the gate open and amplify the recall: the memory bid then takes real
    competition mass and the forward stays finite."""
    torch.manual_seed(0)
    model = MTLNNModel(_model_cfg(memory_bid=True)).eval()
    with torch.no_grad():
        model.memory_bid_gate.fill_(1.0)
        model.memory_fast_weight.W_o.weight.mul_(50.0)   # make the bid distinct
        ids = torch.randint(0, 64, (2, 10))
        out = model(ids, labels=ids)
    diag = model.get_mt_diagnostics()
    assert diag["gwtb_external_bid_weight"] > 0.0
    assert torch.isfinite(out["loss"])


# ---------------------------------------------------------------------------
# 5. Memory + world bids coexist (two independent external competitors)
# ---------------------------------------------------------------------------

def test_memory_and_world_bids_coexist():
    """Both external sources wire independently and the forward stays finite with
    two external competitors in the workspace."""
    torch.manual_seed(0)
    model = MTLNNModel(_model_cfg(memory_bid=True, world_model=True)).train()
    assert model._memory_gwtb_bid is True
    assert model._world_gwtb_bid is True
    ids = torch.randint(0, 64, (2, 10))
    out = model(ids, labels=ids)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    # Both gates are live and independently trainable.
    assert model.memory_bid_gate.grad.abs().item() > 0.0
    assert model.world_bid_gate.grad.abs().item() > 0.0


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
