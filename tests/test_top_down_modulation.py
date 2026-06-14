"""
tests/test_top_down_modulation.py -- top-down modulation contract (P1 closed-loop ②).

Pins the model.py change that threads a high-level goal/context vector
``top_down`` through MTLNNModel.forward -> every MTLNNBlock, where it biases the
representation via a ZERO-INIT-GATED residual adapter (injected after attention,
before the LNN sub-layer). The whole feature is gated by ``config.use_top_down``
and is default-OFF, so the default model is byte-identical (zero regression).

Guarantees pinned here:
  * use_top_down=False (default) -> no new params, passing top_down is a no-op.
  * use_top_down=True at init (gate=0) -> per-block path is BIT-EXACT (x + 0*proj).
  * opening the gate changes the output (the pathway is live).
  * gradients reach top_down_proj and top_down_gate.
  * (B, d_model) broadcast and (B, T, d_model) per-step inputs both work.
  * KV-cache decoding stays consistent with full-context forward.
  * GWT docking entry (top_down_to_gwtb) offers the goal as a workspace bid,
    identity-at-init to within the same O(1e-4) softmax artifact as the world
    bid, and changes the competition once its gate opens.
"""
import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mt_lnn.config import MTLNNConfig                                   # noqa: E402
from mt_lnn.model import MTLNNModel                                     # noqa: E402

D = 104   # divisible by n_heads=13 (microtubule protofilament count)


def _cfg(**over):
    base = dict(
        vocab_size=64, d_model=D, n_layers=2, n_heads=13, n_kv_heads=1,
        d_head=8, max_seq_len=64, gwtb_n_heads=1, dropout=0.0,
        attention_dropout=0.0,
    )
    base.update(over)
    return MTLNNConfig(**base)


def _model(**over):
    return MTLNNModel(_cfg(**over)).eval()


# --- default-off: zero regression -----------------------------------------

def test_disabled_by_default_builds_no_params_and_ignores_top_down():
    m = _model()                                          # use_top_down defaults False
    assert not any("top_down" in n for n, _ in m.named_parameters())
    for blk in m.blocks:
        assert blk.has_top_down is False
        assert blk.top_down_proj is None
    ids = torch.randint(0, 64, (2, 6))
    with torch.no_grad():
        a = m(input_ids=ids)["logits"]
        b = m(input_ids=ids, top_down=torch.randn(2, D))["logits"]  # silently ignored
    assert torch.equal(a, b)


def test_enabling_adds_per_block_adapter_params():
    m = _model(use_top_down=True)
    names = [n for n, _ in m.named_parameters() if "top_down" in n]
    # per block: norm (weight+bias), proj weight, gate -> present for every layer
    assert any(n.endswith("top_down_gate") for n in names)
    assert any(n.endswith("top_down_proj.weight") for n in names)
    for blk in m.blocks:
        assert blk.has_top_down is True
        assert float(blk.top_down_gate) == 0.0           # starts closed


# --- per-block zero-gated identity at init --------------------------------

def test_zero_gated_is_bit_exact_at_init():
    torch.manual_seed(0)
    m = _model(use_top_down=True)
    ids = torch.randint(0, 64, (2, 6))
    td = torch.randn(2, D)
    with torch.no_grad():
        none = m(input_ids=ids, top_down=None)["logits"]
        given = m(input_ids=ids, top_down=td)["logits"]   # gate=0 -> identity
    assert torch.equal(none, given)


def test_opening_the_gate_changes_the_output():
    torch.manual_seed(0)
    m = _model(use_top_down=True)
    ids = torch.randint(0, 64, (2, 6))
    td = torch.randn(2, D)
    with torch.no_grad():
        closed = m(input_ids=ids, top_down=td)["logits"]
        for blk in m.blocks:
            blk.top_down_gate.data.fill_(1.0)
        opened = m(input_ids=ids, top_down=td)["logits"]
    assert not torch.allclose(opened, closed, atol=1e-5)


# --- learning path --------------------------------------------------------

def test_gate_has_live_gradient_at_init():
    """At init (gate=0) the residual is tanh(0)*proj == 0, so the GATE still gets a
    live gradient (proj is small-but-nonzero) -- this is what lets the model LEARN
    to open it. The proj weight is gated to zero contribution at init, so it only
    receives gradient once the gate opens (verified separately below)."""
    torch.manual_seed(0)
    m = MTLNNModel(_cfg(use_top_down=True)).train()
    ids = torch.randint(0, 64, (2, 8))
    td = torch.randn(2, D, requires_grad=True)
    out = m(input_ids=ids, labels=ids, top_down=td)
    out["loss"].backward()
    blk = m.blocks[0]
    assert blk.top_down_gate.grad is not None
    assert float(blk.top_down_gate.grad.abs().sum()) > 0.0     # gate can learn to open
    # td flows only through the gated residual, so at init its gradient is gated to
    # zero (it starts flowing once the gate opens -- see the next test).


def test_proj_and_goal_learn_once_the_gate_is_open():
    torch.manual_seed(0)
    m = MTLNNModel(_cfg(use_top_down=True)).train()
    for blk in m.blocks:
        blk.top_down_gate.data.fill_(1.0)                     # open the gate
    ids = torch.randint(0, 64, (2, 8))
    td = torch.randn(2, D, requires_grad=True)
    out = m(input_ids=ids, labels=ids, top_down=td)
    out["loss"].backward()
    g = m.blocks[0].top_down_proj.weight.grad
    assert g is not None and float(g.abs().sum()) > 0.0
    # with the gate open the external goal source now receives gradient too
    assert td.grad is not None and float(td.grad.abs().sum()) > 0.0


# --- input shapes ---------------------------------------------------------

def test_broadcast_and_per_step_shapes_both_work():
    torch.manual_seed(0)
    m = _model(use_top_down=True)
    for blk in m.blocks:
        blk.top_down_gate.data.fill_(0.5)                 # open so shape path matters
    ids = torch.randint(0, 64, (2, 6))
    with torch.no_grad():
        a = m(input_ids=ids, top_down=torch.randn(2, D))["logits"]        # (B, d)
        b = m(input_ids=ids, top_down=torch.randn(2, 6, D))["logits"]     # (B, T, d)
    assert a.shape == b.shape == (2, 6, 64)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()


# --- cache parity ---------------------------------------------------------

def test_kv_cache_decoding_matches_full_forward_with_top_down():
    torch.manual_seed(0)
    m = _model(use_top_down=True)
    for blk in m.blocks:
        blk.top_down_gate.data.fill_(0.7)                 # exercise the live path
    B, T = 1, 8
    ids = torch.randint(0, 64, (B, T))
    td = torch.randn(B, D)                                # shared goal across steps

    with torch.no_grad():
        full = m(input_ids=ids, top_down=td, use_lnn_recurrence=False)["logits"]
        cache = None
        outs = []
        for t in range(T):
            o = m(input_ids=ids[:, t:t + 1], cache=cache, top_down=td,
                  use_cache=True, use_lnn_recurrence=False)
            cache = o["cache"]
            outs.append(o["logits"])
        cached = torch.cat(outs, dim=1)
    assert (full - cached).abs().max().item() < 1e-3


# --- GWT docking entry ----------------------------------------------------

def _gwt_cfg(**over):
    return dict(
        gwtb_per_block=False, use_competitive_gwtb=True,
        gwtb_external_bids=True, use_top_down=True,
    ) | over


def test_docking_requires_competitive_gwtb():
    # competitive GWTB present -> docking wires up
    on = _model(**_gwt_cfg(top_down_to_gwtb=True))
    assert on._top_down_gwtb_bid is True
    assert hasattr(on, "top_down_gwtb_gate")
    # non-competitive top-level GWTB -> docking stays inert (no params)
    off = _model(use_top_down=True, top_down_to_gwtb=True,
                 gwtb_per_block=False, use_competitive_gwtb=False)
    assert off._top_down_gwtb_bid is False
    assert not hasattr(off, "top_down_gwtb_gate")


def test_docking_is_identity_at_init_within_softmax_artifact():
    # zero-gated bid == x at init -> toggling top_down on/off on the same model
    # perturbs logits only at the world-bid-level O(1e-4) softmax artifact.
    torch.manual_seed(0)
    m = _model(**_gwt_cfg(top_down_to_gwtb=True))
    assert float(m.top_down_gwtb_gate) == 0.0
    ids = torch.randint(0, 64, (2, 10))
    td = torch.randn(2, D)
    with torch.no_grad():
        on = m(input_ids=ids, top_down=td)["logits"]
        off = m(input_ids=ids, top_down=None)["logits"]
    assert (on - off).abs().max().item() < 1e-3


def test_docking_changes_competition_once_its_gate_opens():
    torch.manual_seed(0)
    m = _model(**_gwt_cfg(top_down_to_gwtb=True))
    ids = torch.randint(0, 64, (2, 10))
    td = torch.randn(2, D)
    with torch.no_grad():
        closed = m(input_ids=ids, top_down=td)["logits"]
        m.top_down_gwtb_gate.data.fill_(1.5)
        opened = m(input_ids=ids, top_down=td)["logits"]
        # the workspace now places real mass on the top-down bid
        assert float(m.gwtb.last_external_weight) > 0.0
    assert not torch.allclose(opened, closed, atol=1e-4)
