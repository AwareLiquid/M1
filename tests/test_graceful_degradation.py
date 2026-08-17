"""
tests/test_graceful_degradation.py — P3.2 module graceful degradation.

A multi-day pre-training run must survive a transient NaN/Inf in any of the
bio-inspired auxiliary modules. The model wraps every AUXILIARY contribution
(world-model loss, GWTB orthogonality penalty, Hebbian loss, the world-model
workspace bid, the LAVI surprise signal) in a finiteness guard: a faulty term is
dropped for that step (and counted) instead of poisoning the main loss. The
PRIMARY cross-entropy is never guarded.

These tests pin:
  • the guard helper (_aux_or_skip) semantics,
  • each integration point degrades instead of crashing,
  • the degradation counter / diagnostics surface the events,
  • zero-regression: healthy runs trigger nothing,
  • the OFF switch really lets faults propagate (so the guard is load-bearing).

Tiny dims, fast under pytest.
"""

import types

import torch

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel


def _cfg(graceful=True, external=False):
    return MTLNNConfig(
        vocab_size=64, d_model=104, n_layers=2, n_heads=13, n_kv_heads=1,
        d_head=8, max_seq_len=32, gwtb_n_heads=1, dropout=0.0,
        attention_dropout=0.0,
        use_competitive_gwtb=True, n_competitive_bids=3,
        use_world_model=True, use_hebbian=True,
        gwtb_external_bids=external,
        use_graceful_degradation=graceful,
    )


def _nan():
    return torch.tensor(float("nan"))


# ---------------------------------------------------------------------------
# Guard helper unit semantics
# ---------------------------------------------------------------------------

def test_aux_or_skip_passes_finite_and_drops_nonfinite():
    m = MTLNNModel(_cfg())
    finite = torch.tensor(1.5)
    assert m._aux_or_skip("a", finite) is finite
    assert "a" not in m._degradation_counts
    assert m._aux_or_skip("b", _nan()) is None
    assert m._degradation_counts["b"] == 1
    assert m._aux_or_skip("b", torch.tensor(float("inf"))) is None
    assert m._degradation_counts["b"] == 2          # accumulates per module
    assert m._aux_or_skip("c", None) is None        # None passes through, no count
    assert "c" not in m._degradation_counts


def test_aux_or_skip_is_noop_when_disabled():
    m = MTLNNModel(_cfg(graceful=False))
    nan = _nan()
    # Disabled → returns the value untouched (caller will then propagate the NaN).
    assert m._aux_or_skip("a", nan) is nan
    assert m._degradation_counts == {}


# ---------------------------------------------------------------------------
# Integration: world-model loss fault
# ---------------------------------------------------------------------------

def _inject_nan_world_loss(model):
    proj_dim = model.world_model_head.proj_dim

    def nan_forward(self, x, compute_loss=True):
        z = torch.zeros(x.shape[0], x.shape[1], proj_dim)
        return z, _nan()

    model.world_model_head.forward = types.MethodType(nan_forward, model.world_model_head)


def test_world_model_nan_loss_is_skipped_main_loss_finite():
    torch.manual_seed(0)
    model = MTLNNModel(_cfg()).train()
    _inject_nan_world_loss(model)
    ids = torch.randint(0, 64, (2, 10))
    out = model(ids, labels=ids)
    assert torch.isfinite(out["loss"]), "world-model NaN poisoned the main loss"
    assert "world_model_loss" not in out          # the bad term was dropped
    assert model._degradation_counts.get("world_model_loss", 0) >= 1


def test_guarded_nan_does_not_corrupt_gradients():
    """After a guarded fault, backward must produce finite grads (training lives)."""
    torch.manual_seed(0)
    model = MTLNNModel(_cfg()).train()
    _inject_nan_world_loss(model)
    ids = torch.randint(0, 64, (2, 10))
    out = model(ids, labels=ids)
    out["loss"].backward()
    bad = [n for n, p in model.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite gradients leaked through the guard: {bad[:3]}"


def test_disabled_degradation_lets_world_model_nan_propagate():
    """With the guard OFF the fault reaches the main loss — proves it's load-bearing."""
    torch.manual_seed(0)
    model = MTLNNModel(_cfg(graceful=False)).train()
    _inject_nan_world_loss(model)
    ids = torch.randint(0, 64, (2, 10))
    out = model(ids, labels=ids)
    assert not torch.isfinite(out["loss"]), \
        "guard disabled but NaN did NOT propagate — the guard isn't doing anything"


# ---------------------------------------------------------------------------
# Integration: Hebbian loss fault
# ---------------------------------------------------------------------------

def test_hebbian_nan_loss_is_skipped():
    torch.manual_seed(0)
    model = MTLNNModel(_cfg()).train()
    model.hebbian_reg.compute_loss = types.MethodType(
        lambda self, m: _nan(), model.hebbian_reg
    )
    ids = torch.randint(0, 64, (2, 10))
    out = model(ids, labels=ids)
    assert torch.isfinite(out["loss"])
    assert "hebbian_loss" not in out
    assert model._degradation_counts.get("hebbian_loss", 0) >= 1


# ---------------------------------------------------------------------------
# Integration: world-model bid fault (P3.1 × P3.2)
# ---------------------------------------------------------------------------

def test_world_bid_nan_falls_back_to_raw_state():
    torch.manual_seed(0)
    model = MTLNNModel(_cfg(external=True)).eval()
    with torch.no_grad():
        model.world_bid_gate.fill_(1.0)
        model.world_bid_adapter.weight.fill_(float("nan"))   # bid → NaN
        ids = torch.randint(0, 64, (2, 10))
        out = model(ids, labels=ids)
    assert torch.isfinite(out["loss"]), "NaN world bid was not contained"
    assert model._degradation_counts.get("world_bid", 0) >= 1


# ---------------------------------------------------------------------------
# Diagnostics + zero-regression
# ---------------------------------------------------------------------------

def test_degradation_events_surface_in_diagnostics():
    torch.manual_seed(0)
    model = MTLNNModel(_cfg()).train()
    _inject_nan_world_loss(model)
    ids = torch.randint(0, 64, (2, 10))
    model(ids, labels=ids)
    diag = model.get_mt_diagnostics()
    assert diag.get("degradation_events_total", 0) >= 1
    assert diag.get("degradation_world_model_loss", 0) >= 1


def test_healthy_run_triggers_no_degradation():
    """Zero-regression: a normal forward never fires a guard, counters stay empty,
    and the diagnostic key is absent."""
    torch.manual_seed(0)
    model = MTLNNModel(_cfg(external=True)).train()
    ids = torch.randint(0, 64, (2, 12))
    out = model(ids, labels=ids)
    assert torch.isfinite(out["loss"])
    assert model._degradation_counts == {}
    assert "degradation_events_total" not in model.get_mt_diagnostics()


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
