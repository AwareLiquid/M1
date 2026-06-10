"""
tests/test_causal_steering.py — CausalActivationSteerer (STARS-inspired).

Pins the contract of the inference-time causal steerer:

  • Gating: no correction while the trajectory is consistent (score ≥ floor),
    and a correction once a break drops the score below the floor.
  • Geometry: the correction is the ORTHOGONAL projection onto the legal
    subspace — at full strength the steered state has zero off-subspace
    residual; the in-subspace component is preserved.
  • Reuse: it reads the subspace from the SAME checker (no duplicated SVD), so
    detector and actuator agree.
  • Decoupling: zero learnable parameters; graceful no-ops on degenerate input.
  • Wiring: SpatialReasoner accepts an optional checker+steerer and stays
    backward-compatible when they are omitted.
"""
import warnings

import pytest
import torch

warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn.causality import CausalConsistencyChecker
from mt_lnn.causal_steering import CausalActivationSteerer, SteerResult


def _smooth_checker(direction, n=6, window=8):
    """A checker fed a smooth ramp along one direction → a 1-D legal subspace."""
    ck = CausalConsistencyChecker(window=window, method="subspace", ema_alpha=0.5)
    for t in range(1, n + 1):
        ck.update(direction * float(t))
    return ck


# --- gating ----------------------------------------------------------------

def test_no_steer_when_consistent():
    d = torch.zeros(16); d[0] = 1.0
    ck = _smooth_checker(d)
    st = CausalActivationSteerer(strength=1.0, floor=0.3)
    # A state continuing along the same direction stays consistent → no-op.
    res = st.steer(d * 7.0, ck)
    assert isinstance(res, SteerResult)
    assert not res.applied
    assert res.correction_norm == 0.0
    assert torch.equal(res.steered, d * 7.0)


def test_steer_fires_on_break_and_removes_off_subspace():
    d = torch.zeros(16); d[0] = 1.0
    ck = _smooth_checker(d)
    # A state with a large component ORTHOGONAL to the legal direction. We do
    # NOT feed it into the checker (that would absorb the off-subspace direction
    # into the legal subspace); we pass an explicit low score to open the gate.
    broken = d * 7.0 + 5.0 * torch.eye(16)[5]
    st = CausalActivationSteerer(strength=1.0, floor=0.5, adaptive=False)
    res = st.steer(broken, ck, score=0.1)
    assert res.applied
    assert res.correction_norm > 0
    # Full strength → steered state lies (almost) entirely in the legal subspace.
    basis = ck.principal_subspace()
    assert basis is not None
    Vk, mean = basis
    hc = res.steered.float() - mean
    recon = Vk.t() @ (Vk @ hc)
    residual = hc - recon
    assert residual.norm().item() < 1e-4          # off-subspace component gone
    # In-subspace component is preserved (we only removed the orthogonal part).
    assert recon.norm().item() > 1e-3


def test_partial_strength_leaves_some_residual():
    d = torch.zeros(16); d[0] = 1.0
    ck = _smooth_checker(d)
    broken = d * 7.0 + 5.0 * torch.eye(16)[5]
    full = CausalActivationSteerer(strength=1.0, floor=0.9, adaptive=False).steer(
        broken, ck, score=0.1)
    half = CausalActivationSteerer(strength=0.5, floor=0.9, adaptive=False).steer(
        broken, ck, score=0.1)
    assert half.applied and full.applied
    # Half strength removes less than full strength.
    assert 0 < half.correction_norm < full.correction_norm


# --- shape / dtype / device safety -----------------------------------------

def test_shape_and_dtype_preserved():
    d = torch.zeros(16); d[0] = 1.0
    ck = _smooth_checker(d)
    x = (d * 7.0 + torch.eye(16)[3]).to(torch.float64).reshape(1, 16)
    st = CausalActivationSteerer(strength=1.0, floor=0.9, adaptive=False)
    res = st.steer(x, ck)
    assert res.steered.shape == x.shape
    assert res.steered.dtype == x.dtype


def test_no_subspace_is_noop():
    ck = CausalConsistencyChecker(method="subspace")     # empty history
    st = CausalActivationSteerer(strength=1.0, floor=1.0)
    h = torch.randn(8)
    res = st.steer(h, ck, score=0.0)                     # force gate open
    assert not res.applied
    assert "no subspace" in res.reason
    assert torch.equal(res.steered, h)


def test_dimension_mismatch_is_noop():
    d = torch.zeros(16); d[0] = 1.0
    ck = _smooth_checker(d)
    st = CausalActivationSteerer(strength=1.0, floor=1.0)
    res = st.steer(torch.randn(32), ck, score=0.0)       # wrong D
    assert not res.applied
    assert "mismatch" in res.reason


def test_no_learnable_params_and_validation():
    st = CausalActivationSteerer()
    assert not hasattr(st, "parameters")                 # not an nn.Module
    with pytest.raises(ValueError):
        CausalActivationSteerer(strength=1.5)
    with pytest.raises(ValueError):
        CausalActivationSteerer(floor=-0.1)
    with pytest.raises(ValueError):
        CausalActivationSteerer(energy_keep=0.0)


# --- adaptive gain ---------------------------------------------------------

def test_adaptive_gain_scales_with_break_depth():
    d = torch.zeros(16); d[0] = 1.0
    ck = _smooth_checker(d)
    basis = ck.principal_subspace()
    assert basis is not None
    st = CausalActivationSteerer(strength=1.0, floor=0.4, adaptive=True)
    shallow = st.steer(d * 7.0, ck, score=0.35)   # just below floor
    deep = st.steer(d * 7.0, ck, score=0.0)        # deep break
    # Even with no off-subspace residual here, the gain itself should ramp.
    assert shallow.gain < deep.gain


# --- SpatialReasoner wiring (backward compat + optional use) ---------------

def test_spatial_reasoner_backward_compatible_and_optional_wiring():
    from mt_lnn.config import MTLNNConfig
    from mt_lnn.model import MTLNNModel
    from mt_lnn.spatial_reasoning import SpatialReasoner

    cfg = MTLNNConfig(
        vocab_size=64, d_model=104, n_layers=2, n_heads=13, n_kv_heads=1,
        d_head=8, max_seq_len=64, gwtb_n_heads=1, dropout=0.0,
        attention_dropout=0.0,
    )
    model = MTLNNModel(cfg).eval()
    coords = torch.rand(1, 5, 2)

    # Default (no checker/steerer): original behaviour.
    plain = SpatialReasoner(model, coord_dim=2)
    r0 = plain.reason(coords)
    assert r0.n_spatial == 5
    assert len(r0.trace.steps) == 5

    # With checker + steerer: still runs, produces the same number of steps.
    ck = CausalConsistencyChecker(method="subspace", window=8)
    st = CausalActivationSteerer(strength=1.0, floor=0.5)
    wired = SpatialReasoner(model, coord_dim=2, checker=ck, steerer=st)
    r1 = wired.reason(coords)
    assert r1.n_spatial == 5
    assert len(r1.trace.steps) == 5

    # steerer without checker is rejected (it needs the checker's subspace).
    with pytest.raises(ValueError):
        SpatialReasoner(model, coord_dim=2, steerer=st)


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
