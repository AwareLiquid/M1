"""
tests/test_sensory_frontend.py -- raw streaming-sensor frontend contract.

Pins ``mt_lnn/sensory_frontend.py``: a jittered, timestamped raw stream is
aligned onto the core's fixed-dt grid (linear resampling exact on affine signals),
dropouts are flagged in the coverage / pad mask, the projected tokens have the
backbone's d_model width and feed ``MTLNNModel.forward(inputs_embeds=...)``, and
gradients reach only the trainable projector (the alignment is parameter-free).
"""
import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mt_lnn.sensory_frontend import SensoryEncoding, SensoryFrontend     # noqa: E402
from mt_lnn.model import MTLNNConfig, MTLNNModel                          # noqa: E402

_F = torch.float32
D = 104   # divisible by n_heads=13 (microtubule protofilament count)


def _model(max_seq_len=64):
    cfg = MTLNNConfig(
        vocab_size=64, d_model=D, n_layers=2, n_heads=13, n_kv_heads=1,
        d_head=8, max_seq_len=max_seq_len, gwtb_n_heads=1, dropout=0.0,
        attention_dropout=0.0,
    )
    return MTLNNModel(cfg).eval()


# --- shapes / regimes -----------------------------------------------------

def test_aligned_forward_shapes_and_pad_mask_alias():
    fe = SensoryFrontend(in_dim=4, d_model=D, dt=0.1)
    ts = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4])
    enc = fe(torch.randn(2, 5, 4), ts, n_steps=6, t_start=0.0)
    assert isinstance(enc, SensoryEncoding)
    assert enc.embeds.shape == (2, 6, D)
    assert enc.covered.shape == (2, 6)
    assert bool((enc.pad_mask == enc.covered).all())     # pad_mask is the alias
    assert enc.n_steps == 6


def test_uniform_regime_skips_alignment_all_covered():
    fe = SensoryFrontend(in_dim=4, d_model=D, dt=0.1)
    enc = fe(torch.randn(3, 7, 4))                        # timestamps=None
    assert enc.embeds.shape == (3, 7, D)
    assert enc.fully_covered
    assert enc.coverage_ratio == 1.0


def test_unbatched_input_squeezes_batch_dim():
    fe = SensoryFrontend(in_dim=4, d_model=D, dt=0.1)
    ts = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4])
    enc = fe(torch.randn(5, 4), ts, n_steps=6)
    assert enc.embeds.shape == (6, D)                     # (M, d_model), no batch


# --- temporal correctness -------------------------------------------------

def test_linear_resample_is_exact_on_affine_signal():
    fe = SensoryFrontend(in_dim=3, d_model=D, dt=0.1)
    ts = torch.tensor([0.0, 0.1, 0.2, 0.55, 0.65], dtype=_F)
    slope = torch.arange(3, dtype=_F) + 1.0
    lin = ts.unsqueeze(-1) * slope                        # exactly affine in t
    a = fe.align(lin, ts, n_steps=7, t_start=0.0)
    exact = a.times.unsqueeze(-1) * slope
    err = (a.values - exact)[a.covered].abs().max()
    assert float(err) < 1e-5                              # exact on covered steps


def test_wide_dropout_is_flagged_in_coverage():
    fe = SensoryFrontend(in_dim=3, d_model=D, dt=0.1)
    ts = torch.tensor([0.0, 0.1, 0.2, 1.0, 1.1])          # big hole 0.2 -> 1.0
    enc = fe(torch.randn(2, 5, 3), ts, n_steps=12, t_start=0.0)
    assert not enc.fully_covered
    assert enc.n_gap_steps > 0
    assert enc.coverage_ratio < 1.0
    # the hole's interior steps are flagged; the endpoints are trusted
    assert bool(enc.covered[0, 0]) and not bool(enc.covered[0, 5])


def test_shared_clock_coverage_identical_across_batch():
    fe = SensoryFrontend(in_dim=3, d_model=D, dt=0.1)
    ts = torch.tensor([0.0, 0.1, 0.2, 1.0, 1.1])
    enc = fe(torch.randn(4, 5, 3), ts, n_steps=12)
    assert bool((enc.covered == enc.covered[0:1]).all())  # one shared clock


# --- end-to-end injection into the backbone -------------------------------

def test_embeds_feed_the_backbone():
    torch.manual_seed(0)
    m = _model()
    fe = SensoryFrontend(in_dim=5, d_model=D, dt=0.1)
    ts = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    enc = fe(torch.randn(2, 6, 5), ts, n_steps=8, t_start=0.0)
    with torch.no_grad():
        out = m(inputs_embeds=enc.embeds, pad_mask=enc.pad_mask, use_cache=True)
    assert out["logits"].shape == (2, enc.n_steps, 64)
    assert torch.isfinite(out["logits"]).all()


# --- learning path --------------------------------------------------------

def test_gradients_reach_only_the_projector():
    fe = SensoryFrontend(in_dim=4, d_model=D, dt=0.1)
    ts = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4])
    enc = fe(torch.randn(2, 5, 4), ts, n_steps=6)
    enc.embeds.sum().backward()
    g = fe.projector.proj.weight.grad
    assert g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0


# --- validation -----------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"dt": 0.0},                       # dt must be > 0
    {"dt": 0.1, "max_gap": -1.0},      # max_gap must be > 0 or None
    {"dt": 0.1, "mode": "cubic"},      # unknown mode
])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        SensoryFrontend(in_dim=4, d_model=D, **kwargs)


def test_forward_rejects_bad_rank():
    fe = SensoryFrontend(in_dim=4, d_model=D, dt=0.1)
    with pytest.raises(ValueError):
        fe(torch.randn(4))                                # 1-D not allowed
    with pytest.raises(ValueError):
        fe(torch.randn(2, 5, 4), torch.randn(2, 5))       # timestamps must be 1-D
