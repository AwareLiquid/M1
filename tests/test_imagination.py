"""
tests/test_imagination.py — L4 latent multi-step imagination rollout.

Pins the contract of :class:`mt_lnn.imagination.LatentImagination` — the
"mental simulation" actuator that rolls the world model's *one-step* predictor
forward in latent space to imagine how a trajectory unfolds:

  * THE key property — composition: on a head trained on a real (rotating)
    latent dynamic, the multi-step imagined trajectory tracks the TRUE k-step
    future markedly better than a static "nothing changes" baseline. This is
    what separates a genuine rollout from a wrapper: the learned one-step map
    is composed into a multi-step prediction that actually moves with the world;
  * it adds ZERO trainable parameters (pure inference-time actuator);
  * confidence decays with the horizon; novelty/confidence stay in [0, 1];
  * it is duck-typed on the head (no coupling to model.py / the backbone);
  * deterministic for a fixed input; argument validation behaves.
"""
import math
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mt_lnn.world_model import PredictiveStateHead  # noqa: E402
from mt_lnn.imagination import ImaginedTrajectory, LatentImagination  # noqa: E402


# --- helpers: a learnable, rotating latent dynamic -----------------------

def _rotation(d_model: int, theta: float) -> torch.Tensor:
    """Block-diagonal 2D rotation (orthogonal → norm-preserving, stable)."""
    A = torch.zeros(d_model, d_model)
    c, s = math.cos(theta), math.sin(theta)
    for i in range(0, d_model - 1, 2):
        A[i, i] = c; A[i, i + 1] = -s
        A[i + 1, i] = s; A[i + 1, i + 1] = c
    if d_model % 2:
        A[-1, -1] = 1.0
    return A


def _true_future(x0: torch.Tensor, A: torch.Tensor, H: int) -> torch.Tensor:
    """Ground-truth states x_1 … x_H under x_{t+1} = A x_t. Returns (B, H, d)."""
    xs, x = [], x0
    for _ in range(H):
        x = x @ A.T
        xs.append(x)
    return torch.stack(xs, dim=1)


def _train_head(head, A, d_model, *, steps=800, seq=8, batch=64, seed=0):
    opt = torch.optim.Adam(
        [p for p in head.parameters() if p.requires_grad], lr=1e-2
    )
    g = torch.Generator().manual_seed(seed)
    head.train()
    for _ in range(steps):
        x0 = torch.randn(batch, d_model, generator=g)
        seqx, x = [x0], x0
        for _ in range(seq - 1):
            x = x @ A.T
            seqx.append(x)
        X = torch.stack(seqx, dim=1)                  # (B, seq, d_model)
        _, loss = head(X, compute_loss=True)
        opt.zero_grad()
        loss.backward()
        opt.step()
    head.eval()


def _head(d_model=16, proj_dim=8, **kw):
    return PredictiveStateHead(
        d_model, proj_ratio=proj_dim / d_model, hidden_ratio=1.0,
        warmup_steps=50, **kw,
    )


# --- the headline property: composition beats the static baseline ---------

def test_imagination_composes_and_beats_static_baseline():
    torch.manual_seed(0)
    d_model, proj_dim, H = 16, 8, 5
    head = _head(d_model, proj_dim)
    A = _rotation(d_model, theta=0.5)
    _train_head(head, A, d_model)

    imag = LatentImagination(head, horizon=H, trust_decay=1.0,
                             novelty_penalty=0.0, renorm=True)
    g = torch.Generator().manual_seed(99)
    x0 = torch.randn(8, d_model, generator=g)
    traj = imag.imagine(x0)

    with torch.no_grad():
        true_x = _true_future(x0, A, H)                       # (B, H, d)
        z_true = F.normalize(head.target_proj(true_x), dim=-1)  # (B, H, P)
        z0 = F.normalize(head.online_proj(x0), dim=-1)          # (B, P)

    cos_imag = (traj.latents * z_true).sum(dim=-1)            # (B, H)
    cos_static = (z0.unsqueeze(1) * z_true).sum(dim=-1)       # (B, H)

    # The world genuinely moves, so "assume nothing changes" degrades with the
    # horizon; the composed rollout keeps tracking it.
    assert cos_imag[:, -1].mean() > cos_static[:, -1].mean() + 0.05, (
        f"far-horizon: imag={cos_imag[:, -1].mean():.3f} "
        f"static={cos_static[:, -1].mean():.3f}"
    )
    assert cos_imag.mean() > cos_static.mean()


def test_one_step_imagination_matches_head_prediction():
    # At horizon 1 the imagined latent is exactly the head's own normalised
    # one-step prediction — the rollout is grounded in the real head.
    torch.manual_seed(0)
    head = _head()
    imag = LatentImagination(head, horizon=1)
    g = torch.Generator().manual_seed(3)
    x0 = torch.randn(4, 16, generator=g)
    traj = imag.imagine(x0)
    with torch.no_grad():
        z1 = F.normalize(head.predictor(head.online_proj(x0)), dim=-1)
    assert torch.allclose(traj.latents[:, 0, :], z1, atol=1e-6)


# --- zero parameters / not a Module --------------------------------------

def test_zero_parameters():
    head = _head()
    n_before = sum(p.numel() for p in head.parameters())
    imag = LatentImagination(head, horizon=4)
    imag.imagine(torch.randn(2, 16))
    n_after = sum(p.numel() for p in head.parameters())
    assert imag.n_parameters == 0
    assert n_after == n_before
    assert not isinstance(imag, nn.Module)
    assert not hasattr(imag, "parameters")


# --- confidence / novelty bounds and horizon decay ------------------------

def test_confidence_decays_with_horizon():
    torch.manual_seed(0)
    head = _head()
    imag = LatentImagination(head, horizon=6, trust_decay=0.8, novelty_penalty=0.0)
    traj = imag.imagine(torch.randn(3, 16))
    c = traj.confidence
    # strictly decreasing per step (trust_decay < 1, novelty penalty off)
    assert torch.all(c[:, 1:] < c[:, :-1] + 1e-6)
    assert c[:, -1].mean() < c[:, 0].mean()


def test_confidence_and_novelty_in_unit_interval():
    torch.manual_seed(0)
    head = _head()
    traj = LatentImagination(head, horizon=5).imagine(torch.randn(4, 16))
    for t in (traj.confidence, traj.novelty):
        assert torch.all(t >= 0.0) and torch.all(t <= 1.0)


def test_latents_are_unit_norm_and_finite():
    torch.manual_seed(0)
    head = _head()
    traj = LatentImagination(head, horizon=7).imagine(torch.randn(4, 16))
    norms = traj.latents.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
    assert torch.isfinite(traj.latents).all()


def test_confidence_weighted_drift_is_scalar_per_batch_in_unit_interval():
    torch.manual_seed(0)
    head = _head()
    traj = LatentImagination(head, horizon=5).imagine(torch.randn(6, 16))
    d = traj.confidence_weighted_drift()
    assert d.shape == (6,)
    assert torch.all(d >= 0.0) and torch.all(d <= 1.0)


# --- shapes / API --------------------------------------------------------

def test_accepts_3d_hidden_uses_last_position():
    torch.manual_seed(0)
    head = _head()
    imag = LatentImagination(head, horizon=4)
    g = torch.Generator().manual_seed(1)
    seq = torch.randn(2, 9, 16, generator=g)
    t_seq = imag.imagine(seq)
    t_last = imag.imagine(seq[:, -1, :])
    assert torch.allclose(t_seq.latents, t_last.latents, atol=1e-6)


def test_shapes_and_endpoint():
    torch.manual_seed(0)
    head = _head(d_model=16, proj_dim=8)
    traj = LatentImagination(head, horizon=4).imagine(torch.randn(5, 16))
    assert traj.latents.shape == (5, 4, 8)
    assert traj.confidence.shape == (5, 4)
    assert traj.novelty.shape == (5, 4)
    assert traj.horizon == 4
    assert torch.equal(traj.endpoint, traj.latents[:, -1, :])


def test_horizon_override():
    torch.manual_seed(0)
    head = _head()
    imag = LatentImagination(head, horizon=3)
    traj = imag.imagine(torch.randn(2, 16), horizon=6)
    assert traj.horizon == 6 and traj.latents.shape[1] == 6


# --- determinism ----------------------------------------------------------

def test_deterministic_for_fixed_input():
    torch.manual_seed(0)
    head = _head()
    imag = LatentImagination(head, horizon=5)
    x = torch.randn(3, 16)
    a = imag.imagine(x)
    b = imag.imagine(x)
    assert torch.equal(a.latents, b.latents)
    assert torch.equal(a.confidence, b.confidence)


# --- decoupling: duck-typed on the head, no backbone dependency -----------

class _FakeHead(nn.Module):
    """A minimal world-model-head-shaped object (NOT PredictiveStateHead)."""
    def __init__(self, d_model=12, proj_dim=6):
        super().__init__()
        self.proj_dim = proj_dim
        self.online_proj = nn.Linear(d_model, proj_dim, bias=False)
        self.predictor = nn.Linear(proj_dim, proj_dim, bias=False)
        self.last_pred_error = 0.2


def test_works_on_duck_typed_head():
    torch.manual_seed(0)
    fake = _FakeHead()
    imag = LatentImagination(fake, horizon=4)
    traj = imag.imagine(torch.randn(3, 12))
    assert traj.latents.shape == (3, 4, 6)
    # base trust = 1 - last_pred_error = 0.8 flows into the first-step confidence
    assert traj.confidence[:, 0].max() <= 0.8 + 1e-6


def test_imagination_module_does_not_import_model():
    import mt_lnn.imagination as im
    src = open(im.__file__, encoding="utf-8").read()
    assert "import model" not in src and "from .model" not in src
    assert "from mt_lnn.model" not in src


# --- argument validation --------------------------------------------------

def test_invalid_arguments_raise():
    head = _head()
    with pytest.raises(ValueError):
        LatentImagination(head, horizon=0)
    with pytest.raises(ValueError):
        LatentImagination(head, trust_decay=0.0)
    with pytest.raises(ValueError):
        LatentImagination(head, trust_decay=1.5)
    with pytest.raises(ValueError):
        LatentImagination(head, novelty_penalty=-0.1)
    with pytest.raises(ValueError):
        LatentImagination(head, novelty_penalty=1.5)


def test_missing_head_attributes_raise():
    class Bad:
        pass
    with pytest.raises(TypeError):
        LatentImagination(Bad())


def test_bad_hidden_shape_raises():
    head = _head()
    imag = LatentImagination(head, horizon=3)
    with pytest.raises(ValueError):
        imag.imagine(torch.randn(2, 3, 4, 16))   # 4-D not allowed


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
