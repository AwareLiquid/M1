"""
tests/test_spatial_memory.py — L2 spatial sequence memory (SpatialMemory).

Pins the Bicanski & Burgess (2021) place-indexed associative-memory contract of
``mt_lnn.spatial_memory.SpatialMemory``:

  * zero trainable parameters (fixed buffers + tensor algebra only);
  * write-then-read recovers the stored content at the written location;
  * recall is spatially local (nearby query still recalls; far query does not);
  * distinct, well-separated locations do not cross-talk;
  * pattern completion: a noisy / approximate query still recalls correctly;
  * ``decay < 1`` forgets older writes (recency bias);
  * read-out shape ``(B, M, d_model)`` is fuse()-ready;
  * SessionMemory round-trip (state_blob -> save -> load -> load_blob);
  * a shared PlaceCellCode is genuinely reused (same centre buffer);
  * deterministic for fixed inputs.

Tiny dims, fast under pytest.
"""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mt_lnn.spatial import PlaceCellCode  # noqa: E402
from mt_lnn.spatial_memory import SpatialMemory  # noqa: E402


def _mem(d_model=16, n_place=256, **kw):
    return SpatialMemory(d_model=d_model, n_place=n_place, arena=2.2, **kw)


def _cos(a, b):
    return F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()


def test_zero_trainable_parameters():
    mem = _mem()
    assert sum(p.numel() for p in mem.parameters()) == 0


def test_write_then_read_recovers_content():
    torch.manual_seed(0)
    mem = _mem()
    pos = torch.tensor([[[0.5, 0.5]]])            # (1,1,2)
    content = torch.randn(1, 1, mem.d_model)
    mem.write(pos, content)
    recalled = mem.read(pos)                       # (1,1,d)
    assert recalled.shape == content.shape
    assert _cos(recalled, content) > 0.99


def test_recall_is_spatially_local():
    torch.manual_seed(1)
    # Use the un-normalised read-out so recall *strength* (magnitude) reflects
    # how well the query matches the written location. (With normalisation a
    # single write recalls the exact content for any query — the cancellation
    # key·M / key·k_mass — so locality lives in the magnitude, not direction.)
    mem = _mem(normalize=False)
    pos = torch.tensor([[[1.0, 1.0]]])
    content = torch.randn(1, 1, mem.d_model)
    mem.write(pos, content)

    near = mem.read(torch.tensor([[[1.05, 1.0]]]))     # within a field width
    far = mem.read(torch.tensor([[[1.9, 0.2]]]))       # far away
    # Direction is preserved near the write; strength falls off with distance.
    assert _cos(near, content) > 0.9
    assert near.norm() > far.norm()


def test_distinct_locations_do_not_crosstalk():
    torch.manual_seed(2)
    mem = _mem()
    pa = torch.tensor([[[0.3, 0.3]]])
    pb = torch.tensor([[[1.9, 1.9]]])
    ca = torch.randn(1, 1, mem.d_model)
    cb = torch.randn(1, 1, mem.d_model)
    mem.write(pa, ca).write(pb, cb)
    assert _cos(mem.read(pa), ca) > 0.9
    assert _cos(mem.read(pb), cb) > 0.9
    # Each location recalls its own content much better than the other's.
    assert _cos(mem.read(pa), ca) > _cos(mem.read(pa), cb)


def test_pattern_completion_from_noisy_query():
    torch.manual_seed(3)
    mem = _mem()
    pos = torch.tensor([[[0.8, 1.2]]])
    content = torch.randn(1, 1, mem.d_model)
    mem.write(pos, content)
    noisy = pos + 0.03 * torch.randn_like(pos)     # approximate cue
    assert _cos(mem.read(noisy), content) > 0.9


def test_decay_forgets_old_writes():
    torch.manual_seed(4)
    # decay<1 multiplies the whole map before each write, so an old write at A
    # is attenuated by every later write elsewhere (recency bias).
    mem = _mem(decay=0.5)
    pa = torch.tensor([[[0.4, 0.4]]])
    pb = torch.tensor([[[1.8, 0.4]]])     # far from A => negligible mass at A's peak
    ca = torch.randn(1, 1, mem.d_model)
    cb = torch.randn(1, 1, mem.d_model)

    # Track A's dominant place field specifically.
    idx = mem._key(pa).argmax().item()
    mem.write(pa, ca)
    before = mem.k_mass[idx].item()
    assert before > 0

    # A write at the far location B decays A's field by ~decay (B adds ~0 there).
    mem.write(pb, cb)
    after = mem.k_mass[idx].item()
    assert after < before
    assert after == pytest.approx(0.5 * before, rel=1e-3)


def test_readout_shape_is_fuse_ready():
    mem = _mem(d_model=24)
    pos = torch.rand(2, 5, 2) * 2.2                # (B=2, M=5, 2)
    out = mem.recall_tokens(pos)
    assert out.shape == (2, 5, 24)


def test_session_memory_roundtrip():
    torch.manual_seed(5)
    mem = _mem()
    pos = torch.rand(1, 8, 2) * 2.2
    content = torch.randn(1, 8, mem.d_model)
    mem.write(pos, content)
    before = mem.read(pos)

    blob = mem.state_blob()
    assert len(blob) == 2 and all(t is not None for t in blob)

    # Fresh instance with identical config restores exactly.
    mem2 = _mem()
    mem2.load_blob(blob)
    after = mem2.read(pos)
    assert torch.allclose(before, after, atol=1e-5)


def test_shared_place_code_is_reused():
    pc = PlaceCellCode(n_place=128, arena=2.2, coord_dim=2)
    mem = SpatialMemory(d_model=8, place_code=pc)
    # Same object and same underlying centre buffer — no duplicated fields.
    assert mem.place_code is pc
    assert mem.n_place == 128


def test_deterministic():
    torch.manual_seed(7)
    pos = torch.rand(1, 4, 2) * 2.2
    content = torch.randn(1, 4, 16)
    a = _mem().write(pos, content).read(pos)
    b = _mem().write(pos, content).read(pos)
    assert torch.allclose(a, b)


def test_reset_clears_map():
    mem = _mem()
    pos = torch.tensor([[[0.5, 0.5]]])
    mem.write(pos, torch.randn(1, 1, mem.d_model))
    assert mem.occupancy.sum() > 0
    mem.reset()
    assert mem.occupancy.sum() == 0
    assert torch.count_nonzero(mem.read(pos)) == 0
