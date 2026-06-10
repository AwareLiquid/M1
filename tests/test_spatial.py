"""
tests/test_spatial.py — spatial-reasoning frontends (mt_lnn.spatial) and their
injection into the backbone via inputs_embeds.

Pins the spatial contract that mirrors the multimodal one:

  • GridCellEncoding maps (B,N,coord_dim) → (B,N,out_dim) deterministically, with
    a hexagonal basis when coord_dim==2 and Fourier features otherwise.
  • SpatialCoordEncoder / PointCloudEncoder / VoxelPatchEmbed all emit
    (B, N, d_model) tokens that fuse() and feed the model.
  • Encoders are trainable: gradient reaches their weights and type embeddings.
  • The grid-cell code is position-discriminative (distinct coords → distinct
    codes) and order-equivariant per point.

Tiny dims, fast under pytest.
"""
import warnings

import pytest
import torch

warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel
from mt_lnn.multimodal import fuse
from mt_lnn.spatial import (
    GridCellEncoding,
    SpatialCoordEncoder,
    PointCloudEncoder,
    VoxelPatchEmbed,
)

D = 104


def _model(max_seq_len=64):
    cfg = MTLNNConfig(
        vocab_size=64, d_model=D, n_layers=2, n_heads=13, n_kv_heads=1,
        d_head=8, max_seq_len=max_seq_len, gwtb_n_heads=1, dropout=0.0,
        attention_dropout=0.0,
    )
    return MTLNNModel(cfg).eval()


# --- GridCellEncoding -----------------------------------------------------

def test_gridcell_hexagonal_out_dim_and_shape():
    enc = GridCellEncoding(coord_dim=2, n_scales=6)
    assert enc.hexagonal
    assert enc.out_dim == 6 * 3 * 2                 # scales * dirs * (cos,sin)
    out = enc(torch.rand(2, 5, 2))
    assert out.shape == (2, 5, enc.out_dim)
    assert torch.isfinite(out).all()
    assert out.abs().max() <= 1.0 + 1e-5            # cos/sin bounded


def test_gridcell_fourier_for_3d():
    enc = GridCellEncoding(coord_dim=3, n_scales=4)
    assert not enc.hexagonal
    assert enc.out_dim == 3 * 4 * 2
    assert enc(torch.rand(1, 7, 3)).shape == (1, 7, enc.out_dim)


def test_gridcell_deterministic_and_no_params():
    enc = GridCellEncoding(coord_dim=2, n_scales=3)
    assert sum(p.numel() for p in enc.parameters()) == 0   # fixed feature map
    x = torch.rand(1, 4, 2)
    assert torch.allclose(enc(x), enc(x))                  # deterministic


def test_gridcell_discriminates_position():
    enc = GridCellEncoding(coord_dim=2, n_scales=6)
    a = enc(torch.tensor([[[0.1, 0.2]]]))
    b = enc(torch.tensor([[[0.7, 0.9]]]))
    assert (a - b).abs().sum() > 1e-3                       # different positions → different codes


def test_gridcell_rejects_wrong_coord_dim():
    enc = GridCellEncoding(coord_dim=2)
    with pytest.raises(ValueError):
        enc(torch.rand(1, 4, 3))


# --- SpatialCoordEncoder --------------------------------------------------

def test_spatial_coord_encoder_shape():
    enc = SpatialCoordEncoder(d_model=D, coord_dim=2)
    assert enc(torch.rand(2, 5, 2)).shape == (2, 5, D)


def test_spatial_coord_encoder_with_features():
    enc = SpatialCoordEncoder(d_model=D, coord_dim=2, feat_dim=8)
    out = enc(torch.rand(2, 5, 2), torch.rand(2, 5, 8))
    assert out.shape == (2, 5, D)
    with pytest.raises(ValueError):
        enc(torch.rand(2, 5, 2))                            # missing required feats
    enc0 = SpatialCoordEncoder(d_model=D, coord_dim=2, feat_dim=0)
    with pytest.raises(ValueError):
        enc0(torch.rand(2, 5, 2), torch.rand(2, 5, 4))     # feats passed but feat_dim=0


# --- PointCloudEncoder ----------------------------------------------------

def test_pointcloud_encoder_shape_and_equivariance():
    enc = PointCloudEncoder(d_model=D, in_dim=3, coord_dim=3).eval()
    pts = torch.rand(1, 6, 3)
    out = enc(pts)
    assert out.shape == (1, 6, D)
    # permute points → permuted tokens (permutation equivariance)
    perm = torch.tensor([5, 0, 3, 1, 4, 2])
    out_perm = enc(pts[:, perm, :])
    assert torch.allclose(out_perm, out[:, perm, :], atol=1e-5)


# --- VoxelPatchEmbed ------------------------------------------------------

def test_voxel_patch_embed_shape():
    enc = VoxelPatchEmbed(d_model=D, in_chans=1, patch_size=4)
    out = enc(torch.rand(2, 1, 8, 8, 4))               # (8/4)*(8/4)*(4/4)=2*2*1=4
    assert out.shape == (2, 4, D)


def test_voxel_patch_embed_rejects_indivisible():
    enc = VoxelPatchEmbed(d_model=D, patch_size=4)
    with pytest.raises(ValueError):
        enc(torch.rand(1, 1, 8, 8, 6))                 # 6 not divisible by 4


# --- backbone injection ---------------------------------------------------

def test_spatial_tokens_feed_backbone():
    torch.manual_seed(0)
    m = _model()
    enc = SpatialCoordEncoder(d_model=D, coord_dim=2)
    ids = torch.randint(0, 64, (2, 6))
    fused = fuse(enc(torch.rand(2, 4, 2)), m.embed_tokens(ids))   # 4 + 6 = 10
    with torch.no_grad():
        out = m(inputs_embeds=fused, use_cache=True)
    assert out["logits"].shape == (2, 10, 64)
    assert torch.isfinite(out["logits"]).all()
    assert out["cache"].token_count == 10


def test_spatial_encoders_are_trainable():
    torch.manual_seed(1)
    m = _model()
    m.train()
    coord = SpatialCoordEncoder(d_model=D, coord_dim=2)
    voxel = VoxelPatchEmbed(d_model=D, patch_size=4)
    ids = torch.randint(0, 64, (1, 5))
    fused = fuse(
        coord(torch.rand(1, 3, 2)),
        voxel(torch.rand(1, 1, 4, 4, 4)),
        m.embed_tokens(ids),
    )
    labels = torch.randint(0, 64, (1, fused.shape[1]))
    out = m(inputs_embeds=fused, labels=labels)
    out["loss"].backward()
    assert coord.mlp[0].weight.grad is not None and coord.mlp[0].weight.grad.abs().sum() > 0
    assert coord.type_embed.grad is not None and coord.type_embed.grad.abs().sum() > 0
    assert voxel.proj.weight.grad is not None and voxel.proj.weight.grad.abs().sum() > 0


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
