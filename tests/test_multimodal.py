"""
tests/test_multimodal.py — modality injection path (mt_lnn.multimodal +
MTLNNModel.forward(inputs_embeds=...)).

The backbone is text-token-native; these tests pin the multimodal injection
contract that makes it modality-agnostic:

  • VisionPatchEmbed turns (B,C,H,W) images into (B, N_patch, d_model) tokens.
  • ModalityProjector turns arbitrary (B,N,in_dim) features into d_model tokens.
  • fuse() concatenates modality blocks along the sequence axis.
  • forward(inputs_embeds=...) consumes a fused sequence and returns logits;
    text-only forward is unchanged; the input_ids/inputs_embeds guard fires.
  • Encoders are trainable: gradient reaches their weights.
  • generate() still works via embed_tokens for a multimodal prefix (manual loop).

Tiny dims, fast under pytest.
"""

import warnings

import pytest
import torch

warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel
from mt_lnn.multimodal import (
    ModalityProjector,
    VisionPatchEmbed,
    fuse,
    build_modality_pad_mask,
)

D = 104


def _model(max_seq_len=64):
    cfg = MTLNNConfig(
        vocab_size=64, d_model=D, n_layers=2, n_heads=13, n_kv_heads=1,
        d_head=8, max_seq_len=max_seq_len, gwtb_n_heads=1, dropout=0.0,
        attention_dropout=0.0,
    )
    return MTLNNModel(cfg).eval()


# --- encoders -------------------------------------------------------------

def test_vision_patch_embed_shape():
    vis = VisionPatchEmbed(d_model=D, in_chans=3, patch_size=16)
    out = vis(torch.randn(2, 3, 32, 48))      # (32/16)*(48/16) = 2*3 = 6 patches
    assert out.shape == (2, 6, D)


def test_vision_patch_embed_rejects_indivisible():
    vis = VisionPatchEmbed(d_model=D, patch_size=16)
    with pytest.raises(ValueError):
        vis(torch.randn(1, 3, 30, 32))         # 30 not divisible by 16


def test_modality_projector_shape_and_rejects_2d():
    proj = ModalityProjector(in_dim=40, d_model=D)
    assert proj(torch.randn(2, 5, 40)).shape == (2, 5, D)
    with pytest.raises(ValueError):
        proj(torch.randn(2, 40))               # missing sequence axis


def test_fuse_concatenates_along_sequence():
    a = torch.randn(2, 3, D)
    b = torch.randn(2, 4, D)
    assert fuse(a, b).shape == (2, 7, D)
    with pytest.raises(ValueError):
        fuse(a, torch.randn(2, 4, D + 1))      # mismatched d_model


def test_build_modality_pad_mask_default_all_valid():
    a = torch.randn(2, 3, D)
    b = torch.randn(2, 4, D)
    mask = build_modality_pad_mask(a, b)
    assert mask.shape == (2, 7)
    assert mask.all()


# --- forward injection ----------------------------------------------------

def test_forward_with_fused_inputs_embeds():
    torch.manual_seed(0)
    m = _model()
    vis = VisionPatchEmbed(d_model=D, patch_size=16)
    ids = torch.randint(0, 64, (2, 6))
    fused = fuse(vis(torch.randn(2, 3, 32, 32)), m.embed_tokens(ids))  # 4 + 6 = 10
    with torch.no_grad():
        out = m(inputs_embeds=fused, use_cache=True)
    assert out["logits"].shape == (2, 10, 64)
    assert torch.isfinite(out["logits"]).all()
    assert out["cache"].token_count == 10


def test_text_only_forward_unchanged():
    torch.manual_seed(1)
    m = _model()
    ids = torch.randint(0, 64, (2, 5))
    with torch.no_grad():
        a = m(ids)["logits"]
        # passing the same tokens as embeds should match (dropout off in eval).
        b = m(inputs_embeds=m.embed_tokens(ids))["logits"]
    assert torch.allclose(a, b, atol=1e-5)


def test_input_ids_and_inputs_embeds_are_mutually_exclusive():
    m = _model()
    ids = torch.randint(0, 64, (1, 3))
    with pytest.raises(ValueError):
        m(input_ids=ids, inputs_embeds=m.embed_tokens(ids))
    with pytest.raises(ValueError):
        m()  # neither provided


def test_inputs_embeds_wrong_d_model_raises():
    m = _model()
    with pytest.raises(ValueError):
        m(inputs_embeds=torch.randn(1, 4, D + 8))


def test_encoders_are_trainable():
    torch.manual_seed(2)
    m = _model()
    m.train()
    vis = VisionPatchEmbed(d_model=D, patch_size=16)
    proj = ModalityProjector(in_dim=40, d_model=D)
    ids = torch.randint(0, 64, (1, 5))
    fused = fuse(vis(torch.randn(1, 3, 16, 16)), proj(torch.randn(1, 3, 40)),
                 m.embed_tokens(ids))
    labels = torch.randint(0, 64, (1, fused.shape[1]))
    out = m(inputs_embeds=fused, labels=labels)
    out["loss"].backward()
    assert vis.proj.weight.grad is not None and vis.proj.weight.grad.abs().sum() > 0
    assert proj.proj.weight.grad is not None and proj.proj.weight.grad.abs().sum() > 0
    assert vis.type_embed.grad is not None and vis.type_embed.grad.abs().sum() > 0


def test_multimodal_prefill_then_token_decode():
    """A realistic flow: prefill a fused image+text prefix, then autoregress
    text tokens off the resulting cache."""
    torch.manual_seed(3)
    m = _model()
    vis = VisionPatchEmbed(d_model=D, patch_size=16)
    ids = torch.randint(0, 64, (1, 4))
    fused = fuse(vis(torch.randn(1, 3, 16, 16)), m.embed_tokens(ids))
    with torch.no_grad():
        out = m(inputs_embeds=fused, use_cache=True)
        cache = out["cache"]
        cur = out["logits"][:, -1, :].argmax(-1, keepdim=True)
        toks = [int(cur.item())]
        for _ in range(5):
            o = m(cur, cache=cache, use_cache=True)
            cache = o["cache"]
            cur = o["logits"][:, -1, :].argmax(-1, keepdim=True)
            toks.append(int(cur.item()))
    assert len(toks) == 6
    assert all(0 <= t < 64 for t in toks)


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
