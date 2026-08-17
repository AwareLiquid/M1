"""
mt_lnn/multimodal.py — modality frontends for the MT-LNN backbone.

The core MTLNNModel is a text token model, but its sequence backbone is
modality-agnostic: any (B, T, d_model) sequence can be fed through
``MTLNNModel.forward(inputs_embeds=...)``. This module provides the small,
*real* (not stub) encoders that turn raw non-text inputs into d_model tokens
and fuse them with text token embeddings.

Design
------
Every modality is reduced to the same contract: ``(B, N, d_model)`` — N "tokens"
in the model's embedding space. You then concatenate modality token blocks with
text token embeddings (``model.embed_tokens(ids)``) along the sequence axis and
pass the fused tensor to the backbone.

Example (image + text → fused prefix)::

    from mt_lnn.multimodal import VisionPatchEmbed, fuse
    vis = VisionPatchEmbed(d_model=model.config.d_model, patch_size=16)
    img_tok = vis(pixel_values)              # (B, N_patch, d_model)
    txt_tok = model.embed_tokens(input_ids)  # (B, T, d_model)
    fused   = fuse(img_tok, txt_tok)         # (B, N_patch + T, d_model)
    out     = model(inputs_embeds=fused, use_cache=True)

Example (any pre-extracted feature, e.g. audio frames / video embeddings)::

    proj = ModalityProjector(in_dim=feat.shape[-1], d_model=model.config.d_model)
    tok  = proj(feat)                        # (B, N, d_model)

These encoders are deliberately lightweight (a Conv2d patchifier, a linear
projector). They are trainable: wrap them alongside the model in your optimizer.
The point is to provide a working, tested injection path — not a pretrained
vision/audio tower.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ModalityProjector(nn.Module):
    """Project pre-extracted features of ANY modality into the d_model token
    space (``(B, N, in_dim) → (B, N, d_model)``).

    Use for already-featurised inputs: audio frame features, video frame
    embeddings, sensor streams, or the pooled output of an external encoder.
    A learnable per-modality type embedding is added so the backbone can tell
    modalities apart in a fused sequence.
    """

    def __init__(self, in_dim: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.type_embed = nn.Parameter(torch.zeros(d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.dim() != 3:
            raise ValueError(
                f"expected (B, N, in_dim), got shape {tuple(feats.shape)}"
            )
        x = self.norm(self.proj(feats)) + self.type_embed
        return self.dropout(x)


class VisionPatchEmbed(nn.Module):
    """ViT-style conv patchifier: ``(B, C, H, W) → (B, N_patches, d_model)``.

    H and W must be divisible by ``patch_size``. N_patches = (H/p)·(W/p).
    A learnable modality type embedding is added to every patch token.
    """

    def __init__(
        self,
        d_model: int,
        in_chans: int = 3,
        patch_size: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans, d_model, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(d_model)
        self.type_embed = nn.Parameter(torch.zeros(d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.dim() != 4:
            raise ValueError(
                f"expected (B, C, H, W), got shape {tuple(pixel_values.shape)}"
            )
        _, _, H, W = pixel_values.shape
        if H % self.patch_size or W % self.patch_size:
            raise ValueError(
                f"H={H}, W={W} must be divisible by patch_size={self.patch_size}"
            )
        x = self.proj(pixel_values)              # (B, d_model, H', W')
        x = x.flatten(2).transpose(1, 2)         # (B, N_patches, d_model)
        x = self.norm(x) + self.type_embed
        return self.dropout(x)


class CLIPVisionTower(nn.Module):
    """Pretrained CLIP vision encoder → patch token features.

    Wraps a HuggingFace ``CLIPVisionModel`` and (optionally) its image
    processor. ``transformers`` is imported lazily inside ``__init__`` so the
    rest of ``mt_lnn`` never requires it. The tower is **frozen** by default
    (it is a pretrained perceptual front-end); train the downstream projector
    instead.

    ``encode`` accepts either:
      * a pre-processed pixel tensor ``(B, 3, H, W)`` already normalised for
        CLIP, or
      * a list of PIL images (then the bundled image processor is used).

    Returns patch token features ``(B, N, hidden_size)`` (CLS token dropped by
    default so it lines up with the patch-token contract of this module).
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        freeze: bool = True,
        keep_cls: bool = False,
    ):
        super().__init__()
        try:
            from transformers import CLIPVisionModel, CLIPImageProcessor
        except ImportError as e:  # pragma: no cover - exercised only w/o transformers
            raise ImportError(
                "CLIPVisionTower requires `transformers`; install it or use "
                "ModalityProjector with externally-extracted features."
            ) from e
        self.model_name = model_name
        self.keep_cls = keep_cls
        # Prefer safetensors: newer `transformers` refuses `torch.load` of .bin
        # checkpoints on torch < 2.6 (CVE-2025-32434); safetensors sidesteps it.
        try:
            self.vision = CLIPVisionModel.from_pretrained(
                model_name, use_safetensors=True
            )
        except Exception:
            self.vision = CLIPVisionModel.from_pretrained(model_name)
        try:
            self.processor = CLIPImageProcessor.from_pretrained(model_name)
        except Exception:  # pragma: no cover - processor optional
            self.processor = None
        self.hidden_size = self.vision.config.hidden_size
        self.frozen = freeze
        if freeze:
            self.vision.eval()
            for p in self.vision.parameters():
                p.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.vision.parameters()).device

    def _to_pixels(self, images) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            if images.dim() != 4:
                raise ValueError(
                    f"pixel tensor must be (B, 3, H, W), got {tuple(images.shape)}"
                )
            return images.to(self.device)
        if self.processor is None:
            raise RuntimeError(
                "no image processor available; pass a pre-processed pixel tensor"
            )
        out = self.processor(images=images, return_tensors="pt")
        return out["pixel_values"].to(self.device)

    def encode(self, images) -> torch.Tensor:
        pixel_values = self._to_pixels(images)
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            out = self.vision(pixel_values=pixel_values)
        feats = out.last_hidden_state              # (B, 1 + N_patch, hidden)
        if not self.keep_cls:
            feats = feats[:, 1:, :]                 # drop CLS → patch tokens
        return feats

    def forward(self, images) -> torch.Tensor:
        return self.encode(images)


class CLIPModalityEncoder(nn.Module):
    """End-to-end image → d_model tokens bridge.

    Composes a (frozen) :class:`CLIPVisionTower` with a trainable
    :class:`ModalityProjector`, producing ``(B, N, d_model)`` tokens ready for
    :func:`fuse` and ``MTLNNModel.forward(inputs_embeds=...)``.

    The vision tower can be injected (``tower=...``) — useful for tests and for
    sharing one tower across encoders, or to avoid a network download by
    supplying a stub that exposes ``.hidden_size`` and ``.encode(images)``.
    """

    def __init__(
        self,
        d_model: int,
        model_name: str = "openai/clip-vit-base-patch32",
        tower: Optional[nn.Module] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.tower = tower if tower is not None else CLIPVisionTower(model_name)
        if not hasattr(self.tower, "hidden_size"):
            raise ValueError("tower must expose `.hidden_size`")
        self.projector = ModalityProjector(
            in_dim=self.tower.hidden_size, d_model=d_model, dropout=dropout
        )

    def forward(self, images) -> torch.Tensor:
        feats = self.tower.encode(images)          # (B, N, hidden)
        return self.projector(feats)               # (B, N, d_model)


def fuse(*blocks: torch.Tensor) -> torch.Tensor:
    """Concatenate modality token blocks along the sequence axis (dim=1).

    All blocks must share (B, *, d_model). Order defines the position order seen
    by the causal backbone — typically modality prefix(es) first, text last so
    text attends over the perceptual context.
    """
    if not blocks:
        raise ValueError("fuse() needs at least one token block")
    d = blocks[0].shape[-1]
    for b in blocks:
        if b.dim() != 3 or b.shape[-1] != d:
            raise ValueError(
                "all blocks must be (B, N, d_model) with matching d_model; "
                f"got {[tuple(x.shape) for x in blocks]}"
            )
    return torch.cat(blocks, dim=1)


def build_modality_pad_mask(
    *blocks: torch.Tensor,
    valid: Optional[list] = None,
) -> torch.Tensor:
    """Build a (B, N_total) bool pad mask for a fused sequence.

    By default every fused token is valid (True). Pass ``valid`` as a list of
    per-block (B, N_i) bool masks to mark padding within any block (e.g. a
    variable-length audio block).
    """
    masks = []
    for i, b in enumerate(blocks):
        B, N = b.shape[0], b.shape[1]
        if valid is not None and valid[i] is not None:
            masks.append(valid[i].to(torch.bool))
        else:
            masks.append(torch.ones(B, N, dtype=torch.bool, device=b.device))
    return torch.cat(masks, dim=1)
