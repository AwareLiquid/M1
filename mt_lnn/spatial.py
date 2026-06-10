"""
mt_lnn/spatial.py — spatial-reasoning frontends for the MT-LNN backbone.

Purpose
-------
Give the text-native MT-LNN backbone a *spatial computation* sense: a way to
ingest continuous coordinates, point clouds, and voxel/occupancy grids and turn
them into ``(B, N, d_model)`` tokens that fuse with text via
:func:`mt_lnn.multimodal.fuse` and enter the model through
``MTLNNModel.forward(inputs_embeds=...)``.

This module is a *sibling* of :mod:`mt_lnn.multimodal` and obeys the exact same
contract:

    every encoder returns ``(B, N, d_model)`` and adds a learnable per-modality
    type embedding so the backbone can tell spatial tokens from text/vision.

=> ZERO coupling with the core model. ``model.py`` needs no changes; these
encoders never import it. Train them by wrapping alongside the model in your
optimizer, identical to the vision frontends.

Why "grid cells"?
-----------------
The biological motivation that ties this to the broader MT-LNN research line
(see the ``grid-cell-emergence`` experiment) is the entorhinal *grid cell*: a
neuron that tiles 2-D space with a periodic hexagonal firing lattice at a
characteristic scale. A *population* of grid cells across many scales forms a
metric, near-unique code for continuous position — the brain's spatial
coordinate system. :class:`GridCellEncoding` reproduces that population code as
a fixed (non-trainable) feature map: multi-scale periodic projections of a
coordinate. It is to *space* what rotary/Fourier position codes are to *sequence
index* — but 2-D/3-D and biologically grounded.

Layering
--------
    raw spatial input
        │  (GridCellEncoding / PointNet MLP / Conv3d patchifier)
        ▼
    fixed or learned spatial features
        │  (Linear projector + LayerNorm + type embedding)
        ▼
    (B, N, d_model)  ──fuse()──►  MTLNNModel(inputs_embeds=...)

Quick start::

    from mt_lnn.spatial import SpatialCoordEncoder
    from mt_lnn.multimodal import fuse

    enc = SpatialCoordEncoder(d_model=model.config.d_model, coord_dim=2)
    coords = torch.rand(B, N, 2)                 # N points in the unit square
    spat_tok = enc(coords)                       # (B, N, d_model)
    txt_tok  = model.embed_tokens(input_ids)     # (B, T, d_model)
    out = model(inputs_embeds=fuse(spat_tok, txt_tok), use_cache=True)
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

__all__ = [
    "GridCellEncoding",
    "PlaceCellCode",
    "SpatialCoordEncoder",
    "PointCloudEncoder",
    "VoxelPatchEmbed",
]


# ---------------------------------------------------------------------------
# Grid-cell population code (fixed feature map)
# ---------------------------------------------------------------------------

class GridCellEncoding(nn.Module):
    """Multi-scale periodic encoding of a continuous coordinate.

    Maps ``(B, N, coord_dim) -> (B, N, out_dim)`` where every output channel is
    ``cos`` or ``sin`` of the coordinate projected onto a direction and scaled
    by a spatial frequency. This is a *deterministic, non-trainable* feature map
    (the directions and frequencies are registered buffers), so it is cheap,
    reproducible, and adds no parameters.

    Two regimes
    -----------
    * ``coord_dim == 2`` (default): a biologically faithful **hexagonal** grid
      basis. For each of ``n_scales`` geometric frequencies we use three
      lattice directions 60° apart (0°, 60°, 120°) — the firing pattern of a
      real grid-cell module. ``out_dim = n_scales * 3 * 2``.
    * ``coord_dim != 2``: an axis-generalised **Fourier feature** basis
      (NeRF-style): each axis gets ``n_scales`` frequencies, encoded as
      ``cos``/``sin``. ``out_dim = coord_dim * n_scales * 2``. This keeps the
      module usable for 1-D (e.g. time) and 3-D (volumetric) coordinates.

    Parameters
    ----------
    coord_dim : dimensionality of each input coordinate (1, 2, 3, …).
    n_scales : number of spatial frequency bands.
    base_wavelength : the largest wavelength (in input units). Successive scales
        shrink the wavelength by ``1/scale_ratio`` each step, so the module
        resolves both coarse and fine spatial structure.
    scale_ratio : geometric ratio between adjacent grid-cell module scales.
        ~1.4–1.7 is the ratio observed between real entorhinal modules; 1.5 is a
        good default.

    Notes
    -----
    Inputs are assumed to be in roughly ``[0, 1]`` (normalise your coordinates).
    ``base_wavelength`` is interpreted in those normalised units.
    """

    def __init__(
        self,
        coord_dim: int = 2,
        n_scales: int = 6,
        base_wavelength: float = 1.0,
        scale_ratio: float = 1.5,
    ):
        super().__init__()
        if coord_dim < 1:
            raise ValueError(f"coord_dim must be >= 1, got {coord_dim}")
        if n_scales < 1:
            raise ValueError(f"n_scales must be >= 1, got {n_scales}")
        self.coord_dim = coord_dim
        self.n_scales = n_scales
        self.hexagonal = coord_dim == 2

        # Angular frequencies per scale: ω = 2π / wavelength, wavelength shrinks
        # geometrically. Shape (n_scales,).
        wavelengths = base_wavelength / (scale_ratio ** torch.arange(n_scales).float())
        omegas = (2.0 * math.pi) / wavelengths

        if self.hexagonal:
            # Three hexagonal lattice directions, 60° apart. Shape (3, 2).
            angles = torch.tensor([0.0, math.pi / 3.0, 2.0 * math.pi / 3.0])
            dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
            # Per (scale, direction) wave vector k = ω · direction. (n_scales, 3, 2)
            k = omegas[:, None, None] * dirs[None, :, :]
            k = k.reshape(-1, 2)                       # (n_scales*3, 2)
            self.register_buffer("wave_vectors", k, persistent=True)
            self.out_dim = k.shape[0] * 2             # cos + sin
        else:
            # Axis-aligned Fourier features: each axis × each frequency.
            # Wave vectors are ω·e_axis. (coord_dim*n_scales, coord_dim)
            eye = torch.eye(coord_dim)
            k = (omegas[:, None, None] * eye[None, :, :]).reshape(-1, coord_dim)
            self.register_buffer("wave_vectors", k, persistent=True)
            self.out_dim = k.shape[0] * 2             # cos + sin

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.dim() != 3 or coords.shape[-1] != self.coord_dim:
            raise ValueError(
                f"expected (B, N, coord_dim={self.coord_dim}), "
                f"got shape {tuple(coords.shape)}"
            )
        # phase[b,n,j] = <coords[b,n,:], wave_vectors[j,:]>  → (B, N, n_waves)
        phase = torch.matmul(coords, self.wave_vectors.t())
        return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


# ---------------------------------------------------------------------------
# Place-cell population code  (supervised TARGET for path integration)
# ---------------------------------------------------------------------------

class PlaceCellCode(nn.Module):
    """Population of place cells over an arena → a soft target distribution.

    Classification note — this is the *output target* side of spatial
    computation, the dual of the input-side :class:`GridCellEncoding`:

    * :class:`GridCellEncoding` turns a coordinate INTO model tokens (input).
    * :class:`PlaceCellCode` turns a coordinate into a supervised TARGET — the
      population activity a path-integrating network is trained to predict.

    Why this lives here (and why it matters)
    ----------------------------------------
    Hexagonal **grid cells** do not appear by training a recurrent net to
    localise against *arbitrary* targets. Sorscher et al. 2019 ("A unified
    theory for the origin of grid cells") showed the *shape of the place-cell
    target* is the decisive lever: a plain single-bump **Gaussian** code rarely
    yields grids, whereas a **difference-of-Gaussians** (DoG / center–surround /
    "Mexican-hat") target reliably triggers hexagonal grid emergence in the
    recurrent layer. Until now that recipe lived only inside the throwaway
    ``grid_cell_emergence`` experiment script; pinning it here makes the
    emergence lever a first-class, tested, reusable capability of the spatial
    stack (and removes the duplicated, untested copy).

    ``forward(pos) -> softmax target`` maps ``(B, …, coord_dim)`` positions to
    ``(B, …, n_place)`` non-negative rows summing to 1, ready as the target of a
    cross-entropy / KL path-integration loss.

    Parameters
    ----------
    n_place : number of place fields (target dimensionality).
    arena : arena extent. A float (square/cube ``[0, arena]^coord_dim``) or a
        per-axis sequence of length ``coord_dim``.
    coord_dim : spatial dimensionality of a position (2 for the grid-cell case).
    sigma : place-field width (centre Gaussian std, in arena units).
    mode : ``"gaussian"`` (proven, stable) or ``"dog"`` (the grid-emergence
        trigger). DoG is the recommended setting when the *goal* is to grow
        grid cells in the recurrent layer.
    dog_surround : surround std as a multiple of ``sigma`` (DoG only). ~2 is
        the canonical center–surround ratio.
    dog_amp : surround amplitude, kept strictly in ``(0, 1)``. The peak at the
        field centre is ``1 - dog_amp``; ``dog_amp == 1`` collapses the peak to
        zero, the softmax target goes uniform and the loss floors at ``ln N``
        (the bug that invalidated the first DoG sweep). Values outside ``(0, 1)``
        raise.
    dog_temp : softmax sharpening temperature (DoG only). Raw DoG values span
        only ``~[-dog_amp, 1-dog_amp]`` — far flatter than Gaussian logits — so
        a small temperature is required to restore a peaked yet center–surround
        target the network can actually fit.
    centers : optional explicit field centres ``(n_place, coord_dim)``. When
        omitted, centres are drawn uniformly in the arena from a fixed ``seed``
        so the target code is reproducible across runs/models.
    seed : RNG seed for the random centre layout (ignored if ``centers`` given).

    Notes
    -----
    Centres are stored as a non-trainable buffer (the target geometry is fixed,
    exactly like real place fields). The module has **zero learnable
    parameters** and never imports the backbone — fully decoupled.
    """

    def __init__(
        self,
        n_place: int = 512,
        arena: float = 2.2,
        coord_dim: int = 2,
        sigma: float = 0.12,
        mode: str = "gaussian",
        dog_surround: float = 2.0,
        dog_amp: float = 0.5,
        dog_temp: float = 0.05,
        centers: Optional[torch.Tensor] = None,
        seed: int = 123,
    ):
        super().__init__()
        if n_place < 1:
            raise ValueError(f"n_place must be >= 1, got {n_place}")
        if coord_dim < 1:
            raise ValueError(f"coord_dim must be >= 1, got {coord_dim}")
        if mode not in ("gaussian", "dog"):
            raise ValueError(f"mode must be 'gaussian' or 'dog', got {mode!r}")
        if mode == "dog" and not (0.0 < dog_amp < 1.0):
            raise ValueError(
                f"dog_amp must be in the open interval (0, 1), got {dog_amp} "
                "(amp>=1 collapses the centre peak → uniform target → loss floors at ln N)"
            )
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")

        self.n_place = n_place
        self.coord_dim = coord_dim
        self.sigma = float(sigma)
        self.mode = mode
        self.dog_surround = float(dog_surround)
        self.dog_amp = float(dog_amp)
        self.dog_temp = float(dog_temp)

        # Per-axis arena extent → shape (coord_dim,).
        if isinstance(arena, (int, float)):
            extent = torch.full((coord_dim,), float(arena))
        else:
            extent = torch.as_tensor(arena, dtype=torch.float32)
            if extent.shape != (coord_dim,):
                raise ValueError(
                    f"arena sequence must have length coord_dim={coord_dim}, "
                    f"got shape {tuple(extent.shape)}"
                )

        if centers is not None:
            centers = torch.as_tensor(centers, dtype=torch.float32)
            if centers.shape != (n_place, coord_dim):
                raise ValueError(
                    f"centers must be (n_place={n_place}, coord_dim={coord_dim}), "
                    f"got {tuple(centers.shape)}"
                )
        else:
            # Reproducible uniform layout, independent of global RNG state.
            g = torch.Generator().manual_seed(int(seed))
            centers = torch.rand(n_place, coord_dim, generator=g) * extent

        self.register_buffer("place_centers", centers, persistent=True)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """``(B, …, coord_dim) -> (B, …, n_place)`` softmax place-cell target."""
        if pos.shape[-1] != self.coord_dim:
            raise ValueError(
                f"expected positions (…, coord_dim={self.coord_dim}), "
                f"got shape {tuple(pos.shape)}"
            )
        centers = self.place_centers.to(pos.dtype)
        # Squared distance from every position to every field centre.
        # pos: (…, 1, D)  centres: (n_place, D)  → d2: (…, n_place)
        d2 = ((pos.unsqueeze(-2) - centers) ** 2).sum(dim=-1)
        s1 = self.sigma
        if self.mode == "gaussian":
            # Single-bump Gaussian logits → softmax over fields.
            return torch.softmax(-d2 / (2.0 * s1 * s1), dim=-1)
        # Difference-of-Gaussians with a sub-unity surround (positive centre
        # peak) and a sharpening temperature — the grid-emergence trigger.
        s2 = s1 * self.dog_surround
        center = torch.exp(-d2 / (2.0 * s1 * s1))
        surround = torch.exp(-d2 / (2.0 * s2 * s2))
        dog = (center - self.dog_amp * surround) / self.dog_temp
        return torch.softmax(dog, dim=-1)


# ---------------------------------------------------------------------------
# Coordinate encoder → d_model tokens
# ---------------------------------------------------------------------------

class SpatialCoordEncoder(nn.Module):
    """Continuous coordinates (+ optional per-point features) → d_model tokens.

    ``(B, N, coord_dim) [+ (B, N, feat_dim)] -> (B, N, d_model)``.

    Pipeline: a fixed :class:`GridCellEncoding` of the coordinates is
    concatenated with any raw per-point features, then a small trainable MLP
    projects the result to ``d_model``. A learnable type embedding marks these
    as spatial tokens. This is the recommended entry point for "where is this
    thing" style inputs (object positions, waypoints, sampled query points).

    Parameters
    ----------
    d_model : backbone embedding width.
    coord_dim : coordinate dimensionality (2 → hexagonal grid-cell code).
    feat_dim : width of optional extra per-point features (0 = none).
    n_scales, base_wavelength, scale_ratio : forwarded to GridCellEncoding.
    hidden : MLP hidden width (defaults to d_model).
    dropout : dropout on the output tokens.
    """

    def __init__(
        self,
        d_model: int,
        coord_dim: int = 2,
        feat_dim: int = 0,
        n_scales: int = 6,
        base_wavelength: float = 1.0,
        scale_ratio: float = 1.5,
        hidden: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.coord_dim = coord_dim
        self.feat_dim = feat_dim
        self.grid = GridCellEncoding(
            coord_dim=coord_dim,
            n_scales=n_scales,
            base_wavelength=base_wavelength,
            scale_ratio=scale_ratio,
        )
        in_dim = self.grid.out_dim + feat_dim
        hidden = hidden or d_model
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.type_embed = nn.Parameter(torch.zeros(d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        coords: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        code = self.grid(coords)                       # (B, N, grid.out_dim)
        if self.feat_dim:
            if feats is None:
                raise ValueError(
                    f"feat_dim={self.feat_dim} but no feats tensor was passed"
                )
            if feats.dim() != 3 or feats.shape[-1] != self.feat_dim:
                raise ValueError(
                    f"expected feats (B, N, {self.feat_dim}), "
                    f"got {tuple(feats.shape)}"
                )
            if feats.shape[:2] != code.shape[:2]:
                raise ValueError(
                    "coords and feats must share (B, N); got "
                    f"{tuple(coords.shape[:2])} vs {tuple(feats.shape[:2])}"
                )
            code = torch.cat([code, feats], dim=-1)
        elif feats is not None:
            raise ValueError("feat_dim=0 but a feats tensor was passed")
        x = self.norm(self.mlp(code)) + self.type_embed
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Point-cloud encoder (PointNet-style, permutation-equivariant per point)
# ---------------------------------------------------------------------------

class PointCloudEncoder(nn.Module):
    """Point cloud → d_model tokens via a shared per-point MLP (PointNet-style).

    ``(B, N, in_dim) -> (B, N, d_model)`` where ``in_dim`` is typically 3 (xyz)
    or 6 (xyz + rgb / normal). Each point is embedded independently by the same
    MLP (permutation-equivariant: reorder points → reorder tokens), then a grid-
    cell code of the *xyz* sub-vector is added so absolute spatial position is
    preserved through the otherwise position-free MLP.

    Parameters
    ----------
    d_model : backbone width.
    in_dim : per-point input width (>= coord_dim; first ``coord_dim`` are xyz).
    coord_dim : how many leading channels are spatial coordinates (default 3).
    n_scales, base_wavelength, scale_ratio : grid-cell positional code params.
    dropout : output dropout.
    """

    def __init__(
        self,
        d_model: int,
        in_dim: int = 3,
        coord_dim: int = 3,
        n_scales: int = 6,
        base_wavelength: float = 1.0,
        scale_ratio: float = 1.5,
        dropout: float = 0.0,
    ):
        super().__init__()
        if coord_dim > in_dim:
            raise ValueError(
                f"coord_dim={coord_dim} cannot exceed in_dim={in_dim}"
            )
        self.in_dim = in_dim
        self.coord_dim = coord_dim
        self.grid = GridCellEncoding(
            coord_dim=coord_dim,
            n_scales=n_scales,
            base_wavelength=base_wavelength,
            scale_ratio=scale_ratio,
        )
        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_proj = nn.Linear(self.grid.out_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.type_embed = nn.Parameter(torch.zeros(d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.dim() != 3 or points.shape[-1] != self.in_dim:
            raise ValueError(
                f"expected (B, N, in_dim={self.in_dim}), got {tuple(points.shape)}"
            )
        feat = self.point_mlp(points)                       # (B, N, d_model)
        pos = self.pos_proj(self.grid(points[..., : self.coord_dim]))
        x = self.norm(feat + pos) + self.type_embed
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Voxel / occupancy-grid encoder (3-D analogue of VisionPatchEmbed)
# ---------------------------------------------------------------------------

class VoxelPatchEmbed(nn.Module):
    """Conv3d patchifier for voxel / occupancy grids.

    ``(B, C, X, Y, Z) -> (B, N_vox, d_model)`` where
    ``N_vox = (X/p)·(Y/p)·(Z/p)``. This is the volumetric analogue of
    :class:`mt_lnn.multimodal.VisionPatchEmbed`: a single strided Conv3d tiles
    the volume into non-overlapping patches and projects each to ``d_model``.
    Useful for occupancy maps, signed-distance fields, or any dense 3-D scalar/
    vector field. ``C`` is the per-voxel channel count (1 for occupancy).

    X, Y, Z must each be divisible by ``patch_size``. A learnable type embedding
    is added to every voxel-patch token.
    """

    def __init__(
        self,
        d_model: int,
        in_chans: int = 1,
        patch_size: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_chans, d_model, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(d_model)
        self.type_embed = nn.Parameter(torch.zeros(d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, voxels: torch.Tensor) -> torch.Tensor:
        if voxels.dim() != 5:
            raise ValueError(
                f"expected (B, C, X, Y, Z), got shape {tuple(voxels.shape)}"
            )
        _, _, X, Y, Z = voxels.shape
        p = self.patch_size
        if X % p or Y % p or Z % p:
            raise ValueError(
                f"X={X}, Y={Y}, Z={Z} must each be divisible by patch_size={p}"
            )
        x = self.proj(voxels)                  # (B, d_model, X', Y', Z')
        x = x.flatten(2).transpose(1, 2)       # (B, N_vox, d_model)
        x = self.norm(x) + self.type_embed
        return self.dropout(x)
