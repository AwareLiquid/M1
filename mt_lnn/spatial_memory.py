"""
mt_lnn/spatial_memory.py — L2 spatial sequence memory (Bicanski & Burgess 2021).

Purpose
-------
Give the MT-LNN backbone a *place-indexed associative memory*: as an agent (or a
reasoning trace) moves along a trajectory it can **write** what it perceives at
each location, and later **recall** that content by querying a position — even an
approximate or noisy one. This is the computational core of Bicanski & Burgess
(2021), "A neural-level model of spatial memory and imagery": place-cell codes
act as the addressing key into an attractor-style associative store, and recall
is pattern completion — a partial / nearby spatial cue retrieves the full memory.

Where it sits in the four-layer spatial-cognition roadmap
---------------------------------------------------------
    L1  GridCellEncoding / PlaceCellCode   — "where am I" -> spatial population code
    L2  SpatialMemory  (THIS FILE)         — "what is here" written/recalled by place
    L3  CausalActivationSteerer + demo     — keep the spatial belief on its manifold
    L4  world-model simulation (future)    — roll the dynamics forward

L2 is a *downstream consumer of L1*: it reuses :class:`mt_lnn.spatial.PlaceCellCode`
to turn a continuous position into a soft place-cell key. It writes content
embeddings keyed by that code via a Hebbian outer-product update, and reads them
back with a single key-weighted linear read-out (the linear-associator /
attractor read).

Contract & coupling
-------------------
=> ZERO coupling with the core model. ``model.py`` is never imported here.
=> ZERO trainable parameters: the whole module is fixed buffers + tensor algebra,
   so it runs on CPU and is safe to drop into any optimizer without touching
   gradients. (The *content* you write is of course whatever embeddings you
   produced upstream — e.g. ``model.embed_tokens`` or a frontend.)
=> Read-out shape is ``(B, N, d_model)``: a recalled-memory token stream that
   plugs straight into :func:`mt_lnn.multimodal.fuse` like any other modality.

Persistence is delegated to :class:`mt_lnn.memory.SessionMemory` (SQLite). This
module only exposes its serialisable state via :meth:`state_blob` /
:meth:`load_blob`; it never touches the database itself (classification stays
clean: spatial logic here, durable storage in ``memory.py``).

Quick start::

    from mt_lnn.spatial_memory import SpatialMemory
    from mt_lnn.multimodal import fuse

    mem = SpatialMemory(d_model=model.config.d_model, n_place=512)

    # Walk a trajectory, writing what we see at each step.
    mem.write(pos_seq, content_seq)              # (B,N,2), (B,N,d_model)

    # Later: recall by (approximate) position — pattern completion.
    recalled = mem.read(query_pos)               # (B,M,d_model)
    out = model(inputs_embeds=fuse(recalled, txt_tok), use_cache=True)
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .spatial import PlaceCellCode

__all__ = ["SpatialMemory"]


class SpatialMemory(nn.Module):
    """Place-indexed associative memory (Bicanski & Burgess 2021).

    A ``(n_place, d_model)`` association matrix ``M`` stores content keyed by a
    soft place-cell code. Writing is Hebbian (outer product ``key ⊗ content``
    accumulated into ``M``); reading is the linear-associator read-out
    ``key @ M`` normalised by the key mass, which performs pattern completion:
    a query at — or near — a written location retrieves the stored content.

    The module holds **zero trainable parameters**. ``M`` and the place-cell
    centres are buffers; the maths is plain tensor algebra. Batches share a
    single ``M`` (a *map* of the environment), matching the biological framing
    of one allocentric cognitive map.

    Parameters
    ----------
    d_model:
        Width of the content embeddings written/recalled. Matches the backbone.
    n_place:
        Number of place fields (associative-memory rows). More fields => finer
        spatial resolution and less cross-talk, at ``O(n_place * d_model)`` mem.
    arena, coord_dim, sigma, mode:
        Forwarded to :class:`PlaceCellCode` when this module builds its own
        place code. Ignored if ``place_code`` is supplied.
    decay:
        Per-write multiplicative forgetting applied to ``M`` *before* each write
        (``1.0`` = perfect accumulation; ``<1.0`` = leaky/recency-biased map).
    normalize:
        If ``True`` (default), :meth:`read` divides the raw read-out by the
        per-query key mass so recalled content sits at the written magnitude
        regardless of how peaked the place code is. If ``False``, returns the
        raw ``key @ M`` (useful when you want recall strength to scale with how
        well the query matches written locations).
    place_code:
        An existing :class:`PlaceCellCode` to *share* with L1 (so the memory is
        addressed by the same fields the encoder uses). If ``None``, one is
        constructed from the parameters above.
    eps:
        Numerical floor for the normalisation denominator.
    """

    def __init__(
        self,
        d_model: int,
        n_place: int = 512,
        arena: float = 2.2,
        coord_dim: int = 2,
        sigma: float = 0.12,
        mode: str = "gaussian",
        decay: float = 1.0,
        normalize: bool = True,
        place_code: Optional[PlaceCellCode] = None,
        eps: float = 1e-6,
        seed: int = 123,
    ):
        super().__init__()
        if d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {d_model}")
        if not (0.0 < decay <= 1.0):
            raise ValueError(
                f"decay must be in the half-open interval (0, 1], got {decay}"
            )

        if place_code is None:
            place_code = PlaceCellCode(
                n_place=n_place,
                arena=arena,
                coord_dim=coord_dim,
                sigma=sigma,
                mode=mode,
                seed=seed,
            )
        # The place code carries no trainable params; keep it as a submodule so
        # its centre buffer moves with .to()/.cuda() and serialises with us.
        self.place_code = place_code

        self.d_model = int(d_model)
        self.n_place = int(place_code.n_place)
        self.coord_dim = int(place_code.coord_dim)
        self.decay = float(decay)
        self.normalize = bool(normalize)
        self.eps = float(eps)

        # The associative map M:(n_place, d_model) and the accumulated key mass
        # per field k_mass:(n_place,) — both non-persistent runtime state that
        # is rebuilt by reset()/load_blob() rather than shipped in checkpoints.
        self.register_buffer(
            "M", torch.zeros(self.n_place, self.d_model), persistent=False
        )
        self.register_buffer(
            "k_mass", torch.zeros(self.n_place), persistent=False
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def _key(self, pos: torch.Tensor) -> torch.Tensor:
        """``(B, N, coord_dim) -> (B, N, n_place)`` soft place-cell key."""
        if pos.dim() < 2 or pos.shape[-1] != self.coord_dim:
            raise ValueError(
                f"expected positions (..., coord_dim={self.coord_dim}), "
                f"got shape {tuple(pos.shape)}"
            )
        return self.place_code(pos)

    def write(self, pos: torch.Tensor, content: torch.Tensor) -> "SpatialMemory":
        """Store *content* at *pos* via a Hebbian outer-product accumulation.

        ``pos`` is ``(B, N, coord_dim)`` and ``content`` is ``(B, N, d_model)``.
        For each location we add ``key ⊗ content`` into the shared map ``M`` and
        track the key mass for later normalisation. Returns ``self`` so writes
        chain.
        """
        if content.shape[-1] != self.d_model:
            raise ValueError(
                f"content last dim must be d_model={self.d_model}, "
                f"got {tuple(content.shape)}"
            )
        if content.shape[:-1] != pos.shape[:-1]:
            raise ValueError(
                f"pos and content must share leading dims, got "
                f"{tuple(pos.shape[:-1])} vs {tuple(content.shape[:-1])}"
            )

        key = self._key(pos)                       # (B, N, P)
        key = key.to(self.M.dtype)
        content = content.to(self.M.dtype)

        # Flatten the (B, N) location axis: every written point contributes one
        # outer product. delta_M:(P, d_model) = sum over points of key⊗content.
        kf = key.reshape(-1, self.n_place)         # (BN, P)
        cf = content.reshape(-1, self.d_model)     # (BN, d)
        delta_M = kf.t() @ cf                       # (P, d)
        delta_mass = kf.sum(dim=0)                  # (P,)

        if self.decay != 1.0:
            self.M.mul_(self.decay)
            self.k_mass.mul_(self.decay)
        self.M.add_(delta_M)
        self.k_mass.add_(delta_mass)
        return self

    def read(self, pos: torch.Tensor) -> torch.Tensor:
        """Recall content at *pos* by pattern completion.

        ``(B, M, coord_dim) -> (B, M, d_model)``. Computes the key-weighted
        read-out ``key @ M``; when ``normalize`` is set, divides by the query's
        effective key mass (``key·k_mass``) so the recalled vector lands at the
        written magnitude. Unwritten regions read out as (near-)zero.
        """
        key = self._key(pos).to(self.M.dtype)      # (B, M, P)
        out = key @ self.M                          # (B, M, d)
        if self.normalize:
            denom = (key * self.k_mass).sum(dim=-1, keepdim=True)  # (B, M, 1)
            out = out / denom.clamp_min(self.eps)
        return out

    # Semantic alias: produce fuse()-ready recalled-memory tokens.
    def recall_tokens(self, pos: torch.Tensor) -> torch.Tensor:
        """Alias of :meth:`read` — ``(B, M, d_model)`` tokens for ``fuse()``."""
        return self.read(pos)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """``nn.Module`` entry point == :meth:`read` (recall by position)."""
        return self.read(pos)

    # ------------------------------------------------------------------
    # State management & persistence
    # ------------------------------------------------------------------

    def reset(self) -> "SpatialMemory":
        """Clear the map (forget everything). Returns ``self``."""
        self.M.zero_()
        self.k_mass.zero_()
        return self

    @property
    def occupancy(self) -> torch.Tensor:
        """Per-field accumulated key mass ``(n_place,)`` — how 'written' each
        place field is. Useful for diagnostics / coverage plots."""
        return self.k_mass

    def state_blob(self) -> List[Optional[torch.Tensor]]:
        """Serialisable runtime state ``[M, k_mass]`` for :class:`SessionMemory`.

        ``SessionMemory`` persists a ``list[Tensor | None]``; this returns the
        two buffers that fully define the learned map. The place-cell centres
        are reconstructed from config (or the shared place code), so they are
        intentionally *not* part of the blob.
        """
        return [self.M.detach().cpu().clone(), self.k_mass.detach().cpu().clone()]

    def load_blob(self, blob: List[Optional[torch.Tensor]]) -> "SpatialMemory":
        """Restore state produced by :meth:`state_blob`. Returns ``self``."""
        if blob is None or len(blob) != 2 or blob[0] is None or blob[1] is None:
            raise ValueError("blob must be [M, k_mass] with both tensors present")
        M, k_mass = blob
        if tuple(M.shape) != (self.n_place, self.d_model):
            raise ValueError(
                f"M shape {tuple(M.shape)} != expected "
                f"{(self.n_place, self.d_model)}"
            )
        if tuple(k_mass.shape) != (self.n_place,):
            raise ValueError(
                f"k_mass shape {tuple(k_mass.shape)} != expected {(self.n_place,)}"
            )
        self.M.copy_(M.to(self.M.device, self.M.dtype))
        self.k_mass.copy_(k_mass.to(self.k_mass.device, self.k_mass.dtype))
        return self

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_place={self.n_place}, "
            f"coord_dim={self.coord_dim}, decay={self.decay}, "
            f"normalize={self.normalize}"
        )
