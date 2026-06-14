"""
mt_lnn/sensory_frontend.py -- raw streaming-sensor frontend (P1 closed-loop).

Why this module exists
----------------------
The backbone is modality-agnostic: anything shaped ``(B, T, d_model)`` can be
fed through ``MTLNNModel.forward(inputs_embeds=...)`` (the documented injection
point). ``multimodal.py`` already turns *feature blocks* into d_model tokens.
What a **closed perception loop** additionally needs is the *temporal* front:
a real sensor does not arrive on the liquid core's fixed-``dt`` lattice -- it is
jittered, occasionally drops frames, and drifts. This module is that missing
temporal frontend. It composes two pieces that already exist, with zero new
backbone coupling:

    ingest_ops.align_stream   -- resample a jittered, timestamped raw stream onto
                                 the core's fixed-``dt`` grid AND mark which steps
                                 are real-sample-backed vs deep inside a dropout;
    multimodal.ModalityProjector -- project the aligned per-step feature vector
                                 into the model's ``d_model`` token space.

The output is a :class:`SensoryEncoding`: the ``(B, M, d_model)`` ``inputs_embeds``
ready for the backbone, plus the coverage mask surfaced two ways -- as a
``pad_mask`` (so dropout steps don't pollute attention) and as the *trust* signal
that :class:`mt_lnn.failsafe.BlindRolloutGuard` coasts on. That closes the loop:
raw stream -> aligned tokens -> backbone, with the gaps handled honestly.

Design / coupling
-----------------
* This is a **trainable** ``nn.Module`` (it owns a ``ModalityProjector``), exactly
  like the encoders in ``multimodal.py`` -- wrap it alongside the model in your
  optimizer. It produces ``inputs_embeds``; it **never imports or modifies
  ``model.py``**. The temporal alignment is pure (zero-parameter) ``ingest_ops``.
* Two input regimes: pass raw ``(values, timestamps)`` to align onto the grid, or
  pass values that are already uniform (``timestamps=None``) to skip alignment and
  just project. Batched inputs share one timestamp clock (synchronised sensors);
  use :meth:`SensoryFrontend.align_batch` for heterogeneous per-item clocks.
* Honest scope: a working, tested temporal injection path -- not a pretrained
  perceptual tower. Linear resampling is exact on affine signals only (documented
  in ``ingest_ops``); a too-wide gap is *flagged*, not invented.

Pinned against analytic alignment + projection ground truth in
``tests/test_sensory_frontend.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .ingest_ops import AlignedStream, align_stream
from .multimodal import ModalityProjector

__all__ = ["SensoryEncoding", "SensoryFrontend"]


@dataclass
class SensoryEncoding:
    """The frontend's output: aligned tokens + how much to trust each step.

    Attributes
    ----------
    embeds : (B, M, d_model)
        The ``inputs_embeds`` to feed ``MTLNNModel.forward(inputs_embeds=...)``.
    covered : (B, M) bool
        ``True`` where a real sample backs the step (trust it), ``False`` inside a
        dropout (coast: hand to :class:`~mt_lnn.failsafe.BlindRolloutGuard`).
    pad_mask : (B, M) bool
        Convenience alias of ``covered`` in the backbone's pad-mask convention
        (``True`` == valid/keep), so dropout steps can be masked out of attention.
    times : (M,)
        The uniform query times the stream was aligned to (``t_start + dt * k``).
    dt : float
        The grid spacing (the core's fixed step) the stream was aligned to.
    """

    embeds: torch.Tensor
    covered: torch.Tensor
    pad_mask: torch.Tensor
    times: torch.Tensor
    dt: float

    @property
    def n_steps(self) -> int:
        """Number of uniform steps produced (M)."""
        return int(self.embeds.shape[-2])

    @property
    def n_gap_steps(self) -> int:
        """Worst-case dropout steps over the batch (max over items)."""
        if self.covered.numel() == 0:
            return 0
        return int((~self.covered).sum(dim=-1).max().item())

    @property
    def fully_covered(self) -> bool:
        """True when no step in any batch item fell inside an untrusted gap."""
        return bool(self.covered.all().item())

    @property
    def coverage_ratio(self) -> float:
        """Fraction of steps that are real-sample-backed (1.0 == no dropout)."""
        if self.covered.numel() == 0:
            return 1.0
        return float(self.covered.float().mean().item())


class SensoryFrontend(nn.Module):
    """Raw timestamped sensor stream -> ``inputs_embeds`` for the backbone.

    Composes the (zero-parameter) temporal alignment :func:`ingest_ops.align_stream`
    with a trainable :class:`multimodal.ModalityProjector`. See the module
    docstring for the rationale.

    Parameters
    ----------
    in_dim : int
        Feature dimension ``F`` of one raw sensor sample.
    d_model : int
        The backbone's token width (``config.d_model``).
    dt : float
        The liquid core's fixed timestep to align the stream onto (> 0).
    max_gap : float, optional
        Coverage threshold handed to :func:`align_stream` (default ``1.5 * dt``):
        one slightly-late sample is tolerated, a hole >= 2 steps wide is flagged.
    mode : {"linear", "previous"}
        Resampling kernel (linear interpolation, or zero-order hold).
    dropout : float
        Dropout on the projected tokens.
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int,
        *,
        dt: float,
        max_gap: Optional[float] = None,
        mode: str = "linear",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not (float(dt) > 0.0):
            raise ValueError(f"dt must be > 0, got {dt}")
        if max_gap is not None and not (float(max_gap) > 0.0):
            raise ValueError(f"max_gap must be > 0 or None, got {max_gap}")
        if mode not in ("linear", "previous"):
            raise ValueError(f"mode must be 'linear' or 'previous', got {mode!r}")
        self.in_dim = int(in_dim)
        self.d_model = int(d_model)
        self.dt = float(dt)
        self.max_gap = None if max_gap is None else float(max_gap)
        self.mode = mode
        self.projector = ModalityProjector(in_dim, d_model, dropout=dropout)

    # ------------------------------------------------------------------ #
    # alignment helpers (pure ingest_ops, no parameters)                 #
    # ------------------------------------------------------------------ #

    def align(
        self,
        values: torch.Tensor,
        timestamps: torch.Tensor,
        *,
        n_steps: Optional[int] = None,
        t_start: Optional[float] = None,
    ) -> AlignedStream:
        """Align ONE raw stream ``(T, F)`` onto the fixed-``dt`` grid.

        Thin pass-through to :func:`ingest_ops.align_stream` with this frontend's
        ``dt`` / ``max_gap`` / ``mode``. Returns the :class:`AlignedStream`
        (uniform values, query times, coverage mask).
        """
        return align_stream(
            values, timestamps, dt=self.dt, max_gap=self.max_gap,
            t_start=t_start, n_steps=n_steps, mode=self.mode,
        )

    def align_batch(
        self,
        streams: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        *,
        n_steps: int,
        t_start: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Align a batch of HETEROGENEOUS per-item streams onto a common grid.

        ``streams`` is a list of ``(values_i (T_i, F), timestamps_i (T_i,))`` with
        possibly different clocks. Every item is resampled onto the SAME grid
        (``t_start + dt * k`` for ``k`` in ``range(n_steps)``) so the results stack.
        Returns ``(values (B, n_steps, F), covered (B, n_steps) bool)``.
        """
        if n_steps is None or int(n_steps) < 1:
            raise ValueError(f"align_batch needs n_steps >= 1, got {n_steps}")
        vals: List[torch.Tensor] = []
        covs: List[torch.Tensor] = []
        for v, ts in streams:
            a = self.align(v, ts, n_steps=int(n_steps), t_start=t_start)
            av = a.values if a.values.dim() == 2 else a.values.unsqueeze(-1)
            vals.append(av)
            covs.append(a.covered)
        return torch.stack(vals, dim=0), torch.stack(covs, dim=0)

    # ------------------------------------------------------------------ #
    # forward                                                            #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        values: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        *,
        n_steps: Optional[int] = None,
        t_start: Optional[float] = None,
    ) -> SensoryEncoding:
        """Encode a raw sensor stream into a :class:`SensoryEncoding`.

        Two regimes:

        * ``timestamps`` given -> align the jittered stream onto the fixed-``dt``
          grid first (one shared clock for the whole batch), then project. The
          coverage mask flags dropout steps.
        * ``timestamps is None`` -> ``values`` is assumed already uniform on the
          grid; skip alignment, project directly, every step is ``covered``.

        Parameters
        ----------
        values : (T, F) or (B, T, F)
            Raw (or already-uniform) sensor features. A missing batch dim is added
            and squeezed back out of ``embeds`` on return.
        timestamps : (T,), optional
            Strictly-increasing seconds shared across the batch. ``None`` to skip
            alignment.
        n_steps, t_start : forwarded to :func:`align_stream` when aligning.
        """
        if values.dim() == 2:
            values = values.unsqueeze(0)
            had_batch = False
        elif values.dim() == 3:
            had_batch = True
        else:
            raise ValueError(
                f"values must be (T, F) or (B, T, F), got {tuple(values.shape)}"
            )
        B = values.shape[0]

        if timestamps is None:
            # already uniform: project as-is, all steps trusted.
            uniform = values
            M = uniform.shape[1]
            covered = torch.ones(B, M, dtype=torch.bool, device=values.device)
            times = (
                torch.arange(M, dtype=values.dtype, device=values.device) * self.dt
                + (0.0 if t_start is None else float(t_start))
            )
        else:
            if timestamps.dim() != 1:
                raise ValueError(
                    f"timestamps must be 1-D (shared clock), got {tuple(timestamps.shape)}"
                )
            # align each batch item onto the SAME grid (shared timestamps -> the
            # coverage mask is identical across the batch; compute it once).
            aligned_vals: List[torch.Tensor] = []
            covered_row: Optional[torch.Tensor] = None
            times: Optional[torch.Tensor] = None
            for b in range(B):
                a = self.align(values[b], timestamps, n_steps=n_steps, t_start=t_start)
                av = a.values if a.values.dim() == 2 else a.values.unsqueeze(-1)
                aligned_vals.append(av)
                if covered_row is None:
                    covered_row, times = a.covered, a.times
            uniform = torch.stack(aligned_vals, dim=0)            # (B, M, F)
            covered = covered_row.unsqueeze(0).expand(B, -1).contiguous()

        embeds = self.projector(uniform)                          # (B, M, d_model)
        if not had_batch:
            embeds = embeds.squeeze(0)
        return SensoryEncoding(
            embeds=embeds, covered=covered, pad_mask=covered,
            times=times, dt=self.dt,
        )
