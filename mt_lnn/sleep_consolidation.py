"""
mt_lnn/sleep_consolidation.py — offline sleep-wake memory consolidation.

Why this module exists
----------------------
A continually-running edge model accumulates experience all day ("wake"). The
brain does not keep that experience in its fast, volatile form forever — during
**sleep** it runs an offline maintenance cycle that (1) replays the day's
episodes and ships the important ones from fast hippocampal memory into slow
neocortical storage, (2) globally *downscales* synaptic weight to restore
homeostasis and signal-to-noise, and (3) recombines fragments into novel
configurations. This module is that cycle for MT-LNN, built on machinery the
codebase already has:

* **NREM replay → consolidation** (Wilson & McNaughton 1994). Sample past
  episodes from an experience-replay :class:`mt_lnn.replay.ReservoirBuffer` and
  write the most salient ones, by content, into the long-term
  :class:`mt_lnn.knowledge_memory.PersistentKnowledgeMemory`. This is the
  fast→slow (hippocampus→neocortex) transfer, turning a bounded, volatile replay
  sample into durable, content-addressable knowledge.
* **Synaptic downscaling / SHY** (Tononi & Cirelli 2014, "Synaptic Homeostasis
  Hypothesis"). Wake potentiates synapses; sleep renormalises them *proportionally*
  downward, preserving the *relative* pattern (the memory) while shrinking total
  synaptic weight (restoring capacity and SNR). Implemented as a multiplicative
  weight rescale — direction-preserving by construction.
* **REM generative recombination** (Diekelmann & Born 2010). Convex blends of
  pairs of replayed latents synthesise novel "dreamed" examples for generative
  augmentation.

Design / coupling (same discipline as replay.py / salience_events.py)
---------------------------------------------------------------------
* **Zero model coupling.** Imports only ``torch`` + stdlib. It never imports
  ``model.py``. The replay buffer, the knowledge store and any module to be
  downscaled are reached by **duck-typing** (``.sample`` / ``.write`` /
  ``.named_parameters``), so this orchestrator is decoupled from all of them.
* **Offline / opt-in / reversible.** Nothing in MT-LNN runs a consolidation
  cycle automatically — a caller invokes it between sessions. Synaptic
  downscaling mutates weights *only* when explicitly called and is a pure
  multiplicative rescale (a recorded ``factor`` makes it analysable and, if the
  caller keeps the factor, exactly invertible).
* **Deterministic.** All randomness flows through a single owned
  ``torch.Generator`` seeded at construction; ``reset`` re-seeds it.

Pinned in tests/test_sleep_consolidation.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import torch.nn as nn

__all__ = [
    "ReplayConsolidationResult",
    "DownscaleResult",
    "ConsolidationReport",
    "SleepWakeConsolidator",
]


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReplayConsolidationResult:
    """Outcome of an NREM replay → long-term consolidation pass."""

    replayed: int          # episodes drawn from the replay buffer
    consolidated: int      # episodes written into the knowledge store


@dataclass(frozen=True)
class DownscaleResult:
    """Outcome of a synaptic-downscaling (SHY) pass."""

    factor: float          # multiplicative rescale applied
    n_tensors: int         # weight tensors rescaled
    pre_norm: float        # global L2 norm of the rescaled weights before
    post_norm: float       # …and after (== factor * pre_norm)


@dataclass(frozen=True)
class ConsolidationReport:
    """Combined report from a full :meth:`SleepWakeConsolidator.consolidate` cycle."""

    replay: Optional[ReplayConsolidationResult]
    downscale: Optional[DownscaleResult]
    recombined: int        # number of REM generative samples produced


# ---------------------------------------------------------------------------
# SleepWakeConsolidator
# ---------------------------------------------------------------------------

class SleepWakeConsolidator:
    """Run an offline sleep-wake consolidation cycle over replay/knowledge tiers.

    Parameters
    ----------
    downscale_factor : float
        Default multiplicative weight rescale for :meth:`synaptic_downscale`
        (SHY). Must be in ``(0, 1]``; ``1.0`` is a no-op (downscaling disabled).
    seed : int
        Seed for the owned RNG (replay sampling, REM recombination).
    """

    def __init__(self, *, downscale_factor: float = 0.9, seed: int = 0) -> None:
        if not (0.0 < downscale_factor <= 1.0):
            raise ValueError(
                f"downscale_factor must be in (0, 1], got {downscale_factor}"
            )
        self.downscale_factor = float(downscale_factor)
        self._seed = int(seed)
        self._gen = torch.Generator(device="cpu")
        self.reset()

    def reset(self) -> None:
        """Re-seed the owned RNG so a cycle is reproducible."""
        self._gen.manual_seed(self._seed)

    # ------------------------------------------------------------------
    # 1. NREM replay → long-term consolidation
    # ------------------------------------------------------------------

    def nrem_replay(
        self,
        buffer,
        knowledge_memory,
        *,
        key_field: str,
        content_field: str,
        meta_field: Optional[str] = None,
        salience_field: Optional[str] = None,
        n_replay: Optional[int] = None,
        consolidate_fraction: float = 1.0,
    ) -> ReplayConsolidationResult:
        """Replay episodes from ``buffer`` and consolidate the salient ones.

        Draws a replay batch from a duck-typed experience buffer (anything with a
        ``sample(k)`` returning a ``{field: (m, *shape)}`` dict, e.g.
        :class:`mt_lnn.replay.ReservoirBuffer`), ranks the episodes by salience,
        and writes the top ``consolidate_fraction`` of them into a duck-typed
        long-term store (anything with ``write(key, content, meta)``, e.g.
        :class:`mt_lnn.knowledge_memory.PersistentKnowledgeMemory`).

        Parameters
        ----------
        key_field / content_field :
            Field names whose rows become the knowledge key (an embedding) and
            the stored content, respectively.
        meta_field :
            Optional field stored as each record's metadata.
        salience_field :
            Optional scalar field used to rank episodes (higher = more salient,
            consolidated first). When ``None`` every drawn episode is equally
            salient and the first ``n_keep`` in sample order are consolidated.
        n_replay :
            How many episodes to draw (default: the whole buffer).
        consolidate_fraction :
            Fraction of the replayed episodes to persist, in ``[0, 1]``. ``1.0``
            consolidates everything replayed; smaller keeps only the most salient
            (the brain consolidates a *subset* of the night's replay).
        """
        if not (0.0 <= consolidate_fraction <= 1.0):
            raise ValueError(
                f"consolidate_fraction must be in [0, 1], got {consolidate_fraction}"
            )
        n_have = len(buffer)
        if n_have == 0:
            return ReplayConsolidationResult(replayed=0, consolidated=0)

        n = n_have if n_replay is None else min(int(n_replay), n_have)
        batch = buffer.sample(n, replacement=False)
        if key_field not in batch or content_field not in batch:
            raise KeyError(
                f"replay batch is missing required fields "
                f"{key_field!r}/{content_field!r}; has {sorted(batch)}"
            )
        keys = batch[key_field]
        m = keys.shape[0]

        # Rank by salience (descending). Without a salience field, preserve the
        # sampled order so the pass is deterministic.
        if salience_field is not None:
            sal = batch[salience_field].reshape(m).float()
            order = torch.argsort(sal, descending=True)
        else:
            order = torch.arange(m)

        n_keep = int(round(consolidate_fraction * m))
        if consolidate_fraction > 0.0:
            n_keep = max(1, n_keep)         # consolidate at least one if asked to
        order = order[:n_keep]

        written = 0
        for i in order.tolist():
            content = batch[content_field][i]
            meta = batch[meta_field][i] if meta_field is not None else None
            knowledge_memory.write(keys[i], content, meta)
            written += 1
        return ReplayConsolidationResult(replayed=m, consolidated=written)

    # ------------------------------------------------------------------
    # 2. Synaptic downscaling (SHY)
    # ------------------------------------------------------------------

    def synaptic_downscale(
        self,
        target,
        *,
        factor: Optional[float] = None,
        protect: Optional[Callable[[str], bool]] = None,
    ) -> DownscaleResult:
        """Multiplicatively renormalise weights downward (synaptic homeostasis).

        Scales each weight tensor by ``factor`` in place. Because the rescale is
        purely multiplicative it preserves the *relative* weight pattern exactly
        (cosine of before/after == 1): the memory is kept, only its overall
        magnitude is shrunk — the core SHY claim that sleep restores capacity and
        SNR without erasing what was learned.

        Parameters
        ----------
        target :
            An ``nn.Module`` (its ``named_parameters`` are rescaled) or an
            iterable of weight tensors.
        factor :
            Override the constructor's ``downscale_factor``. ``1.0`` → no-op.
        protect :
            Optional predicate on a parameter's name; when it returns True that
            parameter is left untouched. Only applies to ``nn.Module`` targets
            (e.g. ``lambda n: "norm" in n or n.endswith("bias")`` to spare
            normalisation/bias terms, as homeostatic downscaling targets
            excitatory synaptic *weights*).
        """
        f = self.downscale_factor if factor is None else float(factor)
        if not (0.0 < f <= 1.0):
            raise ValueError(f"factor must be in (0, 1], got {f}")

        if isinstance(target, nn.Module):
            named = list(target.named_parameters())
        else:
            named = [(str(i), t) for i, t in enumerate(target)]

        pre_sq = 0.0
        post_sq = 0.0
        n_tensors = 0
        with torch.no_grad():
            for name, p in named:
                if protect is not None and protect(name):
                    continue
                if not torch.is_tensor(p):
                    raise TypeError(f"target item {name!r} is not a tensor")
                pre_sq += float(p.detach().pow(2).sum())
                p.mul_(f)
                post_sq += float(p.detach().pow(2).sum())
                n_tensors += 1
        return DownscaleResult(
            factor=f,
            n_tensors=n_tensors,
            pre_norm=pre_sq ** 0.5,
            post_norm=post_sq ** 0.5,
        )

    # ------------------------------------------------------------------
    # 3. REM generative recombination
    # ------------------------------------------------------------------

    def rem_recombine(
        self,
        latents: torch.Tensor,
        *,
        n_samples: int,
        alpha: Optional[float] = None,
    ) -> torch.Tensor:
        """Synthesise novel latents by convex-blending random pairs ("dreaming").

        Given ``latents`` of shape ``(m, *feat)``, draws ``n_samples`` random
        pairs and returns ``alpha * a + (1 - alpha) * b`` for each — novel points
        on the segments between real experiences, the generative-augmentation
        analogue of REM recombination. With ``alpha=None`` a fresh random blend
        weight is drawn per sample (more diverse); a fixed ``alpha`` gives the
        deterministic midpoint family.

        Returns ``(n_samples, *feat)``.
        """
        if latents.dim() < 1 or latents.shape[0] < 1:
            raise ValueError("latents must have a non-empty leading (batch) axis")
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        m = latents.shape[0]
        ia = torch.randint(0, m, (n_samples,), generator=self._gen)
        ib = torch.randint(0, m, (n_samples,), generator=self._gen)
        a = latents[ia]
        b = latents[ib]
        if alpha is None:
            w = torch.rand(n_samples, generator=self._gen, dtype=a.dtype)
            # broadcast (n_samples,) over the feature dims
            w = w.reshape((n_samples,) + (1,) * (a.dim() - 1))
        else:
            if not (0.0 <= float(alpha) <= 1.0):
                raise ValueError(f"alpha must be in [0, 1], got {alpha}")
            w = float(alpha)
        return w * a + (1.0 - w) * b

    # ------------------------------------------------------------------
    # Full cycle
    # ------------------------------------------------------------------

    def consolidate(
        self,
        *,
        buffer=None,
        knowledge_memory=None,
        replay_kwargs: Optional[dict] = None,
        module=None,
        downscale_factor: Optional[float] = None,
        protect: Optional[Callable[[str], bool]] = None,
        rem_latents: Optional[torch.Tensor] = None,
        rem_samples: int = 0,
    ) -> ConsolidationReport:
        """Run a full NREM→SHY→REM cycle; each stage is optional.

        Stage 1 (NREM) runs only if both ``buffer`` and ``knowledge_memory`` are
        given (with ``replay_kwargs`` forwarded to :meth:`nrem_replay`). Stage 2
        (SHY) runs only if ``module`` is given. Stage 3 (REM) runs only if
        ``rem_latents`` is given and ``rem_samples > 0``. Returns a combined
        :class:`ConsolidationReport`.
        """
        replay_res = None
        if buffer is not None and knowledge_memory is not None:
            replay_res = self.nrem_replay(
                buffer, knowledge_memory, **(replay_kwargs or {})
            )

        downscale_res = None
        if module is not None:
            downscale_res = self.synaptic_downscale(
                module, factor=downscale_factor, protect=protect
            )

        n_recombined = 0
        if rem_latents is not None and rem_samples > 0:
            dreamed = self.rem_recombine(rem_latents, n_samples=rem_samples)
            n_recombined = dreamed.shape[0]

        return ConsolidationReport(
            replay=replay_res,
            downscale=downscale_res,
            recombined=n_recombined,
        )

    def __repr__(self) -> str:
        return (
            f"SleepWakeConsolidator(downscale_factor={self.downscale_factor}, "
            f"seed={self._seed})"
        )
