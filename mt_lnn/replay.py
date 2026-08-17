"""
mt_lnn/replay.py -- composable experience-replay buffers for streaming /
continual learning: bounded-memory, uniform-over-the-stream rehearsal (P2).

Why this module exists
----------------------
The MT-LNN core has *temporal* memory (the LTC recurrent state, persisted via
``SessionMemory``) and *spatial* associative memory (``SpatialMemory`` -- a
place-indexed Hebbian store). Neither is an **experience replay** buffer: a
bounded sample of *past training examples* that can be interleaved with the live
stream so a continually-trained model does not catastrophically forget what it
already learned. Rehearsal from such a buffer is the single most effective,
architecture-agnostic defence against catastrophic forgetting (Rolnick et al.
2019, "Experience Replay for Continual Learning"), and it mirrors hippocampal
replay re-presenting past episodes to neocortex during consolidation.

The streaming difficulty: the stream is *unbounded* but memory is *fixed*. We
cannot keep every example, and we do not know the stream length in advance. We
still want the retained sample to be an **unbiased, uniform** snapshot of
everything seen so far -- not just a recency window (a ring buffer over-weights
the present and is exactly what forgets the past).

The algorithm
-------------
:class:`ReservoirBuffer` is Vitter's Algorithm R (reservoir sampling). With
capacity ``C``, after the stream has offered ``n`` items:

* if ``n <= C`` the buffer holds all ``n`` items;
* if ``n  > C`` the buffer holds exactly ``C`` items, and **every one of the
  ``n`` items is present with probability ``C / n``** -- a provably uniform
  sample of the whole stream, using ``O(C)`` memory and ``O(1)`` work per item.

The update rule for the ``i``-th item (0-based stream index ``i``):

    if i < C:                       reservoir[i] = item            # fill phase
    else: draw j ~ Uniform{0..i};   if j < C: reservoir[j] = item  # replace phase

This uniformity invariant (not just "it stores stuff") is what is pinned
statistically in the tests.

:class:`ClassBalancedReservoir` composes one sub-reservoir per class over a
*known* set of labels, splitting the budget evenly. Under a class-imbalanced or
task-ordered stream a single reservoir inherits the stream's imbalance; an
equal-split-per-class reservoir guarantees each class keeps an unbiased sample of
its own examples (the class-balanced rehearsal of Chrysakis & Moens 2020, here in
its simplest fixed-class form).

Design / coupling
-----------------
* Both are small **stateful** containers with **zero trainable parameters**;
  neither is an ``nn.Module`` and neither imports ``model.py``. They import only
  the standard library + ``torch``. Storage is lazily-allocated dense tensor
  fields (one ``(capacity, *shape)`` buffer per named field), so a buffer holds
  arbitrary parallel tensors -- e.g. ``inputs_embeds`` and ``labels`` -- as one
  example.
* Fully **deterministic** given the seed, and fully **resettable** (``reset``).
  All randomness flows through a single owned ``torch.Generator``.
* ``sample`` draws a minibatch for interleaving with the live stream; the heavy
  lifting (the actual rehearsal loss) is the caller's training loop, kept out of
  here so this layer stays a pinned, model-free primitive.

Pinned against the analytic uniformity invariant and field bookkeeping in
``tests/test_replay.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor

__all__ = [
    "ReplayStats",
    "ReservoirBuffer",
    "ClassBalancedReservoir",
]


@dataclass(frozen=True)
class ReplayStats:
    """A read-only snapshot of a buffer's occupancy.

    Attributes
    ----------
    size:      number of examples currently held (``<= capacity``).
    capacity:  the buffer's fixed capacity.
    seen:      total number of examples ever offered to the buffer.
    fields:    tuple of the named tensor fields one example carries.
    """

    size: int
    capacity: int
    seen: int
    fields: tuple


# --------------------------------------------------------------------------- #
# 1. uniform reservoir (Vitter Algorithm R)                                   #
# --------------------------------------------------------------------------- #
class ReservoirBuffer:
    """Bounded-memory uniform sample of an unbounded stream (reservoir sampling).

    Each *example* is a set of named parallel tensors (e.g. ``x`` and ``y``); the
    field names and per-field shapes/dtypes are inferred from the first ``add``
    and fixed thereafter. Internally each field ``k`` is a dense
    ``(capacity, *shape_k)`` tensor; example ``s`` lives at row ``s`` across all
    fields, so rows stay aligned.

    Parameters
    ----------
    capacity:  maximum number of examples retained (``> 0``).
    seed:      seeds the owned ``torch.Generator`` for reproducible replacement
               and sampling.
    device:    device for the stored field tensors (default: the device of the
               first added tensor).

    Notes
    -----
    * ``add`` offers one example; ``add_batch`` offers a leading-axis batch,
      applying Algorithm R item by item (the textbook reference form -- correct
      even when several items in a batch would target the same slot).
    * Zero trainable parameters; not an ``nn.Module``.
    """

    def __init__(self, capacity: int, *, seed: int = 0,
                 device: Optional[torch.device] = None) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._capacity = int(capacity)
        self._seed = int(seed)
        self._device = None if device is None else torch.device(device)
        # the generator lives on CPU: randint / randperm are drawn on CPU then
        # used to index device tensors, which keeps the draw stream identical and
        # reproducible regardless of where the payload tensors live.
        self._gen = torch.Generator(device="cpu")
        self.reset()

    # -- lifecycle ---------------------------------------------------------- #
    def reset(self) -> None:
        """Empty the buffer, clear the inferred schema, and re-seed the RNG."""
        self._fields: Dict[str, Tensor] = {}
        self._size = 0      # examples currently stored (<= capacity)
        self._seen = 0      # examples ever offered (the stream index)
        self._gen.manual_seed(self._seed)

    # -- introspection ------------------------------------------------------ #
    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def seen(self) -> int:
        return self._seen

    @property
    def n_parameters(self) -> int:
        """Zero -- a reservoir has no learnable parameters."""
        return 0

    def __len__(self) -> int:
        return self._size

    def stats(self) -> ReplayStats:
        return ReplayStats(
            size=self._size,
            capacity=self._capacity,
            seen=self._seen,
            fields=tuple(sorted(self._fields)),
        )

    # -- schema ------------------------------------------------------------- #
    def _ensure_schema(self, example: Dict[str, Tensor]) -> None:
        if self._fields:                                   # already allocated
            missing = set(self._fields) ^ set(example)
            if missing:
                raise KeyError(
                    f"example fields {sorted(example)} do not match buffer "
                    f"schema {sorted(self._fields)}"
                )
            return
        if not example:
            raise ValueError("cannot add an example with no fields")
        for k, v in example.items():
            if not isinstance(v, Tensor):
                raise TypeError(f"field {k!r} must be a torch.Tensor, got {type(v)}")
            dev = self._device if self._device is not None else v.device
            self._fields[k] = torch.empty(
                (self._capacity, *v.shape), dtype=v.dtype, device=dev
            )
        if self._device is None:                           # lock device in
            self._device = next(iter(self._fields.values())).device

    # -- writes ------------------------------------------------------------- #
    def _write(self, slot: int, example: Dict[str, Tensor]) -> None:
        for k, v in example.items():
            self._fields[k][slot] = v.to(self._fields[k].device)

    def add(self, **example: Tensor) -> None:
        """Offer a single example (one tensor per named field) to the stream.

        Applies Algorithm R: fills empty slots first, then replaces a uniformly
        random existing slot with probability ``capacity / seen``.
        """
        self._ensure_schema(example)
        i = self._seen
        if self._size < self._capacity:                    # fill phase
            self._write(self._size, example)
            self._size += 1
        else:                                              # replace phase
            j = int(torch.randint(0, i + 1, (1,), generator=self._gen).item())
            if j < self._capacity:
                self._write(j, example)
        self._seen += 1

    def add_batch(self, **batch: Tensor) -> None:
        """Offer a leading-axis batch; each row becomes one example.

        All fields must share the same batch size ``B``; row ``r`` of every field
        forms example ``r``. Items are absorbed sequentially (Algorithm R), so
        the uniformity invariant holds across batch boundaries.
        """
        if not batch:
            raise ValueError("add_batch needs at least one field")
        sizes = {k: v.shape[0] for k, v in batch.items()}
        b = next(iter(sizes.values()))
        if any(s != b for s in sizes.values()):
            raise ValueError(f"mismatched batch sizes across fields: {sizes}")
        for r in range(b):
            self.add(**{k: v[r] for k, v in batch.items()})

    # -- reads -------------------------------------------------------------- #
    def sample(self, k: int, *, replacement: bool = False) -> Dict[str, Tensor]:
        """Draw a minibatch of ``k`` examples for rehearsal.

        Returns a dict mapping each field name to a ``(m, *shape)`` tensor with
        rows aligned across fields. Without replacement ``m = min(k, len(self))``;
        with replacement ``m = k`` (an empty buffer raises). Order is shuffled.
        """
        if self._size == 0:
            raise IndexError("cannot sample from an empty buffer")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if replacement:
            idx = torch.randint(0, self._size, (k,), generator=self._gen)
        else:
            m = min(k, self._size)
            idx = torch.randperm(self._size, generator=self._gen)[:m]
        return {name: buf[: self._size][idx] for name, buf in self._fields.items()}

    def snapshot(self) -> Dict[str, Tensor]:
        """Return clones of all currently-stored rows (a ``(len, *shape)`` dict)."""
        return {name: buf[: self._size].clone() for name, buf in self._fields.items()}


# --------------------------------------------------------------------------- #
# 2. class-balanced reservoir (one sub-reservoir per known label)             #
# --------------------------------------------------------------------------- #
class ClassBalancedReservoir:
    """An equal-budget reservoir per class, for class-/task-imbalanced streams.

    The total ``capacity`` is split evenly across ``n_classes`` independent
    :class:`ReservoirBuffer` sub-reservoirs (each gets ``capacity // n_classes``
    slots; any remainder is left unused so every class is treated identically).
    Each class therefore keeps an unbiased uniform sample of *its own* examples,
    regardless of how skewed the arrival rates are -- the failure mode a single
    shared reservoir suffers under a task-ordered stream.

    Parameters
    ----------
    capacity:   total budget across all classes.
    n_classes:  number of distinct integer labels in ``[0, n_classes)``.
    seed:       base seed; sub-reservoir ``c`` is seeded ``seed + c`` so the
                classes do not share a draw stream.
    device:     device for stored tensors (passed through to each sub-reservoir).

    Notes
    -----
    Zero trainable parameters; not an ``nn.Module``. Composition of
    :class:`ReservoirBuffer`, not new machinery.
    """

    def __init__(self, capacity: int, n_classes: int, *, seed: int = 0,
                 device: Optional[torch.device] = None) -> None:
        if n_classes <= 0:
            raise ValueError(f"n_classes must be positive, got {n_classes}")
        per_class = capacity // n_classes
        if per_class <= 0:
            raise ValueError(
                f"capacity {capacity} too small for {n_classes} classes "
                f"(need at least one slot each)"
            )
        self._n_classes = int(n_classes)
        self._per_class = per_class
        self._seed = int(seed)
        self._device = device
        self.reset()

    def reset(self) -> None:
        """Empty every sub-reservoir."""
        self._sub = [
            ReservoirBuffer(self._per_class, seed=self._seed + c, device=self._device)
            for c in range(self._n_classes)
        ]

    @property
    def n_classes(self) -> int:
        return self._n_classes

    @property
    def capacity(self) -> int:
        """Effective capacity actually usable (``per_class * n_classes``)."""
        return self._per_class * self._n_classes

    @property
    def n_parameters(self) -> int:
        return 0

    def __len__(self) -> int:
        return sum(len(s) for s in self._sub)

    def per_class_sizes(self) -> Dict[int, int]:
        """Map each class id to how many of its examples are currently held."""
        return {c: len(s) for c, s in enumerate(self._sub)}

    def add(self, label: int, **example: Tensor) -> None:
        """Route one example to the sub-reservoir of its integer ``label``."""
        c = int(label)
        if not 0 <= c < self._n_classes:
            raise ValueError(f"label {c} outside [0, {self._n_classes})")
        self._sub[c].add(**example)

    def sample(self, k: int, *, replacement: bool = False) -> Dict[str, Tensor]:
        """Draw ~``k`` examples spread as evenly as possible across non-empty
        classes, then concatenate. Returns the same field-dict layout as
        :meth:`ReservoirBuffer.sample`, with a ``label`` field added that records
        each drawn row's class. Rows are shuffled so classes are interleaved.
        """
        if len(self) == 0:
            raise IndexError("cannot sample from an empty buffer")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        live = [c for c, s in enumerate(self._sub) if len(s) > 0]
        base, extra = divmod(k, len(live))
        parts: list = []
        labels: list = []
        for n, c in enumerate(live):
            want = base + (1 if n < extra else 0)
            if want == 0:
                continue
            drawn = self._sub[c].sample(want, replacement=replacement)
            m = len(next(iter(drawn.values())))
            parts.append(drawn)
            labels.append(torch.full((m,), c, dtype=torch.long))
        out: Dict[str, Tensor] = {
            name: torch.cat([p[name] for p in parts], dim=0) for name in parts[0]
        }
        out["label"] = torch.cat(labels, dim=0)
        # shuffle so the concatenated class blocks are interleaved
        total = out["label"].shape[0]
        perm = torch.randperm(total, generator=self._sub[live[0]]._gen)
        return {name: v[perm] for name, v in out.items()}
