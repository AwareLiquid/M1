"""ParametricMemory — an agent-memory runtime whose store IS its parameters.

External agent-memory systems (Mem0, Letta, Zep) keep memory OUTSIDE the
model: vector stores, files, plain-text logs. Storage grows O(n) with
history, every recall scans O(n) candidates, "delete my data" means editing
rows/logs, and the state is only as portable as the store's exporter. This
module packages the capability this repo has already proven at the LM level
(cross-window recall 0.56 vs a structural 0.000 for attention/LoRA —
RESULTS.md "Cross-window associative recall") into a standalone runtime with
a Mem0/Letta-shaped interface whose memory body is a fast-weight associative
matrix instead of a database:

    mem = ParametricMemory()
    mem.write("s1", key=5001, value=7234)               # token-id path
    mem.write("s1", key="favorite color", value="blue")  # text path
    mem.recall("s1", query=5001, top_k=3)
    mem.forget("s1", key=5001)            # ONE binding, surgically
    mem.forget("s1")                      # whole session -> zeroed
    mem.save("s1", "s1.json")             # bit-exact, via session_state.py
    mem.load("s1", "s1.json")

The per-session state is the same (F, z) pair FastWeightMemoryV2 carries
(mt_lnn/mt_lnn_v2.py), with both write rules this repo has trained:

    sum   (mt_v2)        F <- decay * F + k v^T        z <- decay * z + k
                        read r = (q F) / (q . z + eps)
    delta (mt_v2_delta)  F <- decay * F - eta * k (k^T F - v)^T
                        read r = q F   (self-normalising regression)

``sum`` is the classic outer-product accumulator (the 0.56 anchor mechanism);
``delta`` is gradient-as-memory (DeltaNet/Titans style) — it CORRECTS the
existing association along k instead of only adding, which is why repeated
writes to one key resolve to the LATEST value and why ``forget`` can remove a
single binding exactly. Both are exposed as one switch (constructor default +
per-write override) so callers can A/B the two trained mechanisms.

Honest complexity notes (see docs/PARAMETRIC_MEMORY.md):
  * state_bytes(session) is (d^2 + d) floats — CONSTANT in the number of
    writes. No index of bindings is kept; forget() is algebra, not deletion.
  * The session REGISTRY grows with the number of sessions (one fixed-size
    (F, z) each) — per-session memory is O(1) in writes, O(#sessions) total.
  * Text VALUES need a decode lexicon (a read vector must map back to text).
    That lexicon is a FIXED-CAPACITY LRU (``lexicon_capacity``) — bounded,
    unlike an append-only log. Token-id values decode against the fixed
    embedding table and need no lexicon at all.
  * The default text encoder is a deterministic feature hash (exact-match
    keys, zero downloads). Inject a real sentence encoder (e.g.
    mt_lnn.sentence_encoder) for semantic recall quality — same contract as
    fast_weight_store.build_session_key's injected encode_fn.

This is v0: single-head, no thread safety (wrap in the caller's lock, as
serve/ does), decay defaults to 1.0 (nothing is forgotten passively —
forgetting is explicit, which is the product surface being sold).
"""

from __future__ import annotations

import base64
import hashlib
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

from .session_state import (
    HFSessionState,
    load_session as _load_hf_session,
    save_session as _save_hf_session,
)

MemoryValue = Union[int, str]

_SNAP_FORMAT = "parametric-memory-v1"
_EPS = 1e-8


def _hash_vec(text: str, dim: int) -> torch.Tensor:
    """Deterministic unit vector for a string (feature hashing, no download).

    Same construction as fast_weight_store.id_key: sha256-hash the text into
    ``dim`` signed buckets, then L2-normalise. Identical strings map to the
    same direction; distinct strings are near-orthogonal in high dimension.
    Exact-match keying only — semantic neighbours need a real encoder."""
    v = torch.zeros(dim, dtype=torch.float32)
    for i in range(max(dim, 32)):
        h = hashlib.sha256(f"{text}:{i}".encode()).digest()
        idx = int.from_bytes(h[:4], "big") % dim
        v[idx] += 1.0 if h[4] & 1 else -1.0
    return v / v.norm().clamp_min(1e-6)


class _Session:
    """One session's parametric state: (F, z) + the text-value decode lexicon."""

    __slots__ = ("F", "z", "lexicon", "read_rule")

    def __init__(self, d: int, dtype: torch.dtype):
        self.F = torch.zeros(d, d, dtype=dtype)
        self.z = torch.zeros(d, dtype=dtype)
        # value text -> its (precomputed) unit vector; LRU order = use order.
        self.lexicon: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self.read_rule: Optional[str] = None   # rule of the LAST write


class ParametricMemory:
    """Mem0/Letta-shaped memory runtime backed by fast-weight state."""

    def __init__(
        self,
        d_mem: int = 128,
        vocab_size: int = 32000,
        update_rule: str = "sum",
        decay: float = 1.0,
        eta: float = 0.5,
        seed: int = 0,
        text_encoder: Optional[Callable[[str], torch.Tensor]] = None,
        lexicon_capacity: int = 1024,
        dtype: torch.dtype = torch.float32,
    ):
        if update_rule not in ("sum", "delta", "falcon_nlms"):
            raise ValueError(
                f"update_rule must be 'sum'/'delta'/'falcon_nlms', got {update_rule!r}")
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1], got {decay!r}")
        if not 0.0 < eta <= 1.0:
            raise ValueError(f"eta must be in (0, 1], got {eta!r}")
        self.d_mem = d_mem
        self.vocab_size = vocab_size
        self.update_rule = update_rule
        self.decay = decay
        self.eta = eta
        self.lexicon_capacity = lexicon_capacity
        self.dtype = dtype
        # Fixed random unit-norm table: token id -> key/value vector, and the
        # decode codebook for token-id recall. Deterministic given `seed`, so
        # two processes built with the same seed share one code space.
        g = torch.Generator().manual_seed(seed)
        table = torch.randn(vocab_size, d_mem, generator=g, dtype=torch.float32)
        self._table = torch.nn.functional.normalize(table, dim=-1).to(dtype)
        self._text_encoder = text_encoder  # None -> _hash_vec of d_mem
        self._sessions: Dict[str, _Session] = {}

    # ------------------------------------------------------------------ ids

    def sessions(self) -> List[str]:
        return list(self._sessions)

    def _session(self, session_id: str) -> _Session:
        s = self._sessions.get(session_id)
        if s is None:
            s = _Session(self.d_mem, self.dtype)
            self._sessions[session_id] = s
        return s

    def _embed(self, key) -> torch.Tensor:
        """key/value -> unit vector. int = embedding-table row, str = encoder."""
        if isinstance(key, bool):   # bool is an int subclass; refuse the footgun
            raise TypeError("bool is not a memory key; use int token ids or str")
        if isinstance(key, int):
            if not 0 <= key < self.vocab_size:
                raise ValueError(f"token id {key} outside [0, {self.vocab_size})")
            return self._table[key]
        if isinstance(key, str):
            v = self._text_encoder(key, self.d_mem) if self._text_encoder \
                else _hash_vec(key, self.d_mem)
            v = v.to(self.dtype)
            return v / v.norm().clamp_min(_EPS)
        raise TypeError(f"memory key/value must be int or str, got {type(key)!r}")

    # ---------------------------------------------------------------- write

    def write(self, session_id: str, key, value,
              update_rule: Optional[str] = None) -> None:
        """Bind key -> value into the session's (F, z).

        ``update_rule`` overrides the constructor default for THIS write, so
        one store can A/B the two trained mechanisms (mt_v2 vs mt_v2_delta).
        delta carries z unchanged, mirroring FastWeightMemoryV2's contract."""
        rule = update_rule or self.update_rule
        if rule not in ("sum", "delta", "falcon_nlms"):
            raise ValueError(
                f"update_rule must be 'sum'/'delta'/'falcon_nlms', got {rule!r}")
        k = self._embed(key)
        v = self._embed(value)
        s = self._session(session_id)
        if rule == "sum":
            # Outer-product accumulation with decay: F <- decay*F + k v^T.
            s.F = self.decay * s.F + torch.outer(k, v)
            s.z = self.decay * s.z + k
        elif rule == "delta":
            # Gradient-as-memory: F <- decay*F - eta * k (k^T F - v)^T.
            # k must be unit-norm (it is, by _embed) so the per-step
            # contraction eigenvalue is (decay - eta) in (-1, 1) — bounded.
            pred = k @ s.F                          # (d,) = k^T F
            s.F = self.decay * s.F - self.eta * torch.outer(k, pred - v)
        else:
            # Falcon-1 scalar NLMS (arXiv:2608.27763): F <- decay*F +
            # eta (v - F k) k^T / (||k||^2 + eps), scale-invariant step.
            pred = s.F @ k
            s.F = self.decay * s.F + self.eta * torch.outer(v - pred, k) / (
                float(k.dot(k)) + _EPS
            )
        s.read_rule = rule
        if isinstance(value, str):
            s.lexicon[value] = v
            s.lexicon.move_to_end(value)
            while len(s.lexicon) > self.lexicon_capacity:
                s.lexicon.popitem(last=False)

    # ---------------------------------------------------------------- recall

    def recall(self, session_id: str, query, top_k: int = 1,
               candidates: Optional[Sequence[int]] = None
               ) -> List[Tuple[Optional[MemoryValue], float]]:
        """query -> top_k (value, cosine score) pairs, best first.

        Read path: q -> F read -> nearest-neighbour decode.
          * candidates given: decode against those token-id table rows.
          * else if the session has text values: decode against the lexicon.
          * else: decode against the FULL vocab table (chance = 1/vocab).
        A session with no state (never written, or forgotten to zero) returns
        [(None, 0.0)] — "no memory of it", never a stale guess."""
        s = self._sessions.get(session_id)
        if s is None or s.read_rule is None:
            return [(None, 0.0)] * top_k
        q = self._embed(query)
        qF = q @ s.F
        # Numerically-zero guard, RELATIVE to |F|: forget() zeroes a key's row
        # exactly in real arithmetic, but fp32 rounding leaves |qF| ~ 1e-7
        # that still points mostly at the erased value (the cancelled entries
        # were along it). A genuine binding reads |qF| >= ~1 while |F| grows
        # only as sqrt(#writes), so 1e-5*|F| separates them by ~4 orders.
        if float(qF.norm()) <= 1e-5 * float(s.F.norm().clamp_min(1.0)):
            return [(None, 0.0)] * top_k
        if s.read_rule == "sum":
            r = qF / (q @ s.z).clamp_min(_EPS)
        else:
            r = qF
        r = r / r.norm().clamp_min(_EPS)
        if candidates is not None:
            rows, labels = self._table[list(candidates)], list(candidates)
        elif s.lexicon:
            rows = torch.stack(list(s.lexicon.values()))
            labels = list(s.lexicon.keys())            # type: ignore[arg-type]
        else:
            rows, labels = self._table, None
        scores = rows @ r
        k = min(top_k, scores.numel())
        top = torch.topk(scores, k)
        out: List[Tuple[Optional[MemoryValue], float]] = []
        for i in top.indices.tolist():
            val: Optional[MemoryValue] = labels[i] if labels is not None else i
            out.append((val, float(scores[i].item())))
        while len(out) < top_k:
            out.append((None, 0.0))
        return out

    # ---------------------------------------------------------------- forget

    def forget(self, session_id: str, key=None) -> bool:
        """Erase ONE binding (key given) or the WHOLE session (key=None).

        Per-key removal is a delta projection with eta=1: F <- F - k (k^T F),
        which removes the key's ENTIRE association (all its writes, blended)
        while touching other keys only through their ~1/sqrt(d) cross-talk
        with k. No binding index exists — this is algebra on the (d, d) state,
        so state size stays O(1) in writes. z is decremented once; after
        multiple writes to the same key a small residual remains, which only
        scales other keys' denominators by ~(1 + q.k) ~ 1 (harmless, tested).
        Forgetting an absent key subtracts ~cross-talk noise — a no-op."""
        s = self._sessions.get(session_id)
        if s is None:
            return False
        if key is None:
            s.F.zero_()
            s.z.zero_()
            s.lexicon.clear()
            s.read_rule = None
            return True
        k = self._embed(key)
        s.F = s.F - torch.outer(k, k @ s.F)
        s.z = s.z - k
        return True

    # -------------------------------------------------- snapshot / persist

    def snapshot(self, session_id: str) -> Optional[dict]:
        """(F, z) + lexicon as a JSON-safe dict (base64 tensors). Bit-exact."""
        s = self._sessions.get(session_id)
        if s is None:
            return None
        return {
            "format": _SNAP_FORMAT,
            "d_mem": self.d_mem,
            "read_rule": s.read_rule,
            "F": _t2b64(s.F), "z": _t2b64(s.z),
            "lexicon": [[val, _t2b64(vec)] for val, vec in s.lexicon.items()],
        }

    def restore(self, session_id: str, snap: dict) -> None:
        """Inverse of snapshot (bit-exact by construction: raw fp32 bytes)."""
        if snap.get("format") != _SNAP_FORMAT:
            raise ValueError(f"unknown snapshot format {snap.get('format')!r}")
        s = self._session(session_id)
        s.F = _b642t(snap["F"], (self.d_mem, self.d_mem), self.dtype)
        s.z = _b642t(snap["z"], (self.d_mem,), self.dtype)
        s.read_rule = snap["read_rule"]
        s.lexicon.clear()
        for val, vec_b64 in snap["lexicon"]:
            s.lexicon[val] = _b642t(vec_b64, (self.d_mem,), self.dtype)

    def save(self, session_id: str, path: str) -> None:
        """Persist one session through session_state.py's atomic JSON writer
        (HFSessionState.fw_state rides the Capsule v2 envelope unchanged)."""
        snap = self.snapshot(session_id)
        if snap is None:
            raise KeyError(f"no session {session_id!r} to save")
        _save_hf_session(HFSessionState(session_id=session_id, fw_state=snap), path)

    def load(self, session_id: str, path: str) -> None:
        state = _load_hf_session(path)
        if state.fw_state is None:
            raise ValueError(f"{path} carries no fw_state payload")
        self.restore(session_id, state.fw_state)

    # ----------------------------------------------------------------- size

    def state_bytes(self, session_id: str) -> int:
        """Bytes of the session's (F, z) tensors — CONSTANT in writes (the
        O(1) claim). The bounded lexicon is intentionally excluded: it is a
        fixed-capacity decode aid, not the memory body; see module docstring."""
        s = self._sessions.get(session_id)
        if s is None:
            return 0
        el = self.d_mem * self.d_mem + self.d_mem
        return el * torch.tensor([], dtype=self.dtype).element_size()


def _t2b64(t: torch.Tensor) -> str:
    return base64.b64encode(t.detach().cpu().numpy().tobytes()).decode("ascii")


def _b642t(b: str, shape, dtype: torch.dtype) -> torch.Tensor:
    import numpy as np
    t = torch.from_numpy(np.frombuffer(base64.b64decode(b), dtype=np.float32).copy())
    return t.reshape(shape).to(dtype)
