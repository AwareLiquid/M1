"""PMB reference systems — unified ingest / snapshot / restore / answer protocol.

Protocol (PMB_SPEC.md §3). Sessions MUST pass through disk between ingests:
the runner (run_pmb.py) snapshots after every session, writes the bytes to a
file, tears the system instance down, and rebuilds a FRESH instance restored
from those bytes before the next session. A system whose state does not
survive that round-trip scores as if it had no memory — bit-exact persistence
is the entry ticket.

Evaluation modes (declared via the ``mode`` class attribute, scored by the
harness with anti-enumeration truncation — PMB_SPEC.md §7):

* ``mode = "evidence"`` (default): ``answer_context(question)`` returns the
  context the system assembles; the harness truncates it to the context char
  budget (default 4000) and substring-matches the gold 6-digit value.
* ``mode = "generative"``: the harness calls ``answer(question)`` instead and
  truncates the generated text to 200 chars before matching. Fastweight is
  generative — v0 ships the interface skeleton wired to the real mt_lnn APIs
  but requires E2's GPU-tuned adapter artifacts to actually run.

Reference systems:

* ``NoneSystem``      — stores nothing (lower bound).
* ``OracleContextSystem`` — full history in context (upper bound while it
  fits; ``max_tokens`` overflow drops the OLDEST sessions and sets
  ``truncated`` — that failure IS a data point).
* ``RagSystem``       — sentence-level chunks + vector top-k. Encoders are
  pluggable: ``hash`` (pure-numpy hashed bag-of-words, fully offline) or
  ``e5`` (mt_lnn.sentence_encoder.SentenceEncoder, multilingual-e5-small).
* ``FastWeightSystem`` — the project's own contestant: MT-LNN adapter (F, z)
  snapshots + FastWeightSessionStore. Skeleton in v0 (needs a GPU-tuned
  adapter model).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import struct
import tempfile
from typing import Callable, List, Optional, Tuple

import numpy as np

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[a-z']+")


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def _count_tokens(text: str) -> int:
    """Whitespace-word proxy for tokens (v0; consistent across systems)."""
    return len(text.split())


def _repo_root_on_path() -> None:
    """Make ``import mt_lnn`` work when run_pmb.py is launched by file path
    (sys.path[0] = this directory, not the repo root)."""
    try:
        import mt_lnn  # noqa: F401
        return
    except ImportError:
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class MemorySystem:
    """Unified PMB contestant protocol.

    ``ingest``/``snapshot``/``answer`` are the spec's three methods; v0 adds
    ``restore`` so the runner can rebuild a fresh instance from the on-disk
    snapshot (the spec's "恢复状态后作答" step made explicit), and
    ``answer_context`` as the evidence-recall evaluation surface.

    ``mode`` declares the scoring path (PMB_SPEC.md §7):

    * ``"evidence"`` (default) — the harness scores ``answer_context()``,
      truncated to the context char budget.
    * ``"generative"`` — for weight/state-memory systems that cannot emit
      retrieved text: the harness scores ``answer()`` instead, truncated to
      200 chars.
    """

    name = "base"
    mode = "evidence"

    def ingest(self, session_id: str, text: str) -> None:
        raise NotImplementedError

    def snapshot(self, session_id: str) -> bytes:
        """Serialize ALL persistent state to bytes (written to disk by the
        runner). Must be a pure function of ingested history."""
        raise NotImplementedError

    def restore(self, blob: bytes) -> None:
        """Rebuild state from a prior ``snapshot`` blob on a FRESH instance."""
        raise NotImplementedError

    def answer_context(self, question: str) -> str:
        """Return the context assembled for *question* (evidence-recall mode:
        the harness substring-matches the gold value against this)."""
        raise NotImplementedError

    def answer(self, question: str) -> str:
        """Spec-name alias. Context systems answer WITH their context;
        generative systems override this with real generation."""
        return self.answer_context(question)


# ---------------------------------------------------------------------------
# none — lower bound
# ---------------------------------------------------------------------------

class NoneSystem(MemorySystem):
    """Remembers nothing. The floor every real system must beat."""

    name = "none"

    def ingest(self, session_id: str, text: str) -> None:
        pass

    def snapshot(self, session_id: str) -> bytes:
        return b""

    def restore(self, blob: bytes) -> None:
        pass

    def answer_context(self, question: str) -> str:
        return ""


# ---------------------------------------------------------------------------
# oracle-context — upper bound while the window holds
# ---------------------------------------------------------------------------

class OracleContextSystem(MemorySystem):
    """Concatenates the FULL history into the answer context.

    ``max_tokens`` models a finite context window: when the history exceeds
    it, whole OLDEST sessions are dropped (newest kept intact) and
    ``self.truncated`` flips to True. Overflow failure is a reported data
    point, not an error.
    """

    name = "oracle"

    def __init__(self, max_tokens: int = 8000):
        self.max_tokens = int(max_tokens)
        self._history: List[Tuple[str, str]] = []   # (session_id, text)
        self.truncated = False

    def ingest(self, session_id: str, text: str) -> None:
        self._history.append((session_id, text))

    def snapshot(self, session_id: str) -> bytes:
        return json.dumps(
            {"format": "pmb-oracle-v0", "max_tokens": self.max_tokens,
             "history": self._history},
            ensure_ascii=False, sort_keys=True,
        ).encode("utf-8")

    def restore(self, blob: bytes) -> None:
        obj = json.loads(blob.decode("utf-8"))
        if obj.get("format") != "pmb-oracle-v0":
            raise ValueError(f"not an oracle snapshot: {obj.get('format')!r}")
        self._history = [(sid, text) for sid, text in obj["history"]]
        self.max_tokens = int(obj.get("max_tokens", self.max_tokens))

    def answer_context(self, question: str) -> str:
        kept: List[str] = []
        budget = self.max_tokens
        for _, text in reversed(self._history):      # newest first
            cost = _count_tokens(text)
            if cost > budget:
                self.truncated = True
                break
            kept.append(text)
            budget -= cost
        kept.reverse()                               # back to oldest->newest
        return "\n".join(kept)


# ---------------------------------------------------------------------------
# rag — sentence chunks + vector top-k (pluggable encoder)
# ---------------------------------------------------------------------------

class HashingEncoder:
    """Deterministic hashed bag-of-words sentence encoder (pure numpy).

    Each lowercase word token is hashed (md5 — process-stable, unlike
    ``hash()``) into one of *dim* signed buckets; the sum is L2-normalized.
    Fully offline, no model download — the CI-friendly default.
    """

    name = "hash"

    def __init__(self, dim: int = 256):
        self.dim = int(dim)

    def encode(self, text: str, is_query: bool = False) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in _TOKEN_RE.findall(text.lower()):
            d = hashlib.md5(tok.encode("utf-8")).digest()
            idx = int.from_bytes(d[:4], "big") % self.dim
            sign = 1.0 if d[4] & 1 else -1.0
            v[idx] += sign
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v


class E5Encoder:
    """mt_lnn SentenceEncoder wrapper (multilingual-e5-small by default).

    Uses the real asymmetric recipe: ``is_query=True`` -> "query: " prefix,
    ``is_query=False`` -> "passage: " prefix (see mt_lnn/sentence_encoder.py).
    Requires transformers + a (cached) HF download.
    """

    name = "e5"

    def __init__(self, model_id: Optional[str] = None, device: str = "cpu"):
        _repo_root_on_path()
        from mt_lnn.sentence_encoder import SentenceEncoder, DEFAULT_MODEL
        self._enc = SentenceEncoder(model_id=model_id or DEFAULT_MODEL,
                                    device=device)
        self.dim = None  # discovered on first encode (lazy model load)

    def encode(self, text: str, is_query: bool = False) -> np.ndarray:
        vec = self._enc.encode(text, is_query=is_query)   # 1-D L2-normed torch
        out = vec.numpy().astype(np.float32)
        if self.dim is None:
            self.dim = int(out.shape[0])
        return out


def make_encoder(kind: str, dim: int = 256, device: str = "cpu"):
    if kind == "hash":
        return HashingEncoder(dim=dim)
    if kind == "e5":
        return E5Encoder(device=device)
    raise ValueError(f"unknown encoder {kind!r} (expected 'hash' or 'e5')")


class RagSystem(MemorySystem):
    """Sentence-level chunking + cosine top-k retrieval (mini mem0/Zep proxy).

    The snapshot stores BOTH the sentences and their float32 vectors, so
    ``state_bytes`` honestly reflects what a vector store persists, and
    restore needs no re-embedding (bit-exact for e5 too).
    """

    name = "rag"

    def __init__(self, encoder: str = "hash", topk: int = 8, dim: int = 256,
                 device: str = "cpu"):
        self.encoder_name = encoder
        self.topk = int(topk)
        self._enc = make_encoder(encoder, dim=dim, device=device)
        self._sentences: List[str] = []
        self._vecs: List[np.ndarray] = []

    def ingest(self, session_id: str, text: str) -> None:
        for sent in _split_sentences(text):
            self._sentences.append(sent)
            self._vecs.append(self._enc.encode(sent, is_query=False))

    # -- persistence: [8-byte header length][JSON header][raw float32 matrix]
    def snapshot(self, session_id: str) -> bytes:
        dim = int(self._vecs[0].shape[0]) if self._vecs else 0
        header = json.dumps(
            {"format": "pmb-rag-v0", "encoder": self.encoder_name,
             "dim": dim, "sentences": self._sentences},
            ensure_ascii=False, sort_keys=True,
        ).encode("utf-8")
        mat = (np.stack(self._vecs).astype(np.float32)
               if self._vecs else np.zeros((0, 0), dtype=np.float32))
        return struct.pack(">Q", len(header)) + header + mat.tobytes()

    def restore(self, blob: bytes) -> None:
        (hlen,) = struct.unpack(">Q", blob[:8])
        obj = json.loads(blob[8:8 + hlen].decode("utf-8"))
        if obj.get("format") != "pmb-rag-v0":
            raise ValueError(f"not a rag snapshot: {obj.get('format')!r}")
        if obj.get("encoder") != self.encoder_name:
            raise ValueError(
                f"snapshot was built with encoder {obj.get('encoder')!r}, "
                f"this instance uses {self.encoder_name!r}")
        self._sentences = list(obj["sentences"])
        dim = int(obj["dim"])
        raw = np.frombuffer(blob[8 + hlen:], dtype=np.float32)
        if self._sentences:
            mat = raw.reshape(len(self._sentences), dim)
            self._vecs = [mat[i].copy() for i in range(mat.shape[0])]
        else:
            self._vecs = []

    def answer_context(self, question: str) -> str:
        if not self._vecs:
            return ""
        q = self._enc.encode(question, is_query=True)
        mat = np.stack(self._vecs)
        scores = mat @ q
        k = min(self.topk, len(self._sentences))
        top = np.argsort(-scores)[:k]
        return " ".join(self._sentences[int(i)] for i in top)


# ---------------------------------------------------------------------------
# fastweight — MT-LNN (F, z) session snapshots (v0: interface skeleton)
# ---------------------------------------------------------------------------

class FastWeightSystem(MemorySystem):
    """The project's own contestant: adapter fast-weight (F, z) snapshots
    persisted through :class:`mt_lnn.fast_weight_store.FastWeightSessionStore`.

    GENERATIVE-mode participant (``mode = "generative"``): it answers by
    restoring the recalled (F, z) into the model and generating, not by
    returning retrieved text — the harness scores its ``answer()`` output
    (truncated to 200 chars, PMB_SPEC.md §7), never ``answer_context()``.

    Protocol compliance (v0.1): there is NO shared-CWD side channel. The
    session store's sqlite db lives in an instance-owned temp directory, and
    ``snapshot()`` packs BOTH the adapter (F, z) streams and the FULL store db
    file bytes into one container — every byte of persistent state must ride
    through the snapshot/restore disk round-trip, and ``state_bytes`` is the
    full package length.

    v0 status (spec §5, honest): requires a GPU-fine-tuned MT-LNN adapter
    model (the E2 deliverable). Constructing without one raises immediately.
    The wiring below targets the REAL mt_lnn APIs (verified against source):

    * ``mt_lnn.llama_adapter.set_adapter_streaming(model, enabled) -> int``
    * ``mt_lnn.llama_adapter.snapshot_adapter_streams(model) -> dict``
    * ``mt_lnn.llama_adapter.restore_adapter_streams(model, snap, batch=None) -> int``
    * ``mt_lnn.fast_weight_store.FastWeightSessionStore(db_path, key_dim, max_entries)``
      with ``write_session(session_id, key_vec, fw_snapshot, meta, surprise)``
      and ``recall_session(query_vec, top_k, center, score_floor,
      expected_session_id)`` returning ``[(fw_snapshot, score, session_id,
      meta), ...]``
    * ``mt_lnn.fast_weight_store.build_session_key(text, encode_fn, max_chars)``
    * ``mt_lnn.sentence_encoder.SentenceEncoder(...).as_fn(is_query=...)``
    """

    name = "fastweight"
    mode = "generative"

    _SNAP_FORMAT = "pmb-fastweight-v1"

    def __init__(self,
                 model=None,
                 tokenizer=None,
                 encoder_model_id: Optional[str] = None,
                 generate_fn: Optional[Callable[[str], str]] = None,
                 device: str = "cpu"):
        if model is None or tokenizer is None:
            raise RuntimeError(
                "FastWeightSystem needs a GPU-fine-tuned MT-LNN adapter model "
                "(the E2 training deliverable): pass --model <checkpoint> so a "
                "model+tokenizer with MTResidualAdapter/MTRecurrentMixer "
                "modules can be loaded. v0 ships this contestant as an "
                "interface skeleton only — none/oracle/rag run CPU-only "
                "without it (PMB_SPEC.md §5)."
            )
        _repo_root_on_path()
        import torch
        from mt_lnn.llama_adapter import set_adapter_streaming
        from mt_lnn.fast_weight_store import FastWeightSessionStore
        from mt_lnn.sentence_encoder import SentenceEncoder, DEFAULT_MODEL

        self._torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.generate_fn = generate_fn
        self._sent_enc = SentenceEncoder(
            model_id=encoder_model_id or DEFAULT_MODEL, device=device)
        # Instance-owned temp dir: NO shared-CWD side channel — two instances
        # in the same process/cwd can never see each other's store, and the
        # only way state crosses sessions is the snapshot()/restore() bytes.
        self._tmpdir = tempfile.mkdtemp(prefix="pmb_fastweight_")
        self._db_path = os.path.join(self._tmpdir, "store.db")
        # key_dim must follow the PLUGGABLE encoder (e5-small is 384-d but
        # e.g. multilingual-e5-base is 768-d) — a hardcoded default crashes
        # the first write_session with a key-dim mismatch.
        self._key_dim = int(self._sent_enc.dim())
        self.store = FastWeightSessionStore(db_path=self._db_path,
                                            key_dim=self._key_dim)
        # Adapter streaming records (F, z) at INFERENCE time only
        # (llama_adapter gates on `not self.training`) — a model arriving
        # straight from a training loop would silently record NOTHING and
        # publish zero-memory numbers with no error anywhere. Force eval.
        self.model.eval()
        set_adapter_streaming(self.model, True)   # also clears stale state

    def _reopen_store(self):
        from mt_lnn.fast_weight_store import FastWeightSessionStore
        self.store = FastWeightSessionStore(db_path=self._db_path,
                                            key_dim=self._key_dim)

    def ingest(self, session_id: str, text: str) -> None:
        from mt_lnn.llama_adapter import snapshot_adapter_streams
        from mt_lnn.fast_weight_store import build_session_key

        enc = self.tokenizer(text, return_tensors="pt",
                             truncation=True).to(self.device)
        with self._torch.no_grad():
            self.model(**enc)                     # writes the (F, z) streams
        fw = snapshot_adapter_streams(self.model)
        key = build_session_key(text, self._sent_enc.as_fn(is_query=False))
        self.store.write_session(session_id, key, fw,
                                 meta={"benchmark": "pmb-v0"})

    # -- persistence container: [">QQ" length header][adapter-streams blob]
    #    [store db file bytes]. ALL persistent state — the live (F, z) streams
    #    AND the session store's sqlite db — must survive as these bytes;
    #    state_bytes therefore reflects the FULL persisted footprint.
    def snapshot(self, session_id: str) -> bytes:
        from mt_lnn.llama_adapter import snapshot_adapter_streams

        buf = io.BytesIO()
        self._torch.save(
            {"format": self._SNAP_FORMAT, "session_id": session_id,
             "fw": snapshot_adapter_streams(self.model)},
            buf,
        )
        stream_blob = buf.getvalue()
        # Close before reading so sqlite flushes everything to the file, then
        # reopen so the live instance stays usable after snapshotting.
        self.store.close()
        with open(self._db_path, "rb") as f:
            db_bytes = f.read()
        self._reopen_store()
        return (struct.pack(">QQ", len(stream_blob), len(db_bytes))
                + stream_blob + db_bytes)

    def restore(self, blob: bytes) -> None:
        from mt_lnn.llama_adapter import restore_adapter_streams

        if len(blob) < 16:
            raise ValueError("fastweight snapshot container too short")
        stream_len, db_len = struct.unpack(">QQ", blob[:16])
        if 16 + stream_len + db_len != len(blob):
            raise ValueError("corrupt fastweight snapshot container")
        stream_blob = blob[16:16 + stream_len]
        db_bytes = blob[16 + stream_len:]

        obj = self._torch.load(io.BytesIO(stream_blob), map_location="cpu",
                               weights_only=False)
        if obj.get("format") != self._SNAP_FORMAT:
            raise ValueError(
                f"not a fastweight snapshot: {obj.get('format')!r}")
        # Rehydrate the session store from the packed db bytes in THIS
        # instance's own temp dir (fresh instance starts empty by design).
        self.store.close()
        with open(self._db_path, "wb") as f:
            f.write(db_bytes)
        self._reopen_store()
        restore_adapter_streams(self.model, obj["fw"], batch=1)

    def close(self) -> None:
        """Release the sqlite connection and the instance temp dir."""
        store = getattr(self, "store", None)
        if store is not None:
            try:
                store.close()
            except Exception:
                pass
            self.store = None
        tmpdir = getattr(self, "_tmpdir", None)
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
            self._tmpdir = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def answer_context(self, question: str) -> str:
        raise NotImplementedError(
            "FastWeightSystem is a generative-mode contestant: state lives in "
            "the (F, z) fast weights, not in retrievable text — use answer().")

    def answer(self, question: str) -> str:
        from mt_lnn.llama_adapter import restore_adapter_streams

        q_vec = self._sent_enc.encode(question, is_query=True)
        hits = self.store.recall_session(q_vec, top_k=1, center=True)
        if hits:
            fw_snapshot, _score, _sid, _meta = hits[0]
            restore_adapter_streams(self.model, fw_snapshot, batch=1)
        if self.generate_fn is None:
            raise NotImplementedError(
                "v0 skeleton: supply generate_fn (model.generate wrapper) "
                "once the E2 GPU-tuned adapter checkpoint exists.")
        return self.generate_fn(question)


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

def make_system(name: str, *, encoder: str = "hash", topk: int = 8,
                dim: int = 256, max_tokens: int = 8000,
                device: str = "cpu", **fastweight_kwargs) -> MemorySystem:
    """Build a fresh system instance (the runner calls this per session to
    enforce the disk round-trip — no in-process state may survive)."""
    if name == "none":
        return NoneSystem()
    if name == "oracle":
        return OracleContextSystem(max_tokens=max_tokens)
    if name == "rag":
        return RagSystem(encoder=encoder, topk=topk, dim=dim, device=device)
    if name == "fastweight":
        return FastWeightSystem(device=device, **fastweight_kwargs)
    raise ValueError(
        f"unknown system {name!r} (expected none|oracle|rag|fastweight)")
