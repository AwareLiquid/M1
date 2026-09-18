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
_TOKEN_RE = re.compile(r"[a-z0-9']+")  # 含数字: 6 位码 gold 必须可被编码（否则 generative 系统全部判负为伪影）


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
                 device: str = "cpu", score_threshold: Optional[float] = None):
        self.encoder_name = encoder
        self.topk = int(topk)
        # t5b abstention: if the best retrieval score is below this floor,
        # answer_context returns "" (the system abstains). None = legacy
        # behaviour (always return top-k text). Opt-in; default unchanged.
        self.score_threshold = score_threshold
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
        if (self.score_threshold is not None
                and float(scores[top[0]]) < self.score_threshold):
            return ""   # abstain: best match below the confidence floor (t5b)
        return " ".join(self._sentences[int(i)] for i in top)


class RagTwoHopSystem(RagSystem):
    """Two-hop retrieval baseline (B-16) — protocol-ceiling reference for the
    cross-session JOIN task, not a general system (oracle-like role).

    t4 questions ask for person→code while quoting person→rating. Single-hop
    rag retrieves one side of the join and the answer sentence never lands in
    the context. Here hop 1 resolves the quoted rating to a person (parse the
    rating sentence out of the question's own top-3), hop 2 re-queries by the
    person's name and returns those sentences. Same store, same encoder,
    same top-k discipline — only the retrieval PROTOCOL differs, which is
    exactly the variable under test: if this reaches retrieval parity where
    single-hop hash scores 0.0, the JOIN failure is a protocol gap (same
    shape as the t5b abstention finding), not an information gap."""

    name = "rag_2hop"
    # question template: "...satisfaction rating is {r}?" (no "out of 5");
    # stored sentence template: "{name} left a satisfaction rating of {r} out of 5."
    _RATING_Q_RE = re.compile(r"satisfaction rating is (\d+)")
    _RATING_SENT_RE = re.compile(
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?) left a satisfaction rating of (\d+)")

    def answer_context(self, question: str) -> str:
        if not self._vecs:
            return ""
        m = self._RATING_Q_RE.search(question)
        if m is None:
            return super().answer_context(question)   # non-JOIN: single-hop
        person = self._person_for_rating(question, m.group(1))
        if person is None:
            return super().answer_context(question)   # hop-1 miss: degrade
        q2 = self._enc.encode(f"{person} membership number is", is_query=True)
        mat = np.stack(self._vecs)
        scores = mat @ q2
        top = np.argsort(-scores)[:min(2, len(self._sentences))]
        return " ".join(self._sentences[int(i)] for i in top)

    def _person_for_rating(self, question: str, rating: str) -> Optional[str]:
        """Hop 1 — resolve rating -> person from the question's own top-3
        sentences (embedding retrieval + mechanical parse of the rating
        sentence). None = hop-1 miss."""
        q = self._enc.encode(question, is_query=True)
        mat = np.stack(self._vecs)
        scores = mat @ q
        top = np.argsort(-scores)[:min(3, len(self._sentences))]
        for i in top:
            m = self._RATING_SENT_RE.search(self._sentences[int(i)])
            if m and m.group(2) == rating:
                return m.group(1)
        return None


# ---------------------------------------------------------------------------
# filesystem — Letta-style trivial baseline (B-23, community arbitration)
# ---------------------------------------------------------------------------

class FilesystemSystem(MemorySystem):
    """B-23: dump every session verbatim into one plain-text store; at
    answer time grep lines by question-word overlap and return the hits
    (the harness truncates at its budget). No embeddings, no index, no
    vectors — state_bytes is the raw file, O(history) by design. This is
    the PMB-isation of Letta's 'filesystem beats memory layers' claim:
    if this trivial baseline matches rag anywhere, that task family does
    not need a retrieval layer."""

    name = "filesystem"
    mode = "evidence"
    _SNAP_FORMAT = "pmb-filesystem-v0"
    _SID = "pmb"

    def __init__(self, **_):
        self.lines: List[str] = []

    def ingest(self, session_id: str, text: str) -> None:
        self.lines.extend(_split_sentences(text))

    def snapshot(self, session_id: str) -> bytes:
        obj = {"format": self._SNAP_FORMAT, "lines": self.lines}
        return json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")

    def restore(self, blob: bytes) -> None:
        obj = json.loads(blob.decode("utf-8"))
        if obj.get("format") != self._SNAP_FORMAT:
            raise ValueError(f"not a filesystem snapshot: {obj.get('format')!r}")
        self.lines = list(obj["lines"])

    def answer_context(self, question: str) -> str:
        if not self.lines:
            return ""
        qwords = set(w for w in re.split(r"[^a-z0-9]+", question.lower()) if w)
        scored = []
        for i, line in enumerate(self.lines):
            overlap = len(qwords & set(
                w for w in re.split(r"[^a-z0-9]+", line.lower()) if w))
            if overlap:
                scored.append((overlap, i))
        if not scored:
            return ""
        scored.sort(reverse=True)
        return " ".join(self.lines[i] for _, i in scored)


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
        if self.generate_fn is None:
            def _default_generate(prompt: str) -> str:
                enc = self.tokenizer(prompt, return_tensors="pt",
                                     truncation=True,
                                     max_length=1024).to(self.device)
                with self._torch.no_grad():
                    out = self.model.generate(
                        **enc, max_new_tokens=64, do_sample=False,
                        pad_token_id=(self.tokenizer.pad_token_id or
                                      self.tokenizer.eos_token_id))
                return self.tokenizer.decode(out[0][enc["input_ids"].shape[1]:],
                                             skip_special_tokens=True)
            self.generate_fn = _default_generate
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
        self._key_dim = int(self._sent_enc.dim)
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
# parametric — fast-weight (F, z) memory with selectable write rule
# ---------------------------------------------------------------------------

_NON_NAME_CAPS = frozenset({
    "The", "What", "Which", "For", "Please", "According", "Note", "Update",
    "Correction", "Effective",
})


class ParametricMemorySystem(MemorySystem):
    """Fast-weight (F, z) memory backed by ``mt_lnn.parametric_memory`` with a
    selectable write rule (``sum`` / ``delta`` / ``falcon_nlms``). Generative
    mode: parses fact sentences into ``key -> code`` bindings written into the
    (F, z) state; answers by recalling the bound code through the state."""

    name = "parametric"
    mode = "generative"
    _SNAP_FORMAT = "pmb-parametric-v1"
    _SID = "pmb"

    def __init__(self, d_mem: int = 256, update_rule: str = "falcon_nlms",
                 decay: float = 0.99, eta: float = 0.5, seed: int = 0,
                 consolidate_mode: str = "none",
                 consolidate_param: Optional[float] = None,
                 read_iters: int = 1, read_beta: float = 2.0,
                 read_protocol: str = "direct"):
        from mt_lnn.parametric_memory import ParametricMemory
        self.memory = ParametricMemory(d_mem=d_mem, update_rule=update_rule,
                                       decay=decay, eta=eta, seed=seed,
                                       consolidate_mode=consolidate_mode,
                                       consolidate_param=consolidate_param,
                                       read_iters=read_iters,
                                       read_beta=read_beta,
                                       read_protocol=read_protocol)
        self.update_rule = update_rule
        self.d_mem = d_mem

    @staticmethod
    def _attr_in(sentence: str) -> Optional[str]:
        try:
            from .tasks import ATTRIBUTES
        except ImportError:
            from tasks import ATTRIBUTES
        for a in ATTRIBUTES:
            if a in sentence:
                return a
        return None

    @staticmethod
    def _key_of(sentence: str) -> Optional[str]:
        """Extract ``name attr`` from a fact/question sentence (None if not
        fact-shaped; no 6-digit code / no known attribute)."""
        attr = ParametricMemorySystem._attr_in(sentence)
        if attr is None:
            return None
        rest = sentence.replace(attr, " ")
        rest = re.sub(r"\b\d{6}\b", " ", rest)
        caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", rest)
        caps = [c for c in caps if c.split()[0] not in _NON_NAME_CAPS]
        if not caps:
            return None
        return f"{caps[-1].strip()} {attr}"

    def ingest(self, session_id: str, text: str) -> None:
        # KDN process noise: every session boundary re-inflates the
        # predictive covariance (no-op for non-kalman_delta rules).
        self.memory.bump_uncertainty(self._SID)
        for sent in _split_sentences(text):
            code = re.search(r"\b\d{6}\b", sent)
            key = self._key_of(sent)
            if code and key:
                self.memory.write(self._SID, key, code.group(0))

    def snapshot(self, session_id: str) -> bytes:
        # B-20: the session boundary IS the sleep window — consolidate before
        # persisting (no-op for consolidate_mode='none').
        self.memory.consolidate(self._SID)
        snap = self.memory.snapshot(self._SID)
        if snap is None:
            return b""
        return json.dumps(snap, ensure_ascii=False).encode("utf-8")

    def restore(self, blob: bytes) -> None:
        if not blob:
            return
        self.memory.restore(self._SID, json.loads(blob.decode("utf-8")))

    def answer(self, question: str) -> str:
        key = self._key_of(question)
        if key is None:
            return ""
        hits = self.memory.recall(self._SID, key, top_k=1)
        if not hits or hits[0][0] is None:
            return ""
        return str(hits[0][0])


# ---------------------------------------------------------------------------
# micro fastweight — gradient-trained keyed associative memory (B-4' v0)
# ---------------------------------------------------------------------------

class MicroFastWeightSystem(MemorySystem):
    """Gradient-trained keyed associative memory — the classic fast-weight
    rule (SGD on a linear associator) as the contrast to ParametricMemory's
    closed-form sum/delta writes. Scientific point (B-7 follow-up): does
    *gradient* writing survive cross-session binding any better than the
    closed-form rules that scored 0.0 on the T1 hard cell?  Catastrophic
    interference is the expected failure mode; this system exists to
    measure it, not to hide it (L008: local CPU, ~30 min for 3 seeds)."""

    name = "micro_fw"
    mode = "generative"
    _SNAP_FORMAT = "pmb-microfw-v1"
    _SID = "pmb"

    def __init__(self, d_mem: int = 104, lr: float = 0.1, inner_steps: int = 8,
                 seed: int = 0, **_):
        import torch
        self._torch = torch
        self.d_mem = d_mem
        self.lr = lr
        self.inner_steps = inner_steps
        self.seed = seed
        g = self._torch.Generator().manual_seed(seed)
        # fixed random projections (part of state: must survive snapshot)
        self.P_key = self._torch.randn(d_mem, d_mem, generator=g) * (d_mem ** -0.5)
        self.P_val = self._torch.randn(d_mem, d_mem, generator=g) * (d_mem ** -0.5)
        # the fast weight itself
        self.W = self._torch.zeros(d_mem, d_mem)
        # value space: code string -> index into value prototype matrix
        self._codes: List[str] = []
        self._val_mat: Optional[self._torch.Tensor] = None
        # hashing encoder for text -> d_mem vector (deterministic, offline)
        self._enc = make_encoder("hash", dim=d_mem)

    def _phi_key(self, text: str):
        v = self._enc.encode(text, is_query=True)
        return self.P_key @ self._torch.tensor(v, dtype=self._torch.float32)

    def _phi_val(self, code: str):
        v = self._enc.encode(code, is_query=False)
        return self.P_val @ self._torch.tensor(v, dtype=self._torch.float32)

    def _ensure_code(self, code: str):
        if code not in self._codes:
            self._codes.append(code)
            v = self._phi_val(code).unsqueeze(0)
            self._val_mat = v if self._val_mat is None else self._torch.cat(
                [self._val_mat, v], dim=0)

    def _train_pair(self, key: str, code: str):
        """inner_steps SGD on (phi_key(key) -> phi_val(code)) — the gradient
        write. Each fact is a few steps; interference across facts is the
        measured phenomenon, not a bug."""
        torch = self._torch
        k = self._phi_key(key).detach()
        self._ensure_code(code)
        idx = self._codes.index(code)
        target = self._val_mat[idx].detach()
        W = self.W.detach().clone().requires_grad_(True)
        opt = self._torch.optim.SGD([W], lr=self.lr)
        for _ in range(self.inner_steps):
            opt.zero_grad()
            loss = ((W @ k - target) ** 2).sum()
            loss.backward()
            opt.step()
        self.W = W.detach()

    def ingest(self, session_id: str, text: str) -> None:
        for sent in _split_sentences(text):
            code = re.search(r"\b\d{6}\b", sent)
            key = ParametricMemorySystem._key_of(sent)
            if code and key:
                self._train_pair(key, code.group(0))

    def snapshot(self, session_id: str) -> bytes:
        obj = {"format": self._SNAP_FORMAT,
               "d": self.d_mem, "lr": self.lr,
               "inner_steps": self.inner_steps, "seed": self.seed,
               "W": [[round(x, 6) for x in row] for row in self.W.tolist()],
               "codes": self._codes,
               "vals": [[round(x, 6) for x in row]
                        for row in (self._val_mat.tolist()
                                    if self._val_mat is not None else [])]}
        return json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")

    def restore(self, blob: bytes) -> None:
        obj = json.loads(blob.decode("utf-8"))
        if obj.get("format") != self._SNAP_FORMAT:
            raise ValueError(f"not a micro_fw snapshot: {obj.get('format')!r}")
        torch = self._torch
        self.d_mem = obj["d"]; self.lr = obj["lr"]
        self.inner_steps = obj["inner_steps"]; self.seed = obj["seed"]
        self.W = self._torch.tensor(obj["W"], dtype=self._torch.float32)
        self._codes = list(obj["codes"])
        self._val_mat = (self._torch.tensor(obj["vals"], dtype=self._torch.float32)
                         if obj["vals"] else None)

    def answer(self, question: str) -> str:
        key = ParametricMemorySystem._key_of(question)
        if key is None or self._val_mat is None or not len(self._codes):
            return ""
        h = self.W @ self._phi_key(key)
        sims = self._val_mat @ h
        return self._codes[int(self._torch.argmax(sims))]


class MicroRLSSystem(MicroFastWeightSystem):
    """B-13: closed-form ridge write — replaces 8-step SGD with an exact
    batch least-squares re-solve over ALL stored pairs (Kohonen/Anderson
    pseudo-inverse lineage). For a linear associator of width d, this
    stores ~d linearly-independent key->value patterns EXACTLY with zero
    sequential interference: 'solve, don't grind'. Capacity ≈ d is the
    honest wall — beyond it, graceful degradation is measured, not hidden.
    State is O(n*d) (the pair list) — same asymptotics as rag's store,
    so the comparison with rag is storage-fair."""

    name = "micro_rls"
    _SNAP_FORMAT = "pmb-microrls-v1"

    def __init__(self, d_mem: int = 104, lam: float = 0.1, seed: int = 0, **_):
        super().__init__(d_mem=d_mem, seed=seed, **_)
        self.lam = lam
        self.pairs: List[Tuple[str, str]] = []

    def _resolve(self):
        torch = self._torch
        if not self.pairs:
            self.W = torch.zeros(self.d_mem, self.d_mem)
            return
        K = torch.stack([self._phi_key(k) for k, _ in self.pairs])
        Vm = torch.stack([self._phi_val(c) for _, c in self.pairs])
        A = K.T @ K + self.lam * torch.eye(self.d_mem)
        self.W = torch.linalg.solve(A, K.T @ Vm)
        self._codes, self._val_mat = [], None
        for _, c in self.pairs:
            self._ensure_code(c)

    def ingest(self, session_id: str, text: str) -> None:
        new = False
        for sent in _split_sentences(text):
            code = re.search(r"\b\d{6}\b", sent)
            key = ParametricMemorySystem._key_of(sent)
            if code and key:
                self.pairs.append((key, code.group(0)))
                new = True
        if new:
            self._resolve()

    def _train_pair(self, key: str, code: str):
        self.pairs.append((key, code))
        self._resolve()

    def snapshot(self, session_id: str) -> bytes:
        obj = {"format": self._SNAP_FORMAT, "d": self.d_mem, "lam": self.lam,
               "seed": self.seed, "pairs": [list(p) for p in self.pairs]}
        return json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")

    def restore(self, blob: bytes) -> None:
        obj = json.loads(blob.decode("utf-8"))
        if obj.get("format") != self._SNAP_FORMAT:
            raise ValueError(f"not a micro_rls snapshot: {obj.get('format')!r}")
        self.d_mem = obj["d"]; self.lam = obj["lam"]; self.seed = obj["seed"]
        self.pairs = [tuple(p) for p in obj["pairs"]]
        self._resolve()


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

def make_system(name: str, *, encoder: str = "hash", topk: int = 8,
                dim: int = 256, max_tokens: int = 8000,
                device: str = "cpu", update_rule: str = "falcon_nlms",
                score_threshold: Optional[float] = None,
                consolidate_mode: str = "none",
                consolidate_param: Optional[float] = None,
                read_iters: int = 1, read_beta: float = 2.0,
                read_protocol: str = "direct",
                **fastweight_kwargs) -> MemorySystem:
    """Build a fresh system instance (the runner calls this per session to
    enforce the disk round-trip — no in-process state may survive)."""
    if name == "none":
        return NoneSystem()
    if name == "oracle":
        return OracleContextSystem(max_tokens=max_tokens)
    if name == "rag":
        return RagSystem(encoder=encoder, topk=topk, dim=dim, device=device,
                         score_threshold=score_threshold)
    if name == "rag_2hop":
        return RagTwoHopSystem(encoder=encoder, topk=topk, dim=dim, device=device)
    if name == "filesystem":
        return FilesystemSystem()
    if name == "parametric":
        return ParametricMemorySystem(d_mem=dim, update_rule=update_rule,
                                      consolidate_mode=consolidate_mode,
                                      consolidate_param=consolidate_param,
                                      read_iters=read_iters,
                                      read_beta=read_beta,
                                      read_protocol=read_protocol)
    if name == "micro_fw":
        return MicroFastWeightSystem(d_mem=dim, lr=0.1)
    if name == "micro_rls":
        return MicroRLSSystem(d_mem=dim)
    if name == "fastweight":
        return FastWeightSystem(device=device, **fastweight_kwargs)
    raise ValueError(
        f"unknown system {name!r} "
        f"(expected none|oracle|rag|parametric|fastweight)")
