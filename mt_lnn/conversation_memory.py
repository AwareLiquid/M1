"""Automatic episodic conversation memory.

Wraps a PersistentKnowledgeMemory with a thin policy layer so a chat assistant
"remembers you": the user's own statements are stored as they are said, and the
relevant ones are recalled on later turns (and later sessions, since the store
is SQLite-backed). This is the concrete, buildable slice of a "Her-like"
experience -- continuity across turns -- as opposed to the science-fiction part.

HONEST SCOPE (read this before claiming anything):
  - This is AUTOMATED RETRIEVAL (RAG) over past user utterances. It makes the
    assistant *recall what you told it*. It does NOT add understanding,
    reasoning, emotion, or self-awareness, and it does not make the base model
    smarter. A 0.5B model with episodic recall is still a 0.5B model.
  - Storage is honest-by-default: trivially short messages and near-duplicates
    are skipped so the store does not fill with noise, and nothing is ever
    fabricated -- recall only returns things the user actually said.
  - Quality is bounded by the encoder. Keys are mean-pooled LM hidden states
    (anisotropy-robust centered cosine); good for edge-scale recall, not a
    substitute for a dedicated sentence-embedding model.

The encoder is injected (encode_fn: str -> 1-D float tensor) so this module
stays decoupled from any specific model/server.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch


# Lightweight heuristics for "is this user turn worth remembering?". A statement
# the user makes about themselves / the world is worth storing; a bare command
# or a one-word acknowledgement is not. Kept deliberately simple and language
# agnostic (works for CJK too, which has no spaces) -- the goal is to cut obvious
# noise, not to be a perfect salience classifier.
_TRIVIAL = {
    "ok", "okay", "yes", "no", "yeah", "nope", "sure", "thanks", "thank you",
    "hi", "hello", "hey", "bye", "k", "lol", "haha",
}


class EpisodicConversationMemory:
    """Store the user's statements and recall the relevant ones on later turns.

    Parameters
    ----------
    store:
        A PersistentKnowledgeMemory (or anything with the same write/query/len
        surface). Persistence and eviction are delegated to it.
    encode_fn:
        Callable mapping text -> a 1-D float tensor key (e.g. the server's
        mean-pooled-hidden-state encoder). Must match the store's key_dim.
    min_chars:
        Skip utterances shorter than this (after strip). Filters "ok"/"yes".
    dedup_threshold:
        Skip an utterance whose centered-cosine similarity to an already-stored
        one is >= this, so repeats ("my name is Alex" said twice) are stored once.
    score_floor:
        Recall hits below this similarity are dropped (the honest floor -- an
        off-topic turn recalls nothing rather than a spurious memory).
    center:
        Whether to use the store's anisotropy-robust centered cosine. Recommended
        True when *encode_fn* is a mean-pooled LM (whose hidden states share a
        dominant direction); set False for a dedicated sentence-embedding model
        (e.g. bge / MiniLM), which is already isotropic enough that subtracting
        the small-corpus mean only adds noise. The score_floor should be tuned
        per-encoder since the two regimes produce different score scales.
    """

    def __init__(
        self,
        store,
        encode_fn: Callable[[str], torch.Tensor],
        min_chars: int = 8,
        dedup_threshold: float = 0.97,
        score_floor: float = 0.15,
        center: bool = True,
    ):
        self.store = store
        self.encode_fn = encode_fn
        self.min_chars = int(min_chars)
        self.dedup_threshold = float(dedup_threshold)
        self.score_floor = float(score_floor)
        self.center = bool(center)

    # -- write side -------------------------------------------------------
    def _worth_remembering(self, text: str) -> bool:
        t = (text or "").strip()
        if len(t) < self.min_chars:
            return False
        if t.lower().rstrip("!.") in _TRIVIAL:
            return False
        return True

    def observe(self, user_text: str, meta: Optional[dict] = None) -> Optional[int]:
        """Store a user utterance if it is worth remembering and not a near-dup.

        Returns the new record id, or None if the turn was skipped (trivial or a
        duplicate of something already stored).
        """
        if not self._worth_remembering(user_text):
            return None
        key = self.encode_fn(user_text)
        # Duplicate check: compare against the closest existing memory. Uses the
        # store's own centered cosine so it matches recall-time scoring.
        if len(self.store) > 0:
            top = self.store.query(key, top_k=1, touch=False, center=self.center)
            if top and top[0][1] >= self.dedup_threshold:
                return None
        return self.store.write(key, user_text.strip(), meta=meta or {})

    # -- read side --------------------------------------------------------
    def recall(self, query_text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Return up to *top_k* past user statements relevant to *query_text*,
        as (content, score) above the honest score floor. Empty when the store
        is empty or nothing is relevant (never fabricates)."""
        if not query_text or not query_text.strip() or len(self.store) == 0:
            return []
        hits = self.store.query(
            self.encode_fn(query_text), top_k=top_k, center=self.center)
        return [(str(c), float(s)) for c, s, _ in hits if s >= self.score_floor]

    def recall_context(self, query_text: str, top_k: int = 3) -> str:
        """Format recalled statements as a grounding block for the prompt, or an
        empty string when nothing is recalled. The caller decides whether/how to
        prepend it -- this never mutates the prompt itself."""
        hits = self.recall(query_text, top_k=top_k)
        if not hits:
            return ""
        lines = "\n".join(f"- {c}" for c, _ in hits)
        return ("The user told you earlier:\n" + lines +
                "\nUse this to stay consistent and personal if relevant.")

    def __len__(self) -> int:
        return len(self.store)
