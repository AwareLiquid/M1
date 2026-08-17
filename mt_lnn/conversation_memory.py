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
        query_encode_fn: Optional[Callable[[str], torch.Tensor]] = None,
        classifier=None,
        type_boost: float = 0.05,
    ):
        self.store = store
        self.encode_fn = encode_fn
        # Asymmetric retrieval: some sentence models (bge-*-en-v1.5) embed a short
        # QUERY better with a task instruction prefix while storing the statement
        # plain. When a separate query encoder is supplied, recall uses it and the
        # write/dedup side keeps encode_fn; otherwise the same encoder is used for
        # both (symmetric). Measured to widen the relevant/off-topic gap from
        # +0.085 to +0.109 on the diagnostic set (_diag_bge_asym.py).
        self.query_encode_fn = query_encode_fn or encode_fn
        self.min_chars = int(min_chars)
        self.dedup_threshold = float(dedup_threshold)
        self.score_floor = float(score_floor)
        self.center = bool(center)
        # Optional typed-card layer (mt_lnn.memory_cards): when a classifier is
        # supplied, every stored statement is tagged with a card_type (identity /
        # preference / emotion / plan / relationship / health / detail) and recall
        # adds `type_boost` to a hit whose type matches the query's classified
        # type, so e.g. an emotion question prefers an emotion memory over an
        # equally-similar preference one. None -> the untyped behaviour is intact.
        self.classifier = classifier
        self.type_boost = float(type_boost)

    # -- write side -------------------------------------------------------
    def _worth_remembering(self, text: str) -> bool:
        t = (text or "").strip()
        if len(t) < self.min_chars:
            return False
        if t.lower().rstrip("!.") in _TRIVIAL:
            return False
        # Skip questions: a turn ending in a question mark is a REQUEST, not a
        # statement the assistant should remember about the user. Storing "what is
        # my name?" would pollute the store and let a later recall surface a past
        # question instead of the fact that answers it. Trailing "?" / full-width
        # "?" is a strong, language-agnostic interrogative signal (CJK included);
        # this is deliberately a noise cut, not a full salience/intent classifier.
        if t.endswith("?") or t.endswith("？"):
            return False
        return True

    def observe(self, user_text: str, meta: Optional[dict] = None) -> Optional[int]:
        """Store a user utterance if it is worth remembering and not a near-dup.

        Returns the new record id, or None if the turn was skipped (trivial or a
        duplicate of something already stored).

        When a classifier is configured, the statement is tagged with its
        ``card_type`` + ``type_confidence`` in meta so recall can prefer a memory
        whose type matches the query's type (the cardTypeBoost idea). The tag is
        only metadata -- it never changes WHAT is stored, just how it is ranked
        later, so the untyped behaviour is byte-identical when classifier is None.
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
        full_meta = dict(meta or {})
        if self.classifier is not None:
            ctype, conf = self.classifier.classify_statement(user_text)
            # Caller-supplied card_type wins (explicit override); otherwise tag.
            full_meta.setdefault("card_type", ctype)
            full_meta.setdefault("type_confidence", round(float(conf), 4))
        return self.store.write(key, user_text.strip(), meta=full_meta)

    # -- read side --------------------------------------------------------
    def recall_cards(self, query_text: str, top_k: int = 3) -> List[dict]:
        """Return up to *top_k* relevant past statements as rich dicts
        ``{content, score, card_type}`` (the typed form). ``card_type`` is the
        stored tag, or ``None`` when the memory was written untyped.

        Ranking: base score is the encoder's centered/plain cosine; when a
        classifier is configured, a hit whose stored ``card_type`` equals the
        query's classified type gets ``+type_boost`` BEFORE the floor + sort, so
        e.g. an emotion question prefers an emotion memory over an equally similar
        preference one. The boost is small and additive -- it re-orders near-ties,
        it does not manufacture relevance (an off-topic hit below the floor stays
        below it; the floor is checked on the boosted score, which can only help a
        genuine type match, never invent a hit out of noise since the raw cosine
        still has to be within ``type_boost`` of the floor)."""
        if not query_text or not query_text.strip() or len(self.store) == 0:
            return []
        # Pull a GENEROUS candidate pool, not just top_k, so the boost can
        # promote a matching-type memory that the raw cosine buried far down.
        # This matters because e5's same-language "magnet" effect can push the
        # correct memory well outside top_k (e.g. an allergy statement ranked
        # below the emotion magnet); a pool of only top_k*k would never see it.
        # The store caps at its own size, so over-asking is free at edge scale.
        pool = max(top_k * 5, 25) if self.classifier is not None else top_k
        raw = self.store.query(
            self.query_encode_fn(query_text), top_k=pool, center=self.center)
        q_type = None
        if self.classifier is not None:
            q_type, _ = self.classifier.classify_query(query_text)
        scored = []
        for content, score, meta in raw:
            ctype = (meta or {}).get("card_type")
            boosted = float(score)
            if q_type is not None and ctype == q_type:
                boosted += self.type_boost
            if boosted >= self.score_floor:
                scored.append({"content": str(content), "score": boosted,
                               "card_type": ctype})
        scored.sort(key=lambda h: h["score"], reverse=True)
        return scored[:top_k]

    def recall(self, query_text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Return up to *top_k* past user statements relevant to *query_text*,
        as (content, score) above the honest score floor. Empty when the store
        is empty or nothing is relevant (never fabricates). Thin (content, score)
        view over :meth:`recall_cards` for callers that don't need the type."""
        return [(h["content"], h["score"])
                for h in self.recall_cards(query_text, top_k=top_k)]

    def recall_context(self, query_text: str, top_k: int = 3) -> str:
        """Format recalled statements as a grounding block for the prompt, or an
        empty string when nothing is recalled. The caller decides whether/how to
        prepend it -- this never mutates the prompt itself. When a memory carries
        a card_type, it is surfaced as a ``[type]`` tag so the model knows a
        recalled line is e.g. emotional rather than a bare fact."""
        hits = self.recall_cards(query_text, top_k=top_k)
        if not hits:
            return ""
        lines = []
        for h in hits:
            ctype = h.get("card_type")
            tag = f"[{ctype}] " if ctype and ctype != "detail" else ""
            lines.append(f"- {tag}{h['content']}")
        return ("The user told you earlier:\n" + "\n".join(lines) +
                "\nUse this to stay consistent and personal if relevant.")

    def __len__(self) -> int:
        return len(self.store)
