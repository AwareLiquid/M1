"""Typed personal memory cards for conversational recall.

DESIGN BORROWED FROM (and credited to) the Awareness-Market project
(github.com/everest-an/Awareness-Market): knowledge is stored as TYPED cards
(category + confidence), and retrieval is biased by classifying the query and
boosting cards of the matching type (its query-type-router / cardTypeBoost).
Awareness-Market uses a 7-category PERSONAL taxonomy
  personal_preference / important_detail / plan_intention /
  activity_preference / health_info / career_info / custom_misc
and a multilingual-e5 embedder for zero-keyword, language-agnostic routing.

ADAPTED TO OUR CASE (a small "Her-like" companion on a 0.5B edge stack):
  - Reuse OUR shared SentenceEncoder (mt_lnn.sentence_encoder), which defaults to
    multilingual-e5-small -- the SAME embedder family Awareness-Market uses -- so
    there is ONE encoder for both recall and typing, and Chinese statements route
    to the same type as their English equivalents (verified 4/4 Chinese in
    _diag_memory_cards.py; an English-only encoder mis-typed them).
  - A companion taxonomy with an explicit EMOTION type (the feature ask), since
    "remembering how you felt" is central to a Her-like experience and was not a
    first-class category upstream.
  - Classification is pure embedding nearest-neighbour to per-type ANCHOR
    phrasings (no hard-coded keywords -> language-agnostic via the shared space).
    Below a confidence floor a statement falls back to the catch-all `detail`
    type rather than being mis-typed.

HONEST SCOPE: this is a lightweight semantic TAGGER, not affect/intent
understanding. Calling a sentence `emotion` means "it embeds near emotional
self-disclosure phrasings", NOT that the system feels or comprehends emotion.
It improves which past statement is recalled and lets the assistant know a memory
is emotional -- useful, bounded, and not a claim of sentience.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Card taxonomy. Each type carries STATEMENT anchors (how a user phrases such a
# fact about themselves) and QUERY anchors (how a later turn asks for it). Both
# are concatenations of diverse natural phrasings -- more anchors -> tighter
# embedding cluster -> better routing (the Awareness-Market archetype recipe).
# English text only; recognition is multilingual via the shared bge space.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CardType:
    id: str
    # Each side holds SEPARATE anchors per language. They are kept apart (not
    # concatenated) so a statement is scored as the MAX cosine over a type's
    # anchors -- an English sentence matches the English anchor cleanly and a
    # Chinese one matches the Chinese anchor, with no centroid dilution. Folding
    # both languages into a single anchor vector was measured to blur the dense
    # e5 clusters and flip English near-ties (preference<->plan).
    statement_anchors: Tuple[str, ...]
    query_anchors: Tuple[str, ...]


# Anchors are example phrasings, not keyword rules -- bilingual (EN + ZH) because
# this is a bilingual project; the per-type max-over-anchors keeps each language
# crisp while the shared e5 space still bridges unseen languages to the nearest.
CARD_TYPES: Tuple[CardType, ...] = (
    CardType(
        "identity",
        ("My name is. I am called. I work as. My job is. I am a. I live in. "
         "I am years old. My profession is. This is who I am.",
         "我叫. 我的名字是. 我是一名. 我的职业是. 我住在. 我今年. 我是谁."),
        ("What is my name. Who am I. What do I do for work. What is my "
         "profession. Where do I live. How old am I.",
         "我叫什么名字. 我是做什么的. 我的职业是什么. 我住在哪里."),
    ),
    CardType(
        "preference",
        ("I love. I like. I enjoy. I prefer. My favourite is. I am a fan of. "
         "I hate. I dislike. I can't stand. I am into. My hobby is.",
         "我喜欢. 我爱. 我享受. 我最喜欢的是. 我讨厌. 我的爱好是. 我喜欢做."),
        ("What do I like. What do I enjoy. What are my hobbies. What is my "
         "favourite. What do I prefer. What do I do for fun.",
         "我喜欢什么. 我的爱好是什么. 我最喜欢什么."),
    ),
    CardType(
        "emotion",
        ("I feel. I am happy. I am sad. I am anxious. I am stressed. I am "
         "excited. I am angry. I am lonely. I am scared. I am proud. I am "
         "worried. It makes me feel. I have been feeling. My mood is.",
         "我感到. 我觉得. 我很开心. 我很难过. 我很焦虑. 我很孤独. 我的心情. 这让我感到."),
        ("How do I feel. How was I feeling. What is my mood. Was I happy or "
         "sad. What was I worried about. How did that make me feel.",
         "我的心情如何. 我感觉怎么样. 我在担心什么. 我开心吗."),
    ),
    CardType(
        "plan",
        ("I plan to. I am going to. I want to. I intend to. I will. My goal is. "
         "I hope to. I am planning. Next I will. I am thinking of doing.",
         "我打算. 我计划. 我想要. 我准备. 我的目标是. 我希望."),
        ("What are my plans. What do I want to do. What is my goal. What am I "
         "going to do. What did I intend to do.",
         "我的计划是什么. 我想做什么. 我的目标是什么."),
    ),
    CardType(
        "relationship",
        ("My friend. My partner. My wife. My husband. My mother. My father. My "
         "brother. My sister. My colleague. My boss. My child. My family.",
         "我的朋友. 我的伴侣. 我的妻子. 我的丈夫. 我的妈妈. 我的爸爸. 我的同事. 我的家人."),
        ("Who are my friends. Who is my partner. Tell me about my family. Who "
         "do I know. Who is my colleague.",
         "我的朋友是谁. 我的家人有谁. 我认识谁."),
    ),
    CardType(
        "health",
        ("I am allergic to. I have a condition. I take medication for. My "
         "diagnosis is. I cannot eat. My doctor. My health. I am intolerant to.",
         "我对某物过敏. 我有某种疾病. 我在吃某种药. 我不能吃. 我的身体. 我的健康状况."),
        ("What am I allergic to. What is my health condition. What medication "
         "do I take. What can't I eat. What are my dietary restrictions.",
         "我对什么过敏. 我的健康状况怎样. 我在吃什么药. 我不能吃什么."),
    ),
    # Catch-all -- a concrete personal fact that fits none of the above. Kept as
    # the fallback when no specific type clears the confidence floor.
    CardType(
        "detail",
        ("An important detail about me. Something to remember about me. A fact "
         "about my life. Personal information worth keeping.",
         "关于我的一个重要细节. 需要记住的关于我的事. 我生活中的一个事实."),
        ("What do you know about me. Remind me of a detail. What did I tell you.",
         "你知道关于我的什么. 提醒我一个细节. 我告诉过你什么."),
    ),
)

CARD_TYPE_IDS: Tuple[str, ...] = tuple(t.id for t in CARD_TYPES)
FALLBACK_TYPE = "detail"


class MemoryCardClassifier:
    """Tag a personal statement (or a recall query) with a card type by nearest
    anchor in the shared sentence-embedding space.

    Parameters
    ----------
    encode_fn:
        ``str -> 1-D float tensor`` statement encoder (e.g.
        ``SentenceEncoder().as_fn(is_query=False)``). Anchors and statements are
        embedded with this.
    query_encode_fn:
        Optional separate query-side encoder (bge's asymmetric recipe). When
        given, query anchors and incoming queries use it; otherwise encode_fn is
        used for both.
    confidence_floor:
        Minimum cosine to the best type before we trust it. A statement under the
        floor is typed as the catch-all ``detail`` (honest: do not over-commit a
        weak signal to a specific category).
    """

    def __init__(
        self,
        encode_fn: Callable[[str], torch.Tensor],
        query_encode_fn: Optional[Callable[[str], torch.Tensor]] = None,
        confidence_floor: float = 0.30,
    ):
        self.encode_fn = encode_fn
        self.query_encode_fn = query_encode_fn or encode_fn
        self.confidence_floor = float(confidence_floor)
        # Pre-embed anchors once (lazily on first use, so construction is cheap).
        # Each is a (anchor_vectors, owner_type_index) pair: anchors from all
        # types are stacked into one matrix and `_owner` maps each row back to its
        # type, so classification is one mat-vec then a max-per-type reduction.
        self._stmt_keys: Optional[torch.Tensor] = None
        self._query_keys: Optional[torch.Tensor] = None
        self._stmt_owner: Optional[torch.Tensor] = None
        self._query_owner: Optional[torch.Tensor] = None

    @staticmethod
    def _stack(anchor_lists, enc) -> Tuple[torch.Tensor, torch.Tensor]:
        vecs, owners = [], []
        for ti, anchors in enumerate(anchor_lists):
            for a in anchors:
                vecs.append(F.normalize(enc(a), dim=0))
                owners.append(ti)
        return torch.stack(vecs), torch.tensor(owners, dtype=torch.long)

    def _ensure_anchors(self) -> None:
        if self._stmt_keys is None:
            self._stmt_keys, self._stmt_owner = self._stack(
                [t.statement_anchors for t in CARD_TYPES], self.encode_fn)
        if self._query_keys is None:
            self._query_keys, self._query_owner = self._stack(
                [t.query_anchors for t in CARD_TYPES], self.query_encode_fn)

    def _type_scores(self, text: str, keys: torch.Tensor, owner: torch.Tensor,
                     enc: Callable[[str], torch.Tensor]) -> torch.Tensor:
        """Per-type score = MAX cosine over that type's anchors (so EN and ZH
        anchors of one type don't dilute each other)."""
        v = F.normalize(enc(text), dim=0)
        raw = keys @ v                                   # (n_anchors,)
        out = torch.full((len(CARD_TYPE_IDS),), -1.0)
        out = out.scatter_reduce(0, owner, raw, reduce="amax", include_self=True)
        return out

    def _classify(self, text, keys, owner, enc) -> Tuple[str, float]:
        if not text or not text.strip():
            return FALLBACK_TYPE, 0.0
        scores = self._type_scores(text, keys, owner, enc)
        idx = int(scores.argmax())
        best = float(scores[idx])
        if best < self.confidence_floor:
            return FALLBACK_TYPE, best
        return CARD_TYPE_IDS[idx], best

    def classify_statement(self, text: str) -> Tuple[str, float]:
        """Return ``(card_type, confidence)`` for a user statement."""
        self._ensure_anchors()
        return self._classify(text, self._stmt_keys, self._stmt_owner, self.encode_fn)

    def classify_query(self, text: str) -> Tuple[str, float]:
        """Return the card type a recall *query* is asking for, plus confidence.

        Used to bias retrieval toward matching-type cards (the cardTypeBoost
        idea). A below-floor query returns ``detail`` -> no meaningful boost.
        """
        self._ensure_anchors()
        return self._classify(text, self._query_keys, self._query_owner,
                              self.query_encode_fn)

    def scores(self, text: str, is_query: bool = False) -> Dict[str, float]:
        """Full per-type cosine map (diagnostic / debugging aid)."""
        self._ensure_anchors()
        if is_query:
            s = self._type_scores(text, self._query_keys, self._query_owner,
                                  self.query_encode_fn)
        else:
            s = self._type_scores(text, self._stmt_keys, self._stmt_owner,
                                  self.encode_fn)
        return {CARD_TYPE_IDS[i]: float(s[i]) for i in range(len(CARD_TYPE_IDS))}
