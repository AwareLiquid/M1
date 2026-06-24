"""A dedicated sentence-embedding encoder for episodic recall.

WHY THIS EXISTS (evidence, not preference):
  The conversation/knowledge stores need a text -> key function. The obvious
  reuse is the chat model's own mean-pooled hidden state, but a diagnostic
  (_diag_conv_mem.py) showed mean-pooled Qwen2.5-0.5B CANNOT separate
  first-person paraphrases: "I love hiking" always loses to "marine biologist"
  (the anisotropic universal nearest neighbour), best 2/3 across every pooling
  + centering variant. A small purpose-built sentence model gets it right, so
  for recall QUALITY we use a real embedder, not the LM's hidden state.

WHY MULTILINGUAL (intfloat/multilingual-e5-small, ~118M, the default):
  This project is bilingual -- the companion is spoken to in Chinese AND English.
  An English-only encoder (BAAI/bge-small-en-v1.5) recalls English well but
  mis-handles Chinese: it could not reliably recall Chinese queries and typed
  Chinese statements at only 4/5. multilingual-e5-small recalls 5/5 including
  Chinese queries and types 10/11 (_diag_e5_multilingual.py vs _diag_bge.py),
  because its embedding space is shared across languages. e5 is the family
  Awareness-Market uses for the same reason. bge is still selectable (English-
  only deploys) -- the per-model recipe below adapts pooling + prefixes.

HONEST SCOPE: this only makes RETRIEVAL good ("好用"); it adds no understanding,
reasoning, or emotion. It is the right tool for "remember what the user said",
nothing more.

The model is loaded lazily on first encode and cached, so importing this module
is cheap and a server that never enables conversation memory pays nothing. Each
model family has its own recipe: bge uses CLS pooling and prefixes only the
query; e5 uses masked-mean pooling and prefixes BOTH sides ("query: "/"passage:
"). Asymmetric prefixing (is_query=True/False) widens the relevant/off-topic gap.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F

# Default: a MULTILINGUAL embedder. This project is bilingual (the companion is
# spoken to in Chinese and English), and an English-only encoder mis-routes
# Chinese (verified: bge-small-en typed Chinese statements at 4/5 and could not
# recall Chinese queries; multilingual-e5-small got recall 5/5 incl. Chinese --
# see _diag_e5_multilingual.py). e5-small is ~118M, fine at edge scale, and is
# the same family Awareness-Market uses for language-agnostic routing.
DEFAULT_MODEL = "intfloat/multilingual-e5-small"

# Per-model "recipe": pooling and the instruction prefixes each side needs.
#   pooling   - "cls" (bge: take last_hidden_state[:,0]) or "mean" (e5: masked
#               mean over tokens).
#   query     - prefix for is_query=True text.
#   passage   - prefix for is_query=False (stored statement) text.
# e5 REQUIRES "query: "/"passage: " on BOTH sides; bge prefixes only the query.
# Matched by substring so revision-tagged ids ("...-v1.5") still resolve; the
# final entry is the default for unknown ids (mean-pool, no prefix -- safe).
_RECIPES = (
    ("multilingual-e5", {"pooling": "mean", "query": "query: ", "passage": "passage: "}),
    ("e5-",             {"pooling": "mean", "query": "query: ", "passage": "passage: "}),
    ("bge-",            {"pooling": "cls",
                         "query": "Represent this sentence for searching relevant passages: ",
                         "passage": ""}),
    ("",                {"pooling": "mean", "query": "", "passage": ""}),
)


def _recipe_for(model_id: str) -> dict:
    mid = (model_id or "").lower()
    for needle, rec in _RECIPES:
        if needle in mid:
            return rec
    return _RECIPES[-1][1]


class SentenceEncoder:
    """Lazy-loaded, L2-normalized sentence embedder that adapts to the model's
    pooling + instruction recipe (CLS/no-passage-prefix for bge, masked-mean +
    query:/passage: prefixes for e5).

    Parameters
    ----------
    model_id:
        HF encoder id. The pooling and prefixes are inferred from the id via
        ``_recipe_for`` (override with the explicit kwargs below). Defaults to
        the multilingual e5-small so Chinese and English both work.
    device:
        Torch device string for the forward pass. CPU is fine at edge scale.
    pooling / query_instruction / passage_instruction:
        Explicit overrides for the inferred recipe (None -> use the inferred
        value). Set query/passage to "" for symmetric, prefix-free embedding.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str = "cpu",
        pooling: Optional[str] = None,
        query_instruction: Optional[str] = None,
        passage_instruction: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = device
        rec = _recipe_for(model_id)
        self.pooling = (pooling or rec["pooling"]).lower()
        self.query_instruction = (
            rec["query"] if query_instruction is None else query_instruction)
        self.passage_instruction = (
            rec["passage"] if passage_instruction is None else passage_instruction)
        self._tok = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id).eval().to(self.device)

    @torch.no_grad()
    def encode(self, text: str, is_query: bool = False) -> torch.Tensor:
        """Return a 1-D L2-normalized float32 key for *text* (CPU tensor).

        The query/passage instruction for the chosen side is prepended, and the
        configured pooling is applied -- so the same call works for bge (CLS,
        query-only prefix) and e5 (masked mean, both-side prefixes).
        """
        self._ensure_loaded()
        prefix = self.query_instruction if is_query else self.passage_instruction
        enc = self._tok(prefix + (text or " "), return_tensors="pt",
                        truncation=True, max_length=256, padding=True).to(self.device)
        out = self._model(**enc)
        if self.pooling == "cls":
            vec = out.last_hidden_state[:, 0]                       # (1, d)
        else:                                                       # masked mean
            mask = enc.attention_mask.unsqueeze(-1).float()
            summed = (out.last_hidden_state * mask).sum(dim=1)
            vec = summed / mask.sum(dim=1).clamp(min=1e-9)
        return F.normalize(vec, dim=-1)[0].float().cpu()           # (d,)

    @property
    def dim(self) -> int:
        """Embedding dimension (probes the model once)."""
        return int(self.encode("dimension probe").numel())

    def as_fn(self, is_query: bool = False) -> Callable[[str], torch.Tensor]:
        """Return a plain ``str -> tensor`` closure (the encode_fn the memory
        modules expect). ``is_query`` fixes the query/statement side."""
        return lambda text: self.encode(text, is_query=is_query)
