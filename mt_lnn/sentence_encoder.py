"""A dedicated sentence-embedding encoder for episodic recall.

WHY THIS EXISTS (evidence, not preference):
  The conversation/knowledge stores need a text -> key function. The obvious
  reuse is the chat model's own mean-pooled hidden state, but a diagnostic
  (_diag_conv_mem.py) showed mean-pooled Qwen2.5-0.5B CANNOT separate
  first-person paraphrases: "I love hiking" always loses to "marine biologist"
  (the anisotropic universal nearest neighbour), best 2/3 across every pooling
  + centering variant. A small purpose-built sentence model (BAAI/bge-small-en
  -v1.5, ~33M params) gets 3/3 with clear margins (_diag_bge.py). So for recall
  QUALITY we use a real embedder, not the LM's hidden state.

HONEST SCOPE: this only makes RETRIEVAL good ("好用"); it adds no understanding,
reasoning, or emotion. It is the right tool for "remember what the user said",
nothing more.

The model is loaded lazily on first encode and cached, so importing this module
is cheap and a server that never enables conversation memory pays nothing.
bge's documented recipe is CLS-pooling + L2-normalize; for asymmetric
short-query -> statement retrieval bge-*-en-v1.5 also recommends prefixing the
QUERY (not the stored statement) with a fixed instruction, which widens the gap
between relevant and off-topic hits -- exposed via is_query=True.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# bge-en-v1.5 retrieval instruction for queries (documented by the model card).
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class SentenceEncoder:
    """Lazy-loaded CLS-pooled, L2-normalized sentence embedder.

    Parameters
    ----------
    model_id:
        Any HF encoder whose CLS token (index 0 of last_hidden_state) is a
        sentence representation. Defaults to bge-small-en-v1.5.
    device:
        Torch device string for the forward pass. CPU is fine at edge scale.
    query_instruction:
        Prefix prepended to texts encoded with ``is_query=True``. Set to "" to
        disable the asymmetric query prompt (for symmetric s2s similarity).
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str = "cpu",
        query_instruction: Optional[str] = _QUERY_INSTRUCTION,
    ):
        self.model_id = model_id
        self.device = device
        self.query_instruction = query_instruction or ""
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

        When ``is_query`` is True and a query instruction is configured, the
        instruction is prepended -- use it for the recall query, NOT for stored
        statements, so the two sides match bge's asymmetric retrieval recipe.
        """
        self._ensure_loaded()
        s = (self.query_instruction + (text or " ")) if is_query else (text or " ")
        enc = self._tok(s, return_tensors="pt", truncation=True,
                        max_length=256, padding=True).to(self.device)
        out = self._model(**enc)
        cls = out.last_hidden_state[:, 0]                 # (1, d) CLS pooling
        return F.normalize(cls, dim=-1)[0].float().cpu()  # (d,)

    @property
    def dim(self) -> int:
        """Embedding dimension (probes the model once)."""
        return int(self.encode("dimension probe").numel())

    def as_fn(self, is_query: bool = False) -> Callable[[str], torch.Tensor]:
        """Return a plain ``str -> tensor`` closure (the encode_fn the memory
        modules expect). ``is_query`` fixes the query/statement side."""
        return lambda text: self.encode(text, is_query=is_query)
