"""rag.py — 轻量检索层（BM25），为推理栈的事实增强做准备。

模型无关：2B 权重就绪后 serve 栈直接调用（roadmap.md Phase 2）。
零外部依赖（自实现 BM25，确定性、可单测）。

设计依据（H_P1 实测）:
  - big+rag 事实 EM 7.2×（检索是大模型的放大器）
  - coverage（答案证据是否被检索到）是收益上限 —— 当前 BM25 在 NQ 自建语料
    上 coverage≈59%，换 dense 检索器可再提
"""
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

_WORD_RE = re.compile(r"[a-zA-Z0-9\u4e00-\u9fff']+")


def tokenize(text: str) -> List[str]:
    """英文小写分词 + 中文按单字切（BPE 之外的朴素方案，够 BM25 用）。"""
    out: List[str] = []
    for w in _WORD_RE.findall(text):
        w = w.lower()
        if re.fullmatch(r"[\u4e00-\u9fff]+", w):
            out.extend(list(w))  # 中文连续串按单字切
        else:
            out.append(w)
    return out


def chunk_text(text: str, chunk_words: int = 100,
               overlap_words: int = 20) -> List[str]:
    """按词数切块（重叠滑窗）。返回块列表。"""
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    step = max(1, chunk_words - overlap_words)
    while start < len(words):
        chunk = " ".join(words[start:start + chunk_words])
        if len(chunk.strip()) > 40:
            chunks.append(chunk.strip())
        start += step
    return chunks


class BM25Index:
    """标准 Okapi BM25（k1=1.5, b=0.75），索引后只读。"""

    def __init__(self, passages: Sequence[str],
                 k1: float = 1.5, b: float = 0.75):
        self.passages: List[str] = list(passages)
        self.k1, self.b = k1, b
        self._tok: List[List[str]] = [tokenize(p) for p in self.passages]
        self._doc_freq: Dict[str, int] = Counter()
        self._avg_len: float = 0.0
        for toks in self._tok:
            self._avg_len += len(toks)
            for term in set(toks):
                self._doc_freq[term] += 1
        n = max(1, len(self.passages))
        self._avg_len /= n
        self._idf: Dict[str, float] = {
            t: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for t, df in self._doc_freq.items()
        }

    def __len__(self) -> int:
        return len(self.passages)

    def _score(self, query_toks: List[str], doc_idx: int) -> float:
        toks = self._tok[doc_idx]
        if not toks:
            return 0.0
        tf = Counter(toks)
        doc_len = len(toks)
        score = 0.0
        for t in set(query_toks):
            idf = self._idf.get(t)
            if idf is None:
                continue
            f = tf.get(t, 0)
            if f == 0:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * doc_len / self._avg_len)
            score += idf * f * (self.k1 + 1) / denom
        return score

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """返回 [(passage_idx, score)]，按分降序。空查询返回 []。"""
        q_toks = tokenize(query)
        if not q_toks or not self.passages:
            return []
        scored = [(i, self._score(q_toks, i)) for i in range(len(self.passages))]
        scored = [(i, s) for i, s in scored if s > 0.0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:max(1, top_k)]

    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        return [self.passages[i] for i, _ in self.search(query, top_k)]

    def coverage(self, queries: Sequence[str],
                 gold_answers: Sequence[Sequence[str]],
                 top_k: int = 10) -> float:
        """top-k 命中率：任一 gold 答案出现在任一检索块中的问题占比。"""
        if not queries:
            return 0.0
        hit = 0
        for q, golds in zip(queries, gold_answers):
            ctx = " ".join(self.retrieve(q, top_k)).lower()
            if any(g.lower() in ctx for g in golds if g):
                hit += 1
        return hit / len(queries)


def build_index_from_passages(passages: Sequence[str]) -> BM25Index:
    return BM25Index(passages)


def build_index_from_texts(texts: Sequence[str],
                           chunk_words: int = 100) -> BM25Index:
    """多文档文本 → 切块 → 索引。"""
    passages: List[str] = []
    for t in texts:
        passages.extend(chunk_text(t, chunk_words=chunk_words))
    return BM25Index(passages)


def format_context(hits: Sequence[str], max_chars: int = 3000) -> str:
    """检索块拼成上下文（截断保护，含换行符在内不超过 max_chars）。"""
    out: List[str] = []
    used = 0
    for h in hits:
        if used >= max_chars:
            break
        piece = h[:max_chars - used]
        out.append(piece)
        used += len(piece) + 1  # +1 预留换行符
    return "\n".join(out)
