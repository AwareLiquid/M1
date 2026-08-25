"""test_rag.py — mt_lnn/rag 的纯逻辑单测（不依赖 torch）。"""
from mt_lnn.rag import (BM25Index, build_index_from_texts, chunk_text,
                        format_context, tokenize)


PASSAGES = [
    "The first Nobel Prize in Physics was awarded to Wilhelm Conrad Rontgen.",
    "Deadpool 2 was released in the United States on May 18, 2018.",
    "The capital of France is Paris and it sits on the Seine.",
    "Liquid neural networks have constant memory and are inspired by neurons.",
]


def test_tokenize_basic():
    assert tokenize("Hello, WORLD-99") == ["hello", "world", "99"]
    assert tokenize("中文测试") == ["中", "文", "测", "试"]


def test_chunk_text():
    text = " ".join(f"w{i}" for i in range(300))
    chunks = chunk_text(text, chunk_words=100, overlap_words=20)
    assert len(chunks) >= 3
    assert all(len(c) > 40 for c in chunks)


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_search_ranking():
    idx = BM25Index(PASSAGES)
    hits = idx.search("capital of France", top_k=2)
    assert hits, "应命中"
    assert hits[0][0] == 2  # Paris 段落排第一
    assert hits[0][1] >= hits[-1][1]


def test_retrieve_returns_passages():
    idx = BM25Index(PASSAGES)
    out = idx.retrieve("nobel physics", top_k=1)
    assert out == [PASSAGES[0]]


def test_empty_corpus_and_query():
    idx = BM25Index([])
    assert idx.search("anything") == []
    assert idx.retrieve("anything") == []
    idx2 = BM25Index(PASSAGES)
    assert idx2.search("") == []
    assert idx2.search("   ") == []


def test_coverage():
    idx = BM25Index(PASSAGES)
    qs = ["who won the first nobel prize in physics",
          "when did deadpool 2 come out"]
    golds = [["Wilhelm Conrad Rontgen"], ["May 18, 2018"]]
    assert idx.coverage(qs, golds, top_k=2) == 1.0
    # 语料里没有的答案 → 0 覆盖
    qs2 = ["who invented the transistor"]
    assert idx.coverage(qs2, [["John Bardeen"]], top_k=5) == 0.0


def test_coverage_empty_queries():
    idx = BM25Index(PASSAGES)
    assert idx.coverage([], []) == 0.0


def test_build_index_from_texts():
    idx = build_index_from_texts(["word " * 250, "other " * 250])
    assert len(idx) >= 4  # 两文档各切出多块
    assert idx.retrieve("word word word", top_k=1)


def test_format_context_truncation():
    hits = [PASSAGES[0]] * 100
    ctx = format_context(hits, max_chars=300)
    assert len(ctx) <= 300
    assert format_context([], max_chars=10) == ""
