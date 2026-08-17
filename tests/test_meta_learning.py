import numpy as np

from mt_lnn.meta_learning import cluster_evidence, hash_embed, kmeans


def test_hash_embed_deterministic_and_unit_norm():
    v1 = hash_embed("explain m-theory")
    v2 = hash_embed("explain m-theory")
    assert np.allclose(v1, v2)
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-5


def test_kmeans_separates_two_blobs():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=+3.0, size=(20, 4))
    b = rng.normal(loc=-3.0, size=(20, 4))
    X = np.vstack([a, b]).astype(np.float32)
    res = kmeans(X, k=2)
    labels = np.array(res.labels)
    # Each blob should be mostly one label.
    purity = max(
        (labels[:20] == labels[0]).mean() + (labels[20:] != labels[0]).mean(),
        (labels[:20] != labels[0]).mean() + (labels[20:] == labels[0]).mean(),
    ) / 2
    assert purity > 0.85


def test_cluster_evidence_groups_similar_queries():
    rows = [
        {"query": "explain m theory origins", "source": "mock"},
        {"query": "explain m theory history", "source": "mock"},
        {"query": "weather in tokyo today", "source": "gemini"},
        {"query": "weather in tokyo tomorrow", "source": "gemini"},
    ]
    out = cluster_evidence(rows, k=2)
    assert out["n_rows"] == 4
    assert 1 <= len(out["clusters"]) <= 2


def test_cluster_evidence_empty():
    out = cluster_evidence([])
    assert out["n_rows"] == 0
    assert out["clusters"] == []
