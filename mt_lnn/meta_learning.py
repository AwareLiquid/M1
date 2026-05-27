"""Phase 9 — Meta-learning over capsule history.

Cluster ``evidence_log`` rows from many capsules to surface recurring
question patterns. Stdlib + numpy only — no sklearn. The output is used
by the daemon to suggest "you've asked similar things before; here are
the prior absorbed facts" and to seed open_questions for new sessions.

We embed queries with a fixed hashing trick (avoid pulling a sentence
encoder dep into the daemon). This is sufficient for clustering query
*kinds*, not semantic synonymy.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


def hash_embed(text: str, dim: int = 128) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = text.lower().split()
    if not tokens:
        return vec
    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
        sign = 1.0 if (h >> 31) & 1 else -1.0
        vec[h % dim] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


@dataclass
class ClusterResult:
    labels: List[int]
    centroids: np.ndarray
    inertia: float


def kmeans(
    X: np.ndarray,
    k: int,
    *,
    max_iter: int = 50,
    seed: int = 0,
) -> ClusterResult:
    if X.shape[0] == 0:
        return ClusterResult(labels=[], centroids=np.zeros((0, X.shape[1])), inertia=0.0)
    k = max(1, min(k, X.shape[0]))
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=k, replace=False)
    centroids = X[idx].copy()
    labels = np.zeros(X.shape[0], dtype=np.int64)
    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=-1)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for c in range(k):
            members = X[labels == c]
            if members.shape[0]:
                centroids[c] = members.mean(axis=0)
    final_dists = np.linalg.norm(X - centroids[labels], axis=-1)
    return ClusterResult(
        labels=labels.tolist(),
        centroids=centroids,
        inertia=float((final_dists ** 2).sum()),
    )


def cluster_evidence(
    evidence_rows: Sequence[Dict],
    *,
    k: int = 4,
    dim: int = 128,
) -> Dict:
    """Cluster a flat list of evidence_log rows by their ``query`` field.

    Returns a dict with per-cluster summaries: representative query, size,
    sources histogram. Daemons surface these to seed open_questions.
    """
    queries = [str(r.get("query", "")) for r in evidence_rows]
    if not queries:
        return {"clusters": [], "n_rows": 0}
    X = np.stack([hash_embed(q, dim=dim) for q in queries])
    res = kmeans(X, k=k)
    clusters: List[Dict] = []
    for c in range(res.centroids.shape[0]):
        members = [i for i, lab in enumerate(res.labels) if lab == c]
        if not members:
            continue
        sources: Dict[str, int] = {}
        for i in members:
            src = str(evidence_rows[i].get("source", "?"))
            sources[src] = sources.get(src, 0) + 1
        clusters.append(
            {
                "cluster_id": c,
                "size": len(members),
                "representative_query": queries[members[0]],
                "sources": sources,
            }
        )
    clusters.sort(key=lambda d: -d["size"])
    return {"clusters": clusters, "n_rows": len(queries), "inertia": res.inertia}
