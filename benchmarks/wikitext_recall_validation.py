"""
benchmarks/wikitext_recall_validation.py -- scale up the memory-stack validation
from a hand-built probe to REAL WikiText-2, on real e5 embeddings (CPU).

Why this exists (what it adds over memory_recall_validation.py)
--------------------------------------------------------------
memory_recall_validation.py proved the retrieval GEOMETRY on a small, hand-built
corpus: paraphrase recall = 1.00, graph 2-hop lift = +0.62, adaptive threshold
~= hand-tuned. Two honesty gaps remained, both about SCALE:

  1. Does flat retrieval stay "好用" on HUNDREDS of real, noisy WikiText passages
     (not 12 clean curated facts)? -> Probe A.
  2. The adaptive quantile threshold degenerated on a 2-item corpus (centered
     antipodal keys -> threshold ~ -1). Does it STABILISE as the corpus grows?
     -> Probe B.

Both run on CPU against the locally cached WikiText-2 (datasets) and the cached
multilingual-e5-small encoder.

Probe A -- retrieval quality at scale (no labels needed).
  Standard self-retrieval proxy: for each passage, hold out one sentence as the
  QUERY and index the REST as the passage. A query is correct iff it pulls back
  its own source passage out of all N-1 distractors. Reports Recall@1/@5/@10 and
  MRR over N real passages, center=False vs True. This is a legitimate "好用"
  number for the flat store on real text -- high recall means a reworded/partial
  query still finds the right document among hundreds.

Probe B -- adaptive-threshold stability vs corpus size.
  Re-runs calibrate_threshold_from_keys at N = 10, 25, 50, 100, 200 (several
  seeds each) and reports the mean +/- std of the chosen cut. A SHRINKING std as
  N grows is the honest answer to "is the data-derived knob trustworthy at scale,
  given it was degenerate at N=2". center=False here (e5 passage keys are not
  antipodal; centering is for the recurrent-signature consolidation path).

HONEST SCOPE: this is RETRIEVAL effectiveness on real text, still not an
end-to-end QA/task accuracy number (that needs a labelled QA set + a generator).
It answers "does the store recall the right passage on real noisy data at scale",
which is exactly the gap the curated probe left open.

Run:
    python benchmarks/wikitext_recall_validation.py [--n 300]
Needs: datasets (cached wikitext-2-raw-v1) + transformers (cached e5). Prints an
honest skip and exits 0 if either is unavailable.
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import sys
import tempfile
import time
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from mt_lnn.knowledge_memory import PersistentKnowledgeMemory
from mt_lnn.graph_memory import calibrate_threshold_from_keys
from mt_lnn.sentence_encoder import SentenceEncoder


# ---------------------------------------------------------------------------
# Corpus: assemble real WikiText-2 lines into multi-sentence passages
# ---------------------------------------------------------------------------

_HEADER = re.compile(r"^\s*=\s.*\s=\s*$")          # " = Title = " section lines
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _split_sentences(text: str) -> List[str]:
    parts = [s.strip() for s in _SENT_SPLIT.split(text)]
    return [s for s in parts if len(s) > 15]


def load_wikitext_passages(
    n: int, *, min_chars: int = 240, min_sentences: int = 3, seed: int = 0,
) -> List[str]:
    """Return up to ``n`` clean multi-sentence passages from cached WikiText-2.

    Lines are accumulated into paragraphs across blank-line boundaries; section
    headers (" = X = ") are dropped. Only paragraphs with >= ``min_sentences``
    splittable sentences and >= ``min_chars`` characters are kept, so each can be
    split into a held-out query + an indexable remainder.
    """
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    passages: List[str] = []
    buf: List[str] = []

    def _flush() -> None:
        if not buf:
            return
        para = " ".join(buf).strip()
        buf.clear()
        if len(para) >= min_chars and len(_split_sentences(para)) >= min_sentences:
            passages.append(para)

    for row in ds:
        line = row["text"].rstrip("\n")
        if line.strip() == "":
            _flush()
            if len(passages) >= n * 3:        # gather a surplus, then subsample
                break
            continue
        if _HEADER.match(line):
            _flush()
            continue
        buf.append(line.strip())
    _flush()

    # Deterministic subsample to exactly n (surplus -> stable shuffle -> head).
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(passages), generator=g).tolist()
    return [passages[i] for i in idx[:n]]


# ---------------------------------------------------------------------------
# Probe A: retrieval quality at scale (held-out sentence -> own passage)
# ---------------------------------------------------------------------------

def run_retrieval_at_scale(
    enc: SentenceEncoder, dim: int, passages: List[str], *, center: bool,
) -> dict:
    """Hold out one sentence per passage as the query, index the rest, measure
    Recall@1/@5/@10 and MRR over all N passages (the others are distractors)."""
    with tempfile.TemporaryDirectory() as tmp:
        kb = PersistentKnowledgeMemory(key_dim=dim, db_path=os.path.join(tmp, "wt.db"))
        try:
            queries: List[str] = []
            gold_ids: List[int] = []
            for p in passages:
                sents = _split_sentences(p)
                # Hold out the MIDDLE sentence as the query; index the remainder,
                # so the query text is not a literal prefix of the stored passage.
                h = len(sents) // 2
                query = sents[h]
                remainder = " ".join(sents[:h] + sents[h + 1:])
                rid = kb.write(enc.encode(remainder, is_query=False), remainder)
                queries.append(query)
                gold_ids.append(rid)

            hit1 = hit5 = hit10 = 0
            rr_sum = 0.0
            n = len(passages)
            for query, gold in zip(queries, gold_ids):
                qk = enc.encode(query, is_query=True)
                hits = kb.query(qk, top_k=10, touch=False, center=center,
                                return_ids=True)
                ranked = [h[0] for h in hits]
                if gold in ranked:
                    rank = ranked.index(gold) + 1
                    rr_sum += 1.0 / rank
                    hit10 += 1
                    if rank <= 5:
                        hit5 += 1
                    if rank == 1:
                        hit1 += 1
        finally:
            kb.close()

    return {"n": n, "center": center,
            "recall@1": hit1 / n, "recall@5": hit5 / n,
            "recall@10": hit10 / n, "mrr": rr_sum / n}


# ---------------------------------------------------------------------------
# Probe B: adaptive-threshold stability vs corpus size
# ---------------------------------------------------------------------------

def run_threshold_stability(
    enc: SentenceEncoder, passages: List[str], *,
    sizes: Tuple[int, ...] = (10, 25, 50, 100, 200),
    quantile: float = 0.90, n_seeds: int = 5,
) -> List[dict]:
    """Encode the passages once, then for each corpus size draw several random
    subsets and report mean/std of the quantile-derived link threshold. A
    shrinking std with N is the honest evidence that the adaptive cut, degenerate
    at N=2, stabilises at scale."""
    keys = torch.stack(
        [enc.encode(p, is_query=False) for p in passages], dim=0)
    out: List[dict] = []
    for size in sizes:
        if size > keys.shape[0]:
            continue
        vals: List[float] = []
        for s in range(n_seeds):
            g = torch.Generator().manual_seed(1000 + s)
            sub = keys[torch.randperm(keys.shape[0], generator=g)[:size]]
            vals.append(calibrate_threshold_from_keys(
                sub, quantile=quantile, center=False, seed=s))
        out.append({
            "n": size,
            "mean": statistics.mean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        })
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _bar(title: str) -> None:
    print("-" * 72)
    print(f" {title}")
    print("-" * 72)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="number of passages")
    args = ap.parse_args()

    print("=" * 72)
    print(" M1 memory stack -- REAL WikiText-2 retrieval validation (CPU)")
    print("=" * 72)

    # Load corpus first; an unavailable dataset is an honest skip.
    try:
        t0 = time.time()
        passages = load_wikitext_passages(args.n)
        load_t = time.time() - t0
    except Exception as e:  # noqa: BLE001
        print(f"\n SKIP: could not load cached WikiText-2 ({type(e).__name__}: {e}).")
        return 0
    if len(passages) < 20:
        print(f"\n SKIP: only assembled {len(passages)} passages; need >= 20.")
        return 0
    print(f" corpus  : {len(passages)} real WikiText-2 passages "
          f"(assembled in {load_t:.1f}s)")

    enc = SentenceEncoder()
    try:
        t0 = time.time()
        dim = enc.dim
        enc_t = time.time() - t0
    except Exception as e:  # noqa: BLE001
        print(f"\n SKIP: could not load the e5 encoder ({type(e).__name__}: {e}).")
        return 0
    print(f" encoder : {enc.model_id} (dim={dim}), loaded in {enc_t:.1f}s")
    print()

    # ---- Probe A ----------------------------------------------------------
    _bar("PROBE A -- flat retrieval quality at scale (held-out sentence -> passage)")
    t0 = time.time()
    plain = run_retrieval_at_scale(enc, dim, passages, center=False)
    cen = run_retrieval_at_scale(enc, dim, passages, center=True)
    probe_t = time.time() - t0
    for tag, r in (("center=False", plain), ("center=True ", cen)):
        print(f"   {tag}  R@1={r['recall@1']:.3f}  R@5={r['recall@5']:.3f}  "
              f"R@10={r['recall@10']:.3f}  MRR={r['mrr']:.3f}")
    best = max((plain, cen), key=lambda r: r["mrr"])
    rand1 = 1.0 / plain["n"]
    print(f"   (random-guess R@1 baseline at N={plain['n']}: {rand1:.4f}; "
          f"encoded both passes in {probe_t:.1f}s)")
    print()
    # Judge on R@5 + MRR, not R@1 alone: the query is a SINGLE held-out middle
    # sentence (a deliberately weak, partial query), so R@1 understates the store.
    # The honest "好用" question is whether the right passage is reliably in the
    # shortlist a real system would rerank/read -- that's R@5 / MRR.
    if best["recall@5"] >= 0.8 and best["mrr"] >= 0.6:
        grade = ("GOOD -- a single partial sentence pulls the right passage into "
                 "the top-5 ~%.0f%% of the time among hundreds (R@1 is lower only "
                 "because one sentence is a weak query)." % (best["recall@5"] * 100))
    elif best["recall@5"] >= 0.6:
        grade = "MODERATE -- right passage often in top-5 but not reliably."
    else:
        grade = "WEAK -- the store does not reliably shortlist the right passage."
    print(f"   verdict (vs random {rand1:.4f}): R@1={best['recall@1']:.3f} "
          f"R@5={best['recall@5']:.3f} MRR={best['mrr']:.3f} -> {grade}")
    print()

    # ---- Probe B ----------------------------------------------------------
    _bar("PROBE B -- adaptive link-threshold stability vs corpus size")
    stab = run_threshold_stability(enc, passages)
    print("     N     threshold(mean)   std")
    for s in stab:
        print(f"   {s['n']:4d}        {s['mean']:+.3f}        {s['std']:.3f}")
    print()
    if len(stab) >= 2:
        first_std, last_std = stab[0]["std"], stab[-1]["std"]
        shrunk = last_std <= first_std
        print(f"   verdict: std {first_std:.3f} (N={stab[0]['n']}) -> "
              f"{last_std:.3f} (N={stab[-1]['n']}). "
              f"{'STABILISES with scale -- the data-derived cut is trustworthy once the corpus is non-trivial.' if shrunk else 'Does NOT shrink -- adaptive cut stays noisy on this corpus; prefer a fixed threshold.'}")
    print()
    print("=" * 72)
    print(" NOTE: retrieval effectiveness on REAL text. Not end-task QA accuracy")
    print(" (that needs a labelled set + generator). Closes the scale gap the")
    print(" curated probe left open.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
