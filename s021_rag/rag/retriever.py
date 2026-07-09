"""
Hybrid retriever: BM25 + vector search → RRF fusion → top-K results.

RRF (Reciprocal Rank Fusion) combines ranked lists from different retrievers
without needing to calibrate their scores. The formula:

    RRF(doc) = Σ 1 / (k + rank_i(doc))

Where k=60 is a dampening constant (larger k = more weight to lower-ranked items),
and rank_i is the position of the document in the i-th result list (1-indexed).

Why RRF over score normalisation:
    - BM25 scores: [0, ∞)  tf-idf values
    - Cosine similarity: [-1, 1]
    → Different scales, can't directly add. RRF only cares about ranks.

Usage:
    retriever = HybridRetriever(bm25_index, vector_store, embedder)
    results = retriever.search("query text", k=5)
    → [{id, text, heading, source, bm25_rank, vector_rank, rrf_score}, ...]
"""

from .bm25_index import BM25Index
from .vector_store import VectorStore

K = 60  # RRF dampening constant


class HybridRetriever:
    """BM25 + vector hybrid search with RRF fusion."""

    def __init__(self, bm25: BM25Index, vector_store: VectorStore, embedder):
        self.bm25 = bm25
        self.vector_store = vector_store
        self.embedder = embedder

    def search(self, query: str, k: int = 5,
               bm25_top: int = 20, vector_top: int = 20) -> list[dict]:
        """Run BM25 and vector search, fuse with RRF, return top-k.

        Args:
            query: The search query string.
            k: Number of final results to return.
            bm25_top: How many BM25 results to fetch before fusion.
            vector_top: How many vector results to fetch before fusion.
        """
        # ── Run both retrievers ──
        bm25_results = self.bm25.search(query, k=bm25_top)
        vec_results = self.vector_store.search_by_text(query, self.embedder, k=vector_top)

        # ── Compute RRF scores ──
        rrf_scores: dict[str, float] = {}

        for rank, result in enumerate(bm25_results):
            chunk_id = result["id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (K + rank + 1)

        for rank, result in enumerate(vec_results):
            chunk_id = result["id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (K + rank + 1)

        # ── Merge metadata and sort by RRF score ──
        # Build lookup from both result lists
        lookup: dict[str, dict] = {}
        for r in bm25_results:
            lookup[r["id"]] = r
        for r in vec_results:
            if r["id"] not in lookup:
                lookup[r["id"]] = r

        merged = []
        for chunk_id, rrf in rrf_scores.items():
            info = lookup.get(chunk_id, {})
            merged.append({
                "id": chunk_id,
                "text": info.get("text", ""),
                "heading": info.get("heading", ""),
                "source": info.get("source", ""),
                "rrf_score": round(rrf, 6),
            })

        merged.sort(key=lambda x: x["rrf_score"], reverse=True)
        return merged[:k]
