"""
BM25 sparse retrieval index — pure Python, zero dependencies.

BM25 is a bag-of-words ranking function. It excels at exact keyword matches
(API names, error codes, technical terms) where vector search can be fuzzy.

The algorithm:
    BM25(q, d) = Σ IDF(t) × ───────────────────────────────
                              k₁·(1 - b + b·|d|/avgdl) + TF(t,d)

    IDF(t) = log((N - n(t) + 0.5) / (n(t) + 0.5) + 1)

Where:
    N      = total document count
    n(t)   = number of documents containing term t
    TF(t,d)= term frequency of t in document d
    |d|    = document length (in tokens)
    avgdl  = average document length across all docs
    k₁=1.5 = TF saturation (prevents "tf=100 scores 100x tf=1")
    b=0.75 = length normalisation (prevents long-doc bias)

Usage:
    idx = BM25Index()
    idx.add(["chunk one text", "chunk two text", ...])
    results = idx.search("query text", k=5)  → [(chunk_index, score), ...]
"""

import math
import re
from collections import Counter

from . import Chunk

# ── tunables ──

K1 = 1.5    # TF saturation steepness
B = 0.75    # document length normalisation strength

# Simple tokenizer: lowercase + split on non-alphanumeric
# For CJK characters, each character is a separate token (bigram fallback)
_TOKEN_RE = re.compile(r'[a-zA-Z0-9]+|[一-鿿]')


class BM25Index:
    """Sparse keyword index with BM25 scoring."""

    def __init__(self, k1: float = K1, b: float = B):
        self.k1 = k1
        self.b = b

        self._docs: list[list[str]] = []     # tokenised documents
        self._doc_ids: list[str] = []         # chunk IDs
        self._doc_texts: list[str] = []       # original chunk texts
        self._doc_headings: list[str] = []    # chunk headings
        self._doc_sources: list[str] = []     # source file paths

        self._N = 0                           # total document count
        self._avgdl = 0.0                     # average token count per doc
        self._doc_freq: dict[str, int] = {}   # term → number of docs containing it
        self._doc_lengths: list[int] = []     # token count per doc
        self._tf_cache: list[Counter] = []    # term frequencies per doc

    # ── public API ──

    def add(self, texts: list[str], ids: list[str] | None = None,
            headings: list[str] | None = None,
            sources: list[str] | None = None):
        """Index a batch of plain-text documents."""
        tokenised = [_tokenize(t) for t in texts]
        self._docs.extend(tokenised)
        self._doc_ids.extend(ids or [""] * len(texts))
        self._doc_texts.extend(texts)
        self._doc_headings.extend(headings or [""] * len(texts))
        self._doc_sources.extend(sources or [""] * len(texts))

        # Rebuild stats (incremental update is more complex; rebuild is fast enough)
        self._rebuild_stats()

    def add_chunks(self, chunks: list[Chunk]):
        """Index a list of Chunk objects."""
        self.add(
            texts=[c.text for c in chunks],
            ids=[c.id for c in chunks],
            headings=[c.heading for c in chunks],
            sources=[str(c.source) for c in chunks],
        )

    def search(self, query: str, k: int = 10) -> list[dict]:
        """Search for top-k documents matching query.

        Returns list of {id, text, heading, score} sorted by BM25 score desc.
        """
        if self._N == 0:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = []
        for i in range(self._N):
            score = self._score_one(query_tokens, i)
            if score > 0:
                scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc_idx, score in scores[:k]:
            results.append({
                "id": self._doc_ids[doc_idx],
                "text": self._doc_texts[doc_idx],
                "heading": self._doc_headings[doc_idx],
                "source": self._doc_sources[doc_idx] if doc_idx < len(self._doc_sources) else "",
                "score": round(score, 4),
            })
        return results

    def count(self) -> int:
        return self._N

    # ── internal ──

    def _rebuild_stats(self):
        self._N = len(self._docs)
        if self._N == 0:
            return

        self._doc_lengths = [len(doc) for doc in self._docs]
        self._avgdl = sum(self._doc_lengths) / self._N

        # Document frequency: count docs containing each term
        self._doc_freq = {}
        for doc in self._docs:
            for term in set(doc):
                self._doc_freq[term] = self._doc_freq.get(term, 0) + 1

        # Term frequency cache
        self._tf_cache = [Counter(doc) for doc in self._docs]

    def _idf(self, term: str) -> float:
        n = self._doc_freq.get(term, 0)
        if n == 0:
            return 0.0
        return math.log((self._N - n + 0.5) / (n + 0.5) + 1)

    def _score_one(self, query_tokens: list[str], doc_idx: int) -> float:
        """BM25 score for a single query-document pair."""
        score = 0.0
        dl = self._doc_lengths[doc_idx]
        tf_count = self._tf_cache[doc_idx]

        for term in set(query_tokens):
            tf = tf_count.get(term, 0)
            if tf == 0:
                continue

            idf = self._idf(term)

            # BM25 core formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
            score += idf * numerator / denominator

        return score


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase word tokens.

    English: "Knowledge Distillation" → ["knowledge", "distillation"]
    CJK: each character is a separate token (character bigrams would
          be better for longer queries, but single-char is simpler).
    """
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if len(t) > 1 or t.isascii() or '一' <= t <= '鿿']
