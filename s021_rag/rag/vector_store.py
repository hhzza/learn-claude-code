"""
Vector store backed by numpy arrays + JSON metadata.

For the current scale (hundreds to low-thousands of chunks) this is simpler
and more reliable than Milvus Lite. Swap to Milvus later by changing only this
file — the public API (add / search / search_by_text / count) stays the same.

Storage layout:
    <uri>              ← JSON file: [{id, text, source, heading, chunk_index}, ...]
    <uri>.vectors.npy  ← (N, 1024) float32 ndarray
"""

import json
from pathlib import Path

import numpy as np

from . import Chunk

DIMENSION = 1024


class VectorStore:
    """Numpy-backed vector store. API-compatible with a future Milvus backend."""

    def __init__(self, uri: str = "./vectors.json"):
        self.uri = Path(uri)
        self._vectors_path = self.uri.with_suffix(self.uri.suffix + ".vectors.npy")

        self._metadata: list[dict] = []
        self._vectors: np.ndarray | None = None  # (N, 1024) float32, lazy-loaded

        self._load()

    # ── public API ──

    def add(self, ids: list[str], texts: list[str],
            vectors: list[list[float]],
            sources: list[str] | None = None,
            headings: list[str] | None = None,
            chunk_indices: list[int] | None = None) -> int:
        """Insert vectors + metadata. Returns number of rows inserted."""
        n = len(ids)
        if n == 0:
            return 0

        sources = sources or [""] * n
        headings = headings or [""] * n
        chunk_indices = chunk_indices or list(range(n))

        for i in range(n):
            self._metadata.append({
                "id": ids[i],
                "text": texts[i],
                "source": sources[i],
                "heading": headings[i],
                "chunk_index": chunk_indices[i],
            })

        new_vecs = np.array(vectors, dtype=np.float32)
        if self._vectors is None:
            self._vectors = new_vecs
        else:
            self._vectors = np.concatenate([self._vectors, new_vecs], axis=0)

        self._save()
        return n

    def add_chunks(self, chunks: list[Chunk], embedder) -> int:
        """Embed and insert a list of Chunk objects."""
        if not chunks:
            return 0

        print(f"Embedding {len(chunks)} chunks ...")
        vectors = embedder.embed_chunks(chunks)

        return self.add(
            ids=[c.id for c in chunks],
            texts=[c.text for c in chunks],
            vectors=vectors.tolist(),
            sources=[str(c.source) for c in chunks],
            headings=[c.heading for c in chunks],
            chunk_indices=[c.chunk_index for c in chunks],
        )

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[dict]:
        """Cosine similarity search. Returns top-k results with metadata.

        Each result: {id, text, source, heading, chunk_index, score}
        """
        if self._vectors is None or len(self._vectors) == 0:
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)

        # Normalise for cosine similarity
        q_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-10)
        v_norm = self._vectors / (np.linalg.norm(self._vectors, axis=1, keepdims=True) + 1e-10)

        scores = np.dot(v_norm, q_norm.T).flatten()

        if k >= len(scores):
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            meta = self._metadata[int(idx)]
            results.append({
                "id": meta["id"],
                "text": meta["text"],
                "source": meta["source"],
                "heading": meta["heading"],
                "chunk_index": meta["chunk_index"],
                "score": float(scores[idx]),
            })

        return results

    def search_by_text(self, query: str, embedder, k: int = 5) -> list[dict]:
        """Embed a query string and search."""
        vec = embedder.embed_query(query)
        return self.search(vec, k=k)

    def count(self) -> int:
        return len(self._metadata)

    def drop(self):
        """Delete all data."""
        self._metadata = []
        self._vectors = None
        if self.uri.exists():
            self.uri.unlink()
        if self._vectors_path.exists():
            self._vectors_path.unlink()

    # ── persistence ──

    def _save(self):
        self.uri.write_text(json.dumps(self._metadata, ensure_ascii=False), encoding="utf-8")
        if self._vectors is not None:
            np.save(str(self._vectors_path), self._vectors)

    def _load(self):
        if not self.uri.exists():
            return
        self._metadata = json.loads(self.uri.read_text(encoding="utf-8"))
        if self._vectors_path.exists():
            self._vectors = np.load(str(self._vectors_path))
