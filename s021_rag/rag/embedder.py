"""
Embedding wrapper for Alibaba Cloud Bailian (百炼) text-embedding-v4.

Uses OpenAI-compatible mode — no dashscope SDK required, just `pip install openai`.

Usage:
    embedder = Embedder(api_key="sk-xxx")
    vec = embedder.embed_query("什么是知识蒸馏")          # (1024,) ndarray
    vecs = embedder.embed(["文本1", "文本2", ...])        # (N, 1024) ndarray
    vecs = embedder.embed_chunks(chunks)                  # embed Chunk objects
"""

import time
import numpy as np
from openai import OpenAI

from . import Chunk


class Embedder:
    """Bailian text-embedding-v4 via OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-v4",
        dimension: int = 1024,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        max_retries: int = 3,
    ):
        self.model = model
        self.dimension = dimension
        self.max_retries = max_retries

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    # ── public API ──

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of text strings. Auto-batches at 10 (API limit).

        Returns: (N, dimension) float32 ndarray
        """
        if not texts:
            raise ValueError("texts must not be empty")

        all_vectors = []

        # Bailian allows up to 10 inputs per request
        for i in range(0, len(texts), 10):
            batch = texts[i:i + 10]
            vectors = self._embed_batch(batch)
            all_vectors.append(vectors)

        return np.concatenate(all_vectors, axis=0).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string. Returns (dimension,) float32 ndarray."""
        vecs = self.embed([text])
        return vecs[0]

    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        """Embed a list of Chunk objects, using chunk.text as input."""
        texts = [c.text for c in chunks]
        return self.embed(texts)

    # ── internal ──

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Call the API for one batch (≤10 texts). Retries on failure."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                resp = self._client.embeddings.create(
                    model=self.model,
                    input=texts,
                    dimensions=self.dimension,
                    encoding_format="float",
                )
                return np.array(
                    [d.embedding for d in resp.data],
                    dtype=np.float32,
                )

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = (2 ** attempt) * 0.5
                    time.sleep(delay)

        raise RuntimeError(
            f"Embedding failed after {self.max_retries} retries: {last_error}"
        )
