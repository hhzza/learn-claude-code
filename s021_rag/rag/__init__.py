"""
RAG knowledge base module for s021.

Pipeline:
    loader.load_files() → [Document]
    chunker.chunk_documents() → [Chunk]
    embedder + vector_store + bm25 → retrievable index
    retriever.search() → top-K Chunks

Shared types live here so loader and chunker don't circularly import.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    """A loaded file, ready for chunking."""
    path: Path              # absolute path to the source file
    format: str             # "markdown" | "python" | "text" | "json" | "yaml"
    text: str               # extracted plain text
    metadata: dict = field(default_factory=dict)  # format-specific extras


@dataclass
class Chunk:
    """A single retrievable unit. Stored in both BM25 and vector indices."""
    id: str                 # "docs/auth.md#c3"
    text: str               # chunk content
    source: Path            # source file path
    chunk_index: int        # position within the source document (0-based)
    start_line: int         # 1-based line number where chunk starts
    end_line: int           # 1-based line number where chunk ends
    heading: str = ""       # nearest markdown heading, if any
    metadata: dict = field(default_factory=dict)
