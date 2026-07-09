"""
Document loader: read files, detect format, extract clean text.

PDF parsing is powered by Docling — IBM's open-source document understanding
toolkit. It preserves reading order, heading hierarchy, tables, and formulas.

Supported formats:
    .pdf         → Docling → structured markdown + sections metadata
    .md          → markdown (preserved as-is)
    .py          → Python source (AST-aware metadata)
    .txt         → plain text
    .json/.yaml  → key-value flattening for searchability

Usage:
    docs = load_directory(Path("papers/"), patterns=["*.pdf"])
    for doc in docs:
        print(doc.path, len(doc.text), doc.metadata.get("sections_count"))
"""

import json
import logging
from pathlib import Path

import yaml

from . import Document

logger = logging.getLogger(__name__)

# ── format map ──

EXT_MAP = {
    ".pdf": "pdf",
    ".md": "markdown",
    ".markdown": "markdown",
    ".py": "python",
    ".txt": "text",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
}

SKIP_PATTERNS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".idea", ".vscode", ".claude", "dist", "build",
}

SKIP_FILES = {
    "MEMORY.md", "package-lock.json", "yarn.lock",
    "poetry.lock", "Cargo.lock",
}


# ═══════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════

def load_file(path: Path,
              do_ocr: bool = False,
              do_table_structure: bool = True) -> Document | None:
    """Load a single file. PDFs are parsed via Docling; other formats as text.

    Args:
        path: Path to the file.
        do_ocr: (PDF only) Enable OCR for scanned documents. Slow.
        do_table_structure: (PDF only) Extract table structures.
    """
    if not path.is_file():
        return None
    if path.name in SKIP_FILES:
        return None

    ext = path.suffix.lower()
    fmt = EXT_MAP.get(ext, "text")

    # ── PDF path: Docling ──
    if fmt == "pdf":
        return _load_pdf(path, do_ocr=do_ocr, do_table_structure=do_table_structure)

    # ── text formats ──
    try:
        raw = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError):
        return None

    if len(raw.strip()) < 20:
        return None

    text, metadata = _dispatch_extract(raw, fmt, path)

    if not text.strip():
        return None

    return Document(path=path.resolve(), format=fmt, text=text, metadata=metadata)


def load_directory(root: Path,
                   patterns: list[str] | None = None,
                   recursive: bool = True,
                   do_ocr: bool = False) -> list[Document]:
    """Load all documents under a directory."""
    if patterns is None:
        patterns = ["*.pdf", "*.md", "*.py", "*.txt", "*.json", "*.yaml", "*.yml"]

    documents = []
    seen = set()

    for pattern in patterns:
        glob_fn = root.rglob if recursive else root.glob
        for path in glob_fn(pattern):
            if set(path.parts) & SKIP_PATTERNS:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            doc = load_file(path, do_ocr=do_ocr)
            if doc:
                documents.append(doc)

    return documents


# ═══════════════════════════════════════════════════════════════
#  PDF extraction — Docling
# ═══════════════════════════════════════════════════════════════

def _load_pdf(path: Path, do_ocr: bool = False,
              do_table_structure: bool = True) -> Document | None:
    """Parse a PDF with Docling and return a Document with structured metadata.

    Results are cached as JSON alongside the PDF: paper.pdf → paper.pdf.cache.json.
    Cache is invalidated when the PDF's mtime changes.
    """
    cache_path = path.with_suffix(path.suffix + ".cache.json")

    # ── Check cache first ──
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("mtime") == path.stat().st_mtime:
                logger.info(f"Using cached parse for {path.name}")
                return Document(
                    path=path.resolve(),
                    format="pdf",
                    text=cached["text"],
                    metadata=cached["metadata"],
                )
        except (json.JSONDecodeError, KeyError):
            pass

    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError:
        logger.error("Docling not installed. Run: pip install docling")
        return None

    # Validate
    if path.stat().st_size == 0:
        logger.warning(f"Empty file: {path}")
        return None

    with open(path, "rb") as f:
        if not f.read(8).startswith(b"%PDF-"):
            logger.warning(f"Not a valid PDF: {path}")
            return None

    pipeline_options = PdfPipelineOptions(
        do_table_structure=do_table_structure,
        do_ocr=do_ocr,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    try:
        result = converter.convert(str(path))
    except Exception as e:
        logger.error(f"Docling conversion failed for {path}: {e}")
        return None

    doc = result.document

    # ── Extract structured sections ──
    # doc.texts preserves reading order with labels: title, section_header,
    # paragraph, table, formula, list_item, etc.
    sections: list[dict] = []
    current = {"title": "", "content": [], "level": ""}
    tables: list[str] = []

    for item in doc.texts:
        label = getattr(item, "label", "")
        text = getattr(item, "text", "")

        if not text:
            continue

        if label == "table":
            tables.append(text)

        # New section boundary
        if label in ("title", "section_header", "section-header"):
            # Flush previous section
            if current["content"] or current["title"]:
                sections.append({
                    "title": current["title"],
                    "content": "\n".join(current["content"]).strip(),
                    "level": current["level"],
                })
            # Start new section
            current = {"title": text.strip(), "content": [], "level": label}
        else:
            current["content"].append(text)

    # Flush final section
    if current["content"] or current["title"]:
        sections.append({
            "title": current["title"],
            "content": "\n".join(current["content"]).strip(),
            "level": current["level"],
        })

    # ── Full markdown for BM25 ──
    try:
        markdown_text = doc.export_to_markdown()
    except Exception:
        markdown_text = doc.export_to_text()

    if not markdown_text or len(markdown_text.strip()) < 20:
        return None

    # ── Detect language ──
    lang = _detect_language(markdown_text[:2000])

    metadata = {
        "source": "docling",
        "sections_count": len(sections),
        "tables_count": len(tables),
        "language": lang,
        "sections": sections,
        "tables": tables,
    }

    # ── Save cache ──
    try:
        cache_path.write_text(json.dumps({
            "mtime": path.stat().st_mtime,
            "text": markdown_text,
            "metadata": metadata,
        }, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Cached parse to {cache_path.name}")
    except Exception as e:
        logger.warning(f"Failed to cache: {e}")

    return Document(
        path=path.resolve(),
        format="pdf",
        text=markdown_text,
        metadata=metadata,
    )


def _detect_language(text: str) -> str:
    """Quick heuristic: if >15% of chars are CJK, label it 'zh'."""
    cjk = sum(1 for ch in text if '一' <= ch <= '鿿')
    return "zh" if cjk > len(text) * 0.15 else "en"


# ═══════════════════════════════════════════════════════════════
#  Text-format extractors (unchanged from original)
# ═══════════════════════════════════════════════════════════════

def _dispatch_extract(raw: str, fmt: str, path: Path) -> tuple[str, dict]:
    if fmt == "markdown":
        return _extract_markdown(raw)
    elif fmt == "python":
        return _extract_python(raw, path)
    elif fmt == "json":
        return _extract_json(raw)
    elif fmt == "yaml":
        return _extract_yaml(raw)
    else:
        return _extract_plain(raw)


def _extract_markdown(raw: str) -> tuple[str, dict]:
    import re
    headings = re.findall(r'^(#{1,6})\s+(.+)$', raw, re.MULTILINE)
    return raw, {"headings": len(headings), "lines": raw.count("\n") + 1}


def _extract_python(raw: str, path: Path) -> tuple[str, dict]:
    import re
    defs = re.findall(r'^(?:async\s+)?def\s+(\w+)|^class\s+(\w+)', raw, re.MULTILINE)
    func_names = [m[0] for m in defs if m[0]]
    class_names = [m[1] for m in defs if m[1]]
    return raw, {
        "lines": len(raw.splitlines()),
        "functions": func_names,
        "classes": class_names,
    }


def _extract_json(raw: str) -> tuple[str, dict]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw, {"parse_error": True}
    lines = _flatten_dict(data)
    return "\n".join(f"{k}: {v}" for k, v in lines), {"keys": len(lines)}


def _extract_yaml(raw: str) -> tuple[str, dict]:
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw, {"parse_error": True}
    if data is None:
        return "", {"keys": 0}
    if isinstance(data, str):
        return data, {"keys": 1}
    lines = _flatten_dict(data)
    return "\n".join(f"{k}: {v}" for k, v in lines), {"keys": len(lines)}


def _extract_plain(raw: str) -> tuple[str, dict]:
    return raw, {"lines": raw.count("\n") + 1}


def _flatten_dict(obj, prefix: str = "") -> list[tuple[str, str]]:
    result = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                result.extend(_flatten_dict(v, full_key))
            else:
                result.append((full_key, str(v)))
    elif isinstance(obj, list):
        if len(obj) <= 10:
            values = ", ".join(str(v) for v in obj if not isinstance(v, (dict, list)))
            result.append((prefix, f"[{values}]"))
        else:
            result.append((prefix, f"[{len(obj)} items]"))
            for i, item in enumerate(obj[:5]):
                result.extend(_flatten_dict(item, f"{prefix}[{i}]"))
    elif obj is not None:
        result.append((prefix, str(obj)))
    return result
