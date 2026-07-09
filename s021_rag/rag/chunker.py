"""
Document chunker: split loaded Documents into retrievable Chunks.

Strategy per format:
    pdf        → Docling-structured: use pre-parsed sections as chunk boundaries
    markdown   → split by ## / ### headings; merge small / split large
    python     → AST: group by top-level function / class
    json/yaml  → each top-level key is one chunk
    text       → paragraph-aware fixed-window with overlap

Target chunk size: ~500 tokens (~2000 chars mixed CN/EN, ~375 English words).
"""

import ast
import re
from pathlib import Path

from . import Chunk, Document

# ── tunables ──

TARGET_CHARS = 2000
MIN_CHARS = 600
MAX_CHARS = 4000
OVERLAP_CHARS = 400


# ═══════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════

def chunk_documents(documents: list[Document]) -> list[Chunk]:
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_one(doc))
    return all_chunks


def chunk_one(doc: Document) -> list[Chunk]:
    if doc.format == "pdf":
        return _chunk_pdf(doc)
    elif doc.format == "markdown":
        return _chunk_markdown(doc)
    elif doc.format == "python":
        return _chunk_python(doc)
    elif doc.format in ("json", "yaml"):
        return _chunk_config(doc)
    else:
        return _chunk_text(doc)


# ═══════════════════════════════════════════════════════════════
#  PDF chunker — Docling-structured sections
# ═══════════════════════════════════════════════════════════════

def _chunk_pdf(doc: Document) -> list[Chunk]:
    """Chunk a Docling-parsed PDF using its pre-parsed section structure.

    Docling already identified titles, section headers, paragraphs, and tables.
    We use that structure instead of brute-force regex, giving much cleaner
    chunk boundaries that respect the document's logical organization.
    """
    sections = doc.metadata.get("sections", [])

    if not sections:
        # Fallback: treat as markdown
        return _chunk_markdown(doc)

    # ── Build chunk candidates from sections ──
    candidates = []

    for i, sec in enumerate(sections):
        title = sec.get("title", "")
        content = sec.get("content", "")

        if not content and not title:
            continue

        # The full text is title + content
        full_text = f"## {title}\n\n{content}" if title else content

        candidates.append({
            "text": full_text,
            "heading": title,
        })

    if not candidates:
        return _chunk_markdown(doc)

    # ── Merge short adjacent sections ──
    merged = _merge_adjacent_pdf(candidates)

    # ── Split oversized sections at paragraph boundaries ──
    result = []
    for cand in merged:
        if len(cand["text"]) <= MAX_CHARS:
            result.append(cand)
        else:
            result.extend(_split_large_pdf_section(cand))

    # ── Build Chunk objects ──
    chunks = []
    base_name = doc.path.stem

    for i, data in enumerate(result):
        chunk_id = f"{base_name}#c{i}"
        chunks.append(Chunk(
            id=chunk_id,
            text=data["text"],
            source=doc.path,
            chunk_index=i,
            start_line=0,
            end_line=data["text"].count("\n"),
            heading=data.get("heading", ""),
            metadata={"format": "pdf", "source": doc.metadata.get("source", "")},
        ))

    return chunks


def _merge_adjacent_pdf(candidates: list[dict]) -> list[dict]:
    """Greedy forward merge: absorb short sections into the next one.

    PDF sections vary wildly in length — a one-line subsection followed by
    a 50-line paragraph should be one chunk, not two tiny ones.
    """
    if len(candidates) <= 1:
        return candidates

    merged = []
    i = 0
    while i < len(candidates):
        current = dict(candidates[i])
        if len(current["text"]) < MIN_CHARS and i + 1 < len(candidates):
            next_sec = candidates[i + 1]
            sep = "\n\n" if current["text"] and next_sec["text"] else ""
            current = {
                "text": current["text"] + sep + next_sec["text"],
                "heading": current["heading"] or next_sec["heading"],
            }
            i += 2
        else:
            i += 1
        merged.append(current)

    # Last section too short → merge into previous
    if len(merged) >= 2 and len(merged[-1]["text"]) < MIN_CHARS:
        last = merged.pop()
        prev = merged[-1]
        prev["text"] = prev["text"] + "\n\n" + last["text"]
        if not prev["heading"]:
            prev["heading"] = last["heading"]

    return merged


def _split_large_pdf_section(cand: dict) -> list[dict]:
    """Split an oversized section at paragraph boundaries."""
    paragraphs = cand["text"].split("\n\n")
    subs = []
    current_text = ""
    heading = cand.get("heading", "")

    for para in paragraphs:
        if len(current_text) + len(para) < TARGET_CHARS:
            current_text = (current_text + "\n\n" + para).strip() if current_text else para
        else:
            if current_text:
                subs.append({"text": current_text, "heading": heading})
            current_text = para

    if current_text:
        subs.append({"text": current_text, "heading": heading})

    return subs or [cand]


# ═══════════════════════════════════════════════════════════════
#  Markdown chunker — split by headings
# ═══════════════════════════════════════════════════════════════

HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)


def _chunk_markdown(doc: Document) -> list[Chunk]:
    text = doc.text
    headings: list[tuple[int, int, str, str]] = []

    for match in HEADING_RE.finditer(text):
        line_idx = text[:match.start()].count("\n")
        headings.append((line_idx, match.start(), match.group(1), match.group(2)))

    if not headings:
        return _chunk_text(doc)

    sections = []
    for i, (line_idx, char_pos, level, title) in enumerate(headings):
        if level not in ("##", "###"):
            continue
        next_pos = headings[i + 1][1] if i + 1 < len(headings) else len(text)
        section_text = text[char_pos:next_pos].strip()
        sections.append({
            "text": section_text,
            "start_line": line_idx + 1,
            "heading": title,
            "level": level,
        })

    # Preamble
    if headings and headings[0][1] > 0:
        preamble = text[:headings[0][1]].strip()
        if len(preamble) > 10:
            sections.insert(0, {"text": preamble, "start_line": 1, "heading": "", "level": ""})

    if not sections:
        return _chunk_text(doc)

    merged = _merge_short_md_sections(sections)
    result = _split_large_md_sections(merged)
    line_offsets = _line_offsets(text)
    return _build_chunks(result, doc, line_offsets)


def _merge_short_md_sections(sections: list[dict]) -> list[dict]:
    if len(sections) <= 1:
        return sections
    merged = []
    i = 0
    while i < len(sections):
        current = dict(sections[i])
        if len(current["text"]) < MIN_CHARS and i + 1 < len(sections):
            next_s = sections[i + 1]
            current = {
                "text": current["text"] + "\n\n" + next_s["text"],
                "start_line": current["start_line"],
                "heading": current["heading"] or next_s["heading"],
                "level": current["level"] or next_s["level"],
            }
            i += 2
        else:
            i += 1
        merged.append(current)

    if len(merged) >= 2 and len(merged[-1]["text"]) < MIN_CHARS:
        last = merged.pop()
        merged[-1]["text"] = merged[-1]["text"] + "\n\n" + last["text"]

    return merged


def _split_large_md_sections(sections: list[dict]) -> list[dict]:
    result = []
    for sec in sections:
        if len(sec["text"]) <= MAX_CHARS:
            result.append(sec)
            continue
        sub_headings = list(re.finditer(r'^###\s+(.+)$', sec["text"], re.MULTILINE))
        if sub_headings:
            result.extend(_split_by_positions(sec, sub_headings, len(sec["text"])))
        else:
            result.extend(_split_by_paragraphs(sec))
    return result


def _split_by_positions(sec: dict, matches: list, text_end: int) -> list[dict]:
    subs = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else text_end
        sub_text = sec["text"][start:end].strip()
        if sub_text:
            subs.append({
                "text": sub_text,
                "start_line": sec["start_line"] + sec["text"][:start].count("\n"),
                "heading": (sec["heading"] + " / " + m.group(1)) if sec["heading"] else m.group(1),
                "level": "###",
            })
    return subs


def _split_by_paragraphs(sec: dict) -> list[dict]:
    paragraphs = sec["text"].split("\n\n")
    subs = []
    current_text = ""
    current_start = sec["start_line"]

    for para in paragraphs:
        if len(current_text) + len(para) < TARGET_CHARS:
            current_text = (current_text + "\n\n" + para).strip() if current_text else para
        else:
            if current_text:
                subs.append({
                    "text": current_text,
                    "start_line": current_start,
                    "heading": sec["heading"],
                    "level": sec["level"],
                })
                current_start += current_text.count("\n") + 2
            current_text = para

    if current_text:
        subs.append({
            "text": current_text,
            "start_line": current_start,
            "heading": sec["heading"],
            "level": sec["level"],
        })
    return subs


# ═══════════════════════════════════════════════════════════════
#  Python chunker — AST-based
# ═══════════════════════════════════════════════════════════════

def _chunk_python(doc: Document) -> list[Chunk]:
    try:
        return _chunk_python_ast(doc)
    except SyntaxError:
        return _chunk_python_regex(doc)


def _chunk_python_ast(doc: Document) -> list[Chunk]:
    tree = ast.parse(doc.text)
    lines = doc.text.splitlines()

    items = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            items.append({"type": "function", "name": node.name,
                          "start_line": node.lineno,
                          "end_line": node.end_lineno or node.lineno})
        elif isinstance(node, ast.ClassDef):
            items.append({"type": "class", "name": node.name,
                          "start_line": node.lineno,
                          "end_line": node.end_lineno or node.lineno})

    if not items:
        return _chunk_text(doc)

    first_def_line = items[0]["start_line"]
    head_text = "\n".join(lines[:first_def_line - 1]).strip()
    if head_text:
        head_text = _strip_comments_and_docstring(head_text)

    chunks_data = []

    if head_text and len(head_text) > 60:
        chunks_data.append({
            "text": head_text, "start_line": 1,
            "heading": f"module: {doc.path.stem}",
        })

    i = 0
    while i < len(items):
        item = items[i]
        item_text = "\n".join(lines[item["start_line"] - 1:item["end_line"]])
        item_text_clean = _strip_comments_and_docstring(item_text)
        heading = f"{item['type']}: {item['name']}"

        if len(item_text_clean) <= MAX_CHARS:
            # Pack adjacent small items
            group = [item]
            group_text = item_text
            j = i + 1
            while j < len(items):
                next_item = items[j]
                next_text = "\n".join(lines[next_item["start_line"] - 1:next_item["end_line"]])
                if len(group_text) + len(next_text) < TARGET_CHARS:
                    group.append(next_item)
                    group_text += "\n\n" + next_text
                    j += 1
                else:
                    break
            i = j
            group_text_clean = _strip_comments_and_docstring(group_text)
            if len(group) == 1:
                chunks_data.append({
                    "text": group_text_clean,
                    "start_line": item["start_line"],
                    "heading": heading,
                })
            else:
                names = ", ".join(f"{it['type'][:3]}:{it['name']}" for it in group)
                chunks_data.append({
                    "text": group_text_clean,
                    "start_line": group[0]["start_line"],
                    "heading": names,
                })
        else:
            subs = _split_class_methods(lines, item)
            chunks_data.extend(subs)
            i += 1

    line_offsets = _line_offsets(doc.text)
    return _build_chunks(chunks_data, doc, line_offsets, prefix=doc.path.stem)


def _chunk_python_regex(doc: Document) -> list[Chunk]:
    pattern = re.compile(r'^[ \t]*(?:async\s+)?(?:def|class)\s+(\w+)', re.MULTILINE)
    matches = list(pattern.finditer(doc.text))
    if not matches:
        return _chunk_text(doc)

    chunks_data = []
    for i, m in enumerate(matches):
        next_pos = matches[i + 1].start() if i + 1 < len(matches) else len(doc.text)
        section = doc.text[m.start():next_pos].strip()
        start_line = doc.text[:m.start()].count("\n") + 1
        chunks_data.append({"text": section, "start_line": start_line, "heading": m.group(1)})

    line_offsets = _line_offsets(doc.text)
    return _build_chunks(chunks_data, doc, line_offsets, prefix=doc.path.stem)


def _split_class_methods(lines: list[str], class_item: dict) -> list[dict]:
    class_lines = lines[class_item["start_line"] - 1:class_item["end_line"]]
    class_text = "\n".join(class_lines)
    method_re = re.compile(r'^[ \t]{1,8}(?:async\s+)?def\s+(\w+)', re.MULTILINE)
    method_matches = list(method_re.finditer(class_text))

    if not method_matches:
        return [{"text": _strip_comments_and_docstring(class_text),
                 "start_line": class_item["start_line"],
                 "heading": f"class: {class_item['name']}"}]

    # Find end of class docstring
    class_doc_end = 1
    for j, line in enumerate(class_lines[1:], 1):
        stripped = line.strip()
        if stripped.startswith('"""') or stripped.startswith("'''") or stripped.startswith("#"):
            class_doc_end = j + 1
        elif stripped and not stripped.startswith(("#", "@")):
            break
        else:
            class_doc_end = j + 1

    chunks_data = []
    current_text_lines = class_lines[:class_doc_end]
    current_start = class_item["start_line"]

    for mi, m in enumerate(method_matches):
        method_start_in_class = m.start()
        method_start_line = class_item["start_line"] + class_text[:method_start_in_class].count("\n")
        next_start = method_matches[mi + 1].start() if mi + 1 < len(method_matches) else len(class_text)
        method_text = class_text[method_start_in_class:next_start].strip()

        if len("\n".join(current_text_lines)) + len(method_text) < TARGET_CHARS:
            current_text_lines.extend(
                class_lines[method_start_line - class_item["start_line"]:
                            method_start_line - class_item["start_line"]
                            + method_text.count("\n") + 1])
        else:
            chunks_data.append({
                "text": _strip_comments_and_docstring("\n".join(current_text_lines)),
                "start_line": current_start,
                "heading": f"class: {class_item['name']}",
            })
            current_text_lines = [class_lines[0]]
            current_text_lines.extend(
                class_lines[method_start_line - class_item["start_line"]:
                            method_start_line - class_item["start_line"]
                            + method_text.count("\n") + 1])
            current_start = method_start_line

    if len(current_text_lines) > 1:
        chunks_data.append({
            "text": _strip_comments_and_docstring("\n".join(current_text_lines)),
            "start_line": current_start,
            "heading": f"class: {class_item['name']}",
        })

    return chunks_data


# ═══════════════════════════════════════════════════════════════
#  Config chunker
# ═══════════════════════════════════════════════════════════════

def _chunk_config(doc: Document) -> list[Chunk]:
    text = doc.text
    key_re = re.compile(r'^\w[\w-]*:', re.MULTILINE)
    matches = list(key_re.finditer(text))

    if not matches or len(matches) <= 1:
        return _build_chunks(
            [{"text": text, "start_line": 1, "heading": doc.path.stem}],
            doc, _line_offsets(text), prefix=doc.path.stem)

    chunks_data = []
    for i, m in enumerate(matches):
        next_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[m.start():next_pos].strip()
        start_line = text[:m.start()].count("\n") + 1
        key = m.group(0).rstrip(":")
        chunks_data.append({
            "text": section, "start_line": start_line,
            "heading": f"{doc.path.stem}/{key}",
        })

    return _build_chunks(chunks_data, doc, _line_offsets(text), prefix=doc.path.stem)


# ═══════════════════════════════════════════════════════════════
#  Plain-text chunker — paragraph-aware fixed window
# ═══════════════════════════════════════════════════════════════

def _chunk_text(doc: Document) -> list[Chunk]:
    text = doc.text
    paragraphs = text.split("\n\n")

    if len(paragraphs) <= 1 and len(text) > TARGET_CHARS:
        return _chunk_text_fixed_window(doc)

    chunks_data = []
    current_lines = []
    current_start = 1
    current_len = 0

    for para in paragraphs:
        para_clean = para.strip()
        if not para_clean:
            continue

        if current_len + len(para_clean) > MAX_CHARS and current_len >= MIN_CHARS:
            chunks_data.append({
                "text": "\n\n".join(current_lines),
                "start_line": current_start,
                "heading": "",
            })
            overlap_para = current_lines[-1] if current_lines else ""
            current_lines = [overlap_para] if overlap_para else []
            current_start += sum(p.count("\n") for p in current_lines) + 2 * max(len(current_lines) - 1, 0)
            current_len = len(overlap_para)

        current_lines.append(para_clean)
        current_len += len(para_clean)

    if current_lines:
        chunks_data.append({
            "text": "\n\n".join(current_lines),
            "start_line": current_start,
            "heading": "",
        })

    if not chunks_data:
        chunks_data = [{"text": text, "start_line": 1, "heading": ""}]

    return _build_chunks(chunks_data, doc, _line_offsets(text))


def _chunk_text_fixed_window(doc: Document) -> list[Chunk]:
    text = doc.text
    step = TARGET_CHARS - OVERLAP_CHARS
    chunks_data = []
    start = 0

    while start < len(text):
        end = min(start + TARGET_CHARS, len(text))
        if end < len(text):
            search_start = max(start, end - 200)
            for break_char in ("\n", "。", ". ", "! ", "? ", "；"):
                idx = text.rfind(break_char, search_start, end)
                if idx > search_start:
                    end = idx + 1
                    break

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks_data.append({
                "text": chunk_text,
                "start_line": text[:start].count("\n") + 1,
                "heading": "",
            })
        start += step

    return _build_chunks(chunks_data, doc, _line_offsets(text))


# ═══════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════

def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            offsets.append(i + 1)
    return offsets


def _build_chunks(chunks_data: list[dict], doc: Document,
                  line_offsets: list[int], prefix: str = "") -> list[Chunk]:
    result = []
    base_name = prefix or doc.path.stem

    for i, data in enumerate(chunks_data):
        text = data["text"]
        start_line = data.get("start_line", 1)
        end_line = start_line + text.count("\n")
        chunk_id = f"{base_name}#c{i}"
        heading = data.get("heading", "")

        if not heading:
            first_line = text.split("\n")[0].strip()
            if len(first_line) <= 80:
                heading = first_line

        result.append(Chunk(
            id=chunk_id, text=text, source=doc.path,
            chunk_index=i, start_line=start_line, end_line=end_line,
            heading=heading,
            metadata={"format": doc.format},
        ))

    return result


def _strip_comments_and_docstring(text: str) -> str:
    lines = []
    in_docstring = False
    docstring_quote = ""

    for line in text.splitlines():
        stripped = line.strip()
        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_docstring = True
                docstring_quote = stripped[:3]
                if stripped.endswith(docstring_quote) and len(stripped) >= 6:
                    in_docstring = False
                continue
        else:
            if docstring_quote in stripped:
                in_docstring = False
            continue

        comment_pos = line.find("  #")
        if comment_pos > 0:
            line = line[:comment_pos]
        if line.strip():
            lines.append(line)

    return "\n".join(lines)
