from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
UNIT_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]|[^\s]", re.UNICODE)
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")
SENTENCE_PATTERN = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?", re.UNICODE)


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    source: str
    text: str
    start_word: int
    end_word: int


def load_document_text(path: Path) -> str:
    """Load text from txt, markdown, or PDF files."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type: {suffix}. Supported: {supported}")

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF support requires pypdf. Install with: pip install pypdf") from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(page.strip() for page in pages if page.strip())


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 180,
    overlap: int = 35,
) -> list[DocumentChunk]:
    """Split text into readable chunks, preferring Markdown sections and paragraphs."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    raw_chunks = _markdown_paragraph_chunks(text, chunk_size=chunk_size, overlap=overlap)
    chunks: list[DocumentChunk] = []
    cursor = 0
    for index, chunk in enumerate(raw_chunks):
        unit_count = _count_units(chunk)
        chunks.append(
            DocumentChunk(
                chunk_id=f"{source}::{index}",
                source=source,
                text=chunk,
                start_word=cursor,
                end_word=cursor + unit_count,
            )
        )
        cursor += unit_count
    return chunks


def chunks_from_paths(paths: Iterable[Path], chunk_size: int = 180, overlap: int = 35) -> list[DocumentChunk]:
    all_chunks: list[DocumentChunk] = []
    for path in paths:
        text = load_document_text(path)
        all_chunks.extend(chunk_text(text, source=path.name, chunk_size=chunk_size, overlap=overlap))
    return all_chunks


def _markdown_paragraph_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    heading_stack: list[tuple[int, str]] = []
    chunks: list[str] = []

    for block in _iter_blocks(text):
        heading_match = HEADING_PATTERN.match(block)
        if heading_match:
            level = len(heading_match.group(1))
            heading_stack = [(old_level, heading) for old_level, heading in heading_stack if old_level < level]
            heading_stack.append((level, block))
            continue

        context = _heading_context(heading_stack)
        chunks.extend(_split_paragraph(block, context=context, chunk_size=chunk_size, overlap=overlap))

    return chunks


def _iter_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        cleaned = "\n".join(line.rstrip() for line in block.splitlines() if line.strip()).strip()
        if cleaned:
            blocks.append(cleaned)
    return blocks


def _heading_context(heading_stack: list[tuple[int, str]]) -> str:
    if not heading_stack:
        return ""
    return "\n".join(heading for _, heading in heading_stack[-2:])


def _split_paragraph(paragraph: str, context: str, chunk_size: int, overlap: int) -> list[str]:
    if not paragraph.strip():
        return []

    context_units = _count_units(context)
    body_limit = max(1, chunk_size - context_units)
    sentences = _split_sentences(paragraph)
    chunks: list[str] = []
    current: list[str] = []

    for sentence in sentences:
        if _count_units(sentence) > body_limit:
            if current:
                chunks.append(_join_context(context, current))
                current = []
            chunks.extend(_join_context(context, [piece]) for piece in _split_long_text(sentence, body_limit, overlap))
            continue

        candidate = current + [sentence]
        if current and _count_units("".join(candidate)) > body_limit:
            chunks.append(_join_context(context, current))
            current = _sentence_overlap(current, overlap)
            candidate = current + [sentence]

        current = candidate

    if current:
        chunks.append(_join_context(context, current))

    return chunks


def _split_sentences(text: str) -> list[str]:
    sentences = [match.group(0).strip() for match in SENTENCE_PATTERN.finditer(text) if match.group(0).strip()]
    return sentences or [text.strip()]


def _sentence_overlap(sentences: list[str], overlap: int) -> list[str]:
    if overlap == 0 or not sentences:
        return []

    kept: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        units = _count_units(sentence)
        if kept and total + units > overlap:
            break
        kept.insert(0, sentence)
        total += units
    return kept


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    units = list(UNIT_PATTERN.finditer(text))
    if not units:
        return []

    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(units), step):
        end = min(start + chunk_size, len(units))
        start_char = units[start].start()
        end_char = units[end - 1].end()
        chunks.append(text[start_char:end_char].strip())
        if end == len(units):
            break
    return chunks


def _join_context(context: str, sentences: list[str]) -> str:
    body = "".join(sentences).strip()
    if context:
        return f"{context}\n\n{body}"
    return body


def _count_units(text: str) -> int:
    return len(UNIT_PATTERN.findall(text))
