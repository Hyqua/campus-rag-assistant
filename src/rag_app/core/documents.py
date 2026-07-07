from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
UNIT_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]|[^\s]", re.UNICODE)


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
    """Split text into overlapping word chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    units = list(UNIT_PATTERN.finditer(text))
    if not units:
        return []

    chunks: list[DocumentChunk] = []
    step = chunk_size - overlap
    for index, start in enumerate(range(0, len(units), step)):
        end = min(start + chunk_size, len(units))
        start_char = units[start].start()
        end_char = units[end - 1].end()
        chunks.append(
            DocumentChunk(
                chunk_id=f"{source}::{index}",
                source=source,
                text=text[start_char:end_char].strip(),
                start_word=start,
                end_word=end,
            )
        )
        if end == len(units):
            break
    return chunks


def chunks_from_paths(paths: Iterable[Path], chunk_size: int = 180, overlap: int = 35) -> list[DocumentChunk]:
    all_chunks: list[DocumentChunk] = []
    for path in paths:
        text = load_document_text(path)
        all_chunks.extend(chunk_text(text, source=path.name, chunk_size=chunk_size, overlap=overlap))
    return all_chunks
