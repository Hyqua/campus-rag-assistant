from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from rag_app.core.documents import DocumentChunk


TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in TOKEN_PATTERN.findall(text):
        token = raw_token.lower()
        tokens.append(token)
        cjk_chars = [char for char in token if "\u4e00" <= char <= "\u9fff"]
        if len(cjk_chars) == 1:
            tokens.extend(cjk_chars)
        elif cjk_chars:
            tokens.extend("".join(cjk_chars[index : index + 2]) for index in range(len(cjk_chars) - 1))
    return tokens


class LocalSearchIndex:
    """Small dependency-free retrieval baseline.

    It uses token overlap with cosine normalization. This is not a replacement for
    embeddings, but it gives the project a runnable baseline before Chroma/FAISS.
    """

    def __init__(self) -> None:
        self._chunks: list[DocumentChunk] = []
        self._vectors: list[Counter[str]] = []

    @property
    def chunks(self) -> list[DocumentChunk]:
        return list(self._chunks)

    def add(self, chunks: list[DocumentChunk]) -> None:
        for chunk in chunks:
            self._chunks.append(chunk)
            self._vectors.append(Counter(tokenize(chunk.text)))

    def remove_source(self, source: str) -> int:
        kept_chunks: list[DocumentChunk] = []
        kept_vectors: list[Counter[str]] = []
        removed = 0

        for chunk, vector in zip(self._chunks, self._vectors):
            if chunk.source == source:
                removed += 1
                continue
            kept_chunks.append(chunk)
            kept_vectors.append(vector)

        self._chunks = kept_chunks
        self._vectors = kept_vectors
        return removed

    def clear(self) -> None:
        self._chunks = []
        self._vectors = []

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_vector = Counter(tokenize(query))
        if not query_vector:
            return []

        scored: list[tuple[float, DocumentChunk]] = []
        query_norm = _norm(query_vector)
        for chunk, vector in zip(self._chunks, self._vectors):
            score = _cosine(query_vector, vector, query_norm=query_norm)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "source": chunk.source,
                "chunk_id": chunk.chunk_id,
                "score": round(score, 4),
                "text": chunk.text,
            }
            for score, chunk in scored[:top_k]
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(chunk) for chunk in self._chunks]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LocalSearchIndex":
        index = cls()
        if not path.exists():
            return index

        payload = json.loads(path.read_text(encoding="utf-8"))
        chunks = [DocumentChunk(**item) for item in payload]
        index.add(chunks)
        return index


def _norm(vector: Counter[str]) -> float:
    return math.sqrt(sum(value * value for value in vector.values()))


def _cosine(query_vector: Counter[str], doc_vector: Counter[str], query_norm: float | None = None) -> float:
    if not query_vector or not doc_vector:
        return 0.0
    query_norm = query_norm if query_norm is not None else _norm(query_vector)
    doc_norm = _norm(doc_vector)
    if query_norm == 0 or doc_norm == 0:
        return 0.0
    dot = sum(query_vector[token] * doc_vector.get(token, 0) for token in query_vector)
    return dot / (query_norm * doc_norm)
