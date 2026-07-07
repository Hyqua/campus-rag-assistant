from __future__ import annotations

import shutil
import time
from pathlib import Path
import re

from rag_app.core.documents import chunk_text, load_document_text
from rag_app.core.index import LocalSearchIndex, tokenize


ANSWER_SENTENCE_PATTERN = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?", re.UNICODE)


class KnowledgeBase:
    def __init__(self, data_dir: Path | str = ".rag_data") -> None:
        self.data_dir = Path(data_dir)
        self.upload_dir = self.data_dir / "uploads"
        self.index_path = self.data_dir / "index.json"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.index = LocalSearchIndex.load(self.index_path)

    def add_document(self, source_path: Path, chunk_size: int = 180, overlap: int = 35) -> dict:
        if not source_path.exists():
            raise FileNotFoundError(source_path)

        target = self.upload_dir / source_path.name
        if source_path.resolve() != target.resolve():
            shutil.copyfile(source_path, target)

        text = load_document_text(target)
        chunks = chunk_text(text, source=target.name, chunk_size=chunk_size, overlap=overlap)
        replaced_chunks = self.index.remove_source(target.name)
        self.index.add(chunks)
        self.index.save(self.index_path)
        return {
            "source": target.name,
            "chunks_added": len(chunks),
            "chunks_replaced": replaced_chunks,
            "total_chunks": len(self.index.chunks),
        }

    def ask(self, question: str, top_k: int = 2) -> dict:
        started_at = time.perf_counter()
        sources = self.index.search(question, top_k=top_k)
        answer = compose_answer(question, sources)
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        return {"answer": answer, "sources": sources, "latency_ms": latency_ms}

    def health(self) -> dict:
        return {"status": "ok", "chunks": len(self.index.chunks), "data_dir": str(self.data_dir)}

    def reset(self, clear_uploads: bool = False) -> dict:
        self.index.clear()
        if self.index_path.exists():
            self.index_path.unlink()
        removed_uploads = 0
        if clear_uploads and self.upload_dir.exists():
            for path in self.upload_dir.iterdir():
                if path.is_file():
                    path.unlink()
                    removed_uploads += 1
        return {"status": "reset", "chunks": 0, "uploads_removed": removed_uploads}


def compose_answer(question: str, sources: list[dict]) -> str:
    if not sources:
        return (
            "当前知识库没有检索到足够相关的内容。建议补充文档，或换一种更贴近文档原文的问法。"
        )

    relevant_sentences = _select_relevant_sentences(question, sources, limit=4)
    if relevant_sentences:
        answer_text = "".join(item["sentence"] for item in relevant_sentences)
    else:
        answer_text = _fallback_summary(sources[0]["text"])

    reference_lines = []
    seen_references: set[str] = set()
    for item in relevant_sentences:
        source = item["source"]
        key = f"{source['source']}::{source['chunk_id']}"
        if key in seen_references:
            continue
        seen_references.add(key)
        reference_lines.append(
            f"[{len(reference_lines) + 1}] {source['source']} / {source['chunk_id']}"
        )

    if not reference_lines:
        source = sources[0]
        reference_lines.append(f"[1] {source['source']} / {source['chunk_id']}")

    return (
        f"问题：{question}\n\n"
        f"回答：{answer_text}\n\n"
        "来源：\n"
        + "\n".join(reference_lines)
    )


def _select_relevant_sentences(question: str, sources: list[dict], limit: int) -> list[dict]:
    scored: list[dict] = []
    for source in sources:
        for sentence in _extract_sentences(source["text"]):
            score = _sentence_score(question, sentence)
            if score <= 0:
                continue
            scored.append({"score": score, "sentence": sentence, "source": source})

    scored.sort(key=lambda item: item["score"], reverse=True)

    selected: list[dict] = []
    seen_sentences: set[str] = set()
    for item in scored:
        normalized = item["sentence"].strip()
        if normalized in seen_sentences:
            continue
        seen_sentences.add(normalized)
        selected.append(item)
        if len(selected) >= limit:
            break

    selected.sort(key=lambda item: (item["source"]["chunk_id"], item["source"]["text"].find(item["sentence"])))
    return selected


def _extract_sentences(text: str) -> list[str]:
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        cleaned_lines.append(stripped)

    cleaned = "\n".join(cleaned_lines)
    sentences = [
        match.group(0).strip()
        for match in ANSWER_SENTENCE_PATTERN.finditer(cleaned)
        if match.group(0).strip()
    ]
    return sentences or [cleaned.strip()]


def _sentence_score(question: str, sentence: str) -> float:
    question_tokens = set(tokenize(question))
    sentence_tokens = set(tokenize(sentence))
    if not question_tokens or not sentence_tokens:
        return 0.0
    overlap = question_tokens & sentence_tokens
    return len(overlap) / (len(question_tokens) ** 0.5 * len(sentence_tokens) ** 0.5)


def _fallback_summary(text: str) -> str:
    sentences = _extract_sentences(text)
    if not sentences:
        return "已检索到相关片段，但当前版本无法提炼出可读回答。"
    return "".join(sentences[:2])
