from __future__ import annotations

import shutil
import time
from pathlib import Path

from rag_app.core.documents import chunk_text, load_document_text
from rag_app.core.index import LocalSearchIndex


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

    def ask(self, question: str, top_k: int = 3) -> dict:
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

    source_lines = []
    for idx, source in enumerate(sources, start=1):
        preview = source["text"][:180].strip()
        source_lines.append(f"[{idx}] {preview}... 来源：{source['source']} / {source['chunk_id']}")

    return (
        f"问题：{question}\n\n"
        "基于知识库中检索到的内容，可以先参考以下信息：\n"
        + "\n".join(source_lines)
        + "\n\n说明：当前版本使用可解释的本地检索基线生成回答；接入 LLM 后，可把这些片段作为上下文生成更自然的答案。"
    )
