from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag_app.core.documents import chunk_text
from rag_app.core.index import LocalSearchIndex
from rag_app.core.rag import KnowledgeBase


class RagCoreTests(unittest.TestCase):
    def test_chunk_text_keeps_source_metadata(self) -> None:
        chunks = chunk_text("one two three four five six", source="demo.md", chunk_size=3, overlap=1)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].source, "demo.md")
        self.assertEqual(chunks[0].chunk_id, "demo.md::0")

    def test_index_search_returns_relevant_chunk(self) -> None:
        chunks = chunk_text("奖学金 申请 需要 成绩 单 和 推荐 信", source="policy.md", chunk_size=20, overlap=0)
        index = LocalSearchIndex()
        index.add(chunks)
        results = index.search("奖学金 怎么 申请", top_k=1)
        self.assertEqual(results[0]["source"], "policy.md")

    def test_knowledge_base_ingest_and_ask(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            doc_path = root / "policy.md"
            doc_path.write_text("奖学金申请需要提交成绩单、个人陈述和导师推荐信。", encoding="utf-8")

            kb = KnowledgeBase(data_dir=root / "data")
            ingest_result = kb.add_document(doc_path, chunk_size=30, overlap=0)
            ask_result = kb.ask("奖学金申请需要什么材料？")

            self.assertEqual(ingest_result["chunks_added"], 1)
            self.assertTrue(ask_result["sources"])
            self.assertIn("来源", ask_result["answer"])

    def test_reingest_same_source_replaces_old_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            doc_path = root / "policy.md"
            doc_path.write_text("第一次内容 奖学金", encoding="utf-8")

            kb = KnowledgeBase(data_dir=root / "data")
            first = kb.add_document(doc_path, chunk_size=30, overlap=0)
            second = kb.add_document(doc_path, chunk_size=30, overlap=0)

            self.assertEqual(first["chunks_replaced"], 0)
            self.assertEqual(second["chunks_replaced"], 1)
            self.assertEqual(kb.health()["chunks"], 1)


if __name__ == "__main__":
    unittest.main()
