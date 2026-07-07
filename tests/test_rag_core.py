from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag_app.core.documents import chunk_text
from rag_app.core.index import LocalSearchIndex
from rag_app.core.rag import KnowledgeBase, compose_answer


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

    def test_markdown_chunking_prefers_paragraph_boundaries(self) -> None:
        text = """# 校园事务

## 奖学金申请

学生申请奖学金需要提交成绩单和个人陈述。

申请流程包括填写申请表、上传材料、导师推荐和学院公示。
"""
        chunks = chunk_text(text, source="policy.md", chunk_size=45, overlap=5)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(chunks[0].text.startswith("# 校园事务"))
        self.assertFalse(chunks[1].text.startswith("，"))

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

    def test_compose_answer_is_extractive_and_concise(self) -> None:
        answer = compose_answer(
            "学生如何申请奖学金？",
            [
                {
                    "source": "policy.md",
                    "chunk_id": "policy.md::0",
                    "score": 0.7,
                    "text": "# 校园事务\n\n## 奖学金申请\n\n学生申请奖学金需要提交成绩单、个人陈述、导师推荐信和获奖证明。申请流程包括填写申请表、上传材料、导师推荐和学院评审公示。",
                }
            ],
        )
        self.assertIn("回答：", answer)
        self.assertIn("成绩单", answer)
        self.assertIn("来源：", answer)
        self.assertNotIn("基于知识库中检索到的内容", answer)


if __name__ == "__main__":
    unittest.main()
