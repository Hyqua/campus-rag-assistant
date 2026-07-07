from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from rag_app.core.documents import SUPPORTED_EXTENSIONS
from rag_app.core.rag import KnowledgeBase


st.set_page_config(page_title="Campus RAG Assistant", layout="wide")

kb = KnowledgeBase()

st.title("Campus RAG Assistant")

with st.sidebar:
    st.subheader("Knowledge Base")
    health = kb.health()
    st.metric("Indexed chunks", health["chunks"])
    uploaded = st.file_uploader("Upload document", type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS])
    if uploaded is not None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / uploaded.name
            temp_path.write_bytes(uploaded.getvalue())
            result = kb.add_document(temp_path)
        st.success(f"Indexed {result['chunks_added']} chunks from {result['source']}")

question = st.text_input("Ask a question", placeholder="学生如何申请奖学金？")
top_k = st.slider("Top-k sources", min_value=1, max_value=8, value=3)

if st.button("Ask", type="primary", disabled=not question.strip()):
    result = kb.ask(question, top_k=top_k)
    st.subheader("Answer")
    st.write(result["answer"])
    st.caption(f"Latency: {result['latency_ms']} ms")

    st.subheader("Sources")
    for source in result["sources"]:
        with st.expander(f"{source['source']} | score={source['score']}"):
            st.write(source["text"])
