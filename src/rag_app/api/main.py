from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from rag_app.core.documents import SUPPORTED_EXTENSIONS
from rag_app.core.rag import KnowledgeBase


app = FastAPI(title="Campus RAG Assistant", version="0.1.0")
kb = KnowledgeBase()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["学生如何申请奖学金？"])
    top_k: int = Field(default=3, ge=1, le=8)


@app.get("/health")
def health() -> dict:
    return kb.health()


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Supported: {supported}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / Path(file.filename or "document").name
        temp_path.write_bytes(await file.read())
        return kb.add_document(temp_path)


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    return kb.ask(request.question, top_k=request.top_k)
