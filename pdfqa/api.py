"""HTTP API and chat page.

    uvicorn pdfqa.api:app --reload
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .index import Index
from .ingest import chunk_pdf
from .llm import get_embedder, get_llm
from .qa import answer

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
STATIC = Path(__file__).parent / "static"


class Store:
    """Uploaded PDFs plus the index built from them, kept in DATA_DIR."""

    def __init__(self, data_dir: str | Path) -> None:
        self.dir = Path(data_dir)
        self.docs = self.dir / "docs"
        self.docs.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.json"
        self.embed = get_embedder()
        self.index = Index.load(self.index_path, self.embed) if self.index_path.exists() else Index(embed=self.embed)

    def add_pdf(self, filename: str, data: bytes) -> int:
        safe = re.sub(r"[^\w.\- ]", "_", Path(filename).name) or "document.pdf"
        path = self.docs / safe
        if safe in self.index.documents():
            raise HTTPException(409, f"{safe} is already indexed.")
        path.write_bytes(data)
        try:
            chunks = chunk_pdf(path)
        except Exception as exc:  # pypdf raises many different errors on broken files
            path.unlink(missing_ok=True)
            raise HTTPException(422, f"Could not read {safe}: {exc}") from exc
        if not chunks:
            path.unlink(missing_ok=True)
            raise HTTPException(422, f"{safe} has no extractable text (scanned PDF? run OCR first).")
        self.index.add(chunks)
        self.index.save(self.index_path)
        return len(chunks)

    def reset(self) -> None:
        for pdf in self.docs.glob("*"):
            pdf.unlink()
        self.index_path.unlink(missing_ok=True)
        self.index = Index(embed=self.embed)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=4, ge=1, le=10)


def create_app(data_dir: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="PDF Q&A Bot", version="1.0.0")
    store = Store(data_dir or os.getenv("DATA_DIR", "data"))
    llm = get_llm()
    app.state.store = store

    @app.get("/", include_in_schema=False)
    def page() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "llm": llm.name if llm else "none (extractive mode)",
                "embeddings": bool(store.embed), "chunks": len(store.index)}

    @app.get("/documents")
    def documents() -> dict:
        return {"documents": store.index.documents(), "chunks": len(store.index)}

    @app.post("/documents", status_code=201)
    async def upload(files: list[UploadFile] = File(...)) -> dict:
        added = {}
        for file in files:
            if not (file.filename or "").lower().endswith(".pdf"):
                raise HTTPException(415, f"{file.filename}: only PDF files are supported.")
            data = await file.read()
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"{file.filename} is larger than 20 MB.")
            added[file.filename] = store.add_pdf(file.filename, data)
        return {"added": added, "documents": store.index.documents()}

    @app.delete("/documents")
    def reset() -> dict:
        store.reset()
        return {"documents": []}

    @app.post("/ask")
    def ask(q: Question) -> dict:
        if not len(store.index):
            raise HTTPException(400, "Upload at least one PDF first.")
        result = answer(q.question, store.index, llm, k=q.k)
        return {"answer": result.text, "found": result.found, "mode": result.mode,
                "sources": [asdict(s) for s in result.sources]}

    return app


app = create_app()
