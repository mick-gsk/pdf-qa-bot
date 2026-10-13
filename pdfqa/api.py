"""HTTP API and chat page.

    uvicorn pdfqa.api:app --reload
"""

from __future__ import annotations

import logging
import secrets
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .config import Settings
from .llm import get_llm
from .qa import answer
from .store import Store, StoreError

STATIC = Path(__file__).parent / "static"
log = logging.getLogger("pdfqa")


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    k: int | None = Field(default=None, ge=1, le=10, description="excerpts to use; default TOP_K")


def require_token(token: str | None):
    """Dependency that checks `Authorization: Bearer <token>` when API_TOKEN is set."""

    def check(authorization: str = Header(default="")) -> None:
        # compare bytes: compare_digest rejects str with non-ASCII characters. Starlette decodes
        # headers as Latin-1, so encoding back that way gives the bytes the client sent.
        if token and not secrets.compare_digest(authorization.encode("latin-1"), f"Bearer {token}".encode()):
            raise HTTPException(401, "Missing or wrong API token.", headers={"WWW-Authenticate": "Bearer"})

    return check


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    app = FastAPI(title="PDF Q&A Bot", version="1.1.0")
    store = Store(settings)
    llm = get_llm(settings)
    app.state.store = store
    max_bytes = settings.max_upload_mb * 1024 * 1024
    auth = [Depends(require_token(settings.api_token))]
    log.info("LLM: %s, embeddings: %s, %d chunks loaded",
             llm.name if llm else "none", settings.embeddings, len(store.index))

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        log.info("%s %s %d %.0fms id=%s", request.method, request.url.path, response.status_code,
                 (time.perf_counter() - start) * 1000, request_id)
        return response

    @app.exception_handler(StoreError)
    async def store_error(request: Request, exc: StoreError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=exc.status)

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "-")
        log.exception("unhandled error id=%s", request_id)
        # this handler runs outside request_log, so the header has to be set here
        return JSONResponse({"detail": "Internal error.", "request_id": request_id}, status_code=500,
                            headers={"X-Request-ID": request_id})

    @app.get("/", include_in_schema=False)
    def page() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "llm": llm.name if llm else "none (extractive mode)",
                "embeddings": bool(store.embed), "chunks": len(store.index),
                "auth": bool(settings.api_token)}

    @app.get("/documents", dependencies=auth)
    def documents() -> dict:
        return {"documents": store.documents(), "chunks": len(store.index)}

    @app.post("/documents", status_code=201, dependencies=auth)
    async def upload(files: list[UploadFile] = File(...)) -> dict:
        added = {}
        for file in files:
            if not (file.filename or "").lower().endswith(".pdf"):
                raise HTTPException(415, f"{file.filename}: only PDF files are supported.")
            data = await file.read()
            if len(data) > max_bytes:
                raise HTTPException(413, f"{file.filename} is larger than {settings.max_upload_mb} MB.")
            if not data.startswith(b"%PDF-"):
                raise HTTPException(415, f"{file.filename} is not a PDF file.")
            # parsing and embedding are blocking; keep them off the event loop
            added[file.filename] = await run_in_threadpool(store.add_pdf, file.filename, data)
        return {"added": added, "documents": store.documents()}

    @app.get("/documents/{name}", dependencies=auth, response_class=FileResponse)
    def download(name: str) -> FileResponse:
        path = store.path(name)
        # Starlette encodes non-Latin names per RFC 5987 (filename*=utf-8''...)
        return FileResponse(path, media_type="application/pdf", filename=path.name,
                            content_disposition_type="inline")

    @app.delete("/documents/{name}", dependencies=auth)
    def remove(name: str) -> dict:
        store.remove(name)
        return {"documents": store.documents()}

    @app.delete("/documents", dependencies=auth)
    def reset() -> dict:
        store.reset()
        return {"documents": []}

    @app.post("/ask", dependencies=auth)
    def ask(q: Question) -> dict:
        if not len(store.index):
            raise HTTPException(400, "Upload at least one PDF first.")
        try:
            result = answer(q.question, store.index, llm, k=q.k or settings.top_k)
        except Exception as exc:
            log.warning("LLM request failed: %s", exc)
            raise HTTPException(502, "The language model did not respond. Try again in a moment.") from exc
        return {"answer": result.text, "found": result.found, "mode": result.mode,
                "sources": [asdict(s) for s in result.sources]}

    return app


app = create_app()
