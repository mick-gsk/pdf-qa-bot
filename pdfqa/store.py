"""Uploaded PDFs plus the search index built from them, kept in DATA_DIR."""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import BinaryIO

from .config import Settings
from .index import Index
from .ingest import chunk_pdf
from .llm import get_embedder

log = logging.getLogger(__name__)


class StoreError(Exception):
    status = 400


class DuplicateDocument(StoreError):
    status = 409


class UnreadableDocument(StoreError):
    status = 422


class DocumentNotFound(StoreError):
    status = 404


def safe_name(filename: str) -> str:
    return re.sub(r"[^\w.\- ]", "_", Path(filename).name).strip(" .") or "document.pdf"


class Store:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.docs = settings.data_dir / "docs"
        self.docs.mkdir(parents=True, exist_ok=True)
        self.index_path = settings.data_dir / "index.json"
        self.embed = get_embedder(settings)
        self.index = Index.load(self.index_path, self.embed) if self.index_path.exists() else Index(embed=self.embed)
        # Endpoints run in a thread pool; writes to the index must not interleave. Searches
        # take no lock: writers build a new index and swap it in.
        self.lock = threading.Lock()

    def documents(self) -> list[str]:
        return self.index.documents()

    def path(self, name: str) -> Path:
        path = self.docs / safe_name(name)
        if safe_name(name) not in self.index.documents() or not path.is_file():
            raise DocumentNotFound(f"{name} is not indexed.")
        return path

    def open(self, name: str) -> BinaryIO:
        """An open handle to the PDF. It stays readable if the document is removed meanwhile,
        so a download that is still streaming is not cut off by a delete."""
        with self.lock:
            return self.path(name).open("rb")

    def add_pdf(self, filename: str, data: bytes) -> int:
        name = safe_name(filename)
        with self.lock:
            if name in self.index.documents():
                raise DuplicateDocument(f"{name} is already indexed.")
            path = self.docs / name
            path.write_bytes(data)
            try:
                chunks = chunk_pdf(path, max_words=self.settings.chunk_words)
            except Exception as exc:  # pypdf raises many different errors on broken files
                path.unlink(missing_ok=True)
                raise UnreadableDocument(f"Could not read {name}: {exc}") from exc
            if not chunks:
                path.unlink(missing_ok=True)
                raise UnreadableDocument(f"{name} has no extractable text (scanned PDF? run OCR first).")
            index = self.index.copy()
            index.add(chunks)
            index.save(self.index_path)
            self.index = index
        log.info("indexed %s: %d chunks", name, len(chunks))
        return len(chunks)

    def remove(self, name: str) -> None:
        with self.lock:
            path = self.path(name)
            index = self.index.copy()
            index.remove(path.name)
            index.save(self.index_path)
            self.index = index
            path.unlink()
        log.info("removed %s", path.name)

    def reset(self) -> None:
        with self.lock:
            for pdf in self.docs.glob("*"):
                pdf.unlink()
            self.index_path.unlink(missing_ok=True)
            self.index = Index(embed=self.embed)
        log.info("removed all documents")
