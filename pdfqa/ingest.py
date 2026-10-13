"""Read PDFs and split them into overlapping, page-aware chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class Chunk:
    doc: str
    page: int
    text: str

    @property
    def citation(self) -> str:
        return f"{self.doc}, p. {self.page}"


def read_pdf(path: str | Path) -> list[tuple[int, str]]:
    """Return (page_number, text) for every page that contains text."""
    reader = PdfReader(str(path))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = _clean(page.extract_text() or "")
        if text:
            pages.append((number, text))
    return pages


def _clean(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)  # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split into sentences; short unpunctuated lines (headings) become their own item."""
    sentences: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        joined = " ".join(buffer)
        sentences.extend(s.strip() for s in _SENTENCE_END.split(joined) if s.strip())
        buffer.clear()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for i, line in enumerate(lines):
        following = lines[i + 1] if i + 1 < len(lines) else ""
        # a wrapped sentence continues in lower case; a heading is followed by a new sentence
        if is_heading(line) and (not following or following[0].isupper()):
            flush()
            sentences.append(line)
        else:
            buffer.append(line)
    flush()
    return sentences


def is_heading(line: str) -> bool:
    return len(line.split()) <= 8 and not line.rstrip().endswith((".", "!", "?", ",", ";", ":"))


def chunk_page(doc: str, page: int, text: str, max_words: int = 120, overlap: int = 1) -> list[Chunk]:
    """Group whole sentences into chunks of about max_words, overlapping by `overlap` sentences.

    Chunks never cross page boundaries, so every answer can cite an exact page.
    """
    sentences = split_sentences(text)
    chunks: list[Chunk] = []
    current: list[str] = []
    words = 0
    for sentence in sentences:
        n = len(sentence.split())
        if current and words + n > max_words:
            chunks.append(Chunk(doc, page, "\n".join(current)))
            current = current[-overlap:] if overlap else []
            words = sum(len(s.split()) for s in current)
        current.append(sentence)
        words += n
    if current:
        chunks.append(Chunk(doc, page, "\n".join(current)))
    return chunks


def chunk_pdf(path: str | Path, max_words: int = 120) -> list[Chunk]:
    name = Path(path).name
    chunks: list[Chunk] = []
    for page, text in read_pdf(path):
        chunks.extend(chunk_page(name, page, text, max_words=max_words))
    return chunks
