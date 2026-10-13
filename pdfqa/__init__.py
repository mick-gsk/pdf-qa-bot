"""Ask questions about your PDFs and get answers with page citations."""

from .index import Index
from .ingest import Chunk, chunk_pdf
from .qa import Answer, answer

__all__ = ["Answer", "Chunk", "Index", "answer", "chunk_pdf"]
__version__ = "1.0.0"
