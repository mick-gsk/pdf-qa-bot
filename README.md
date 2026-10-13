# PDF Q&A Bot

Upload PDFs, ask questions in plain language, and get answers that cite the exact page they came from.
If the answer is not in your documents, the bot says so instead of guessing.

![Screenshot: two questions answered from an employee handbook, each with page citations](docs/screenshot.png)

Typical uses: an internal assistant for HR handbooks and policies, a support bot for product manuals,
or quick answers from contracts and reports.

## What it does

- **Answers with sources.** Every statement is marked `[1]`, `[2]` and linked to the PDF page it came from.
- **Says "not found" rather than making things up.** The model is instructed to use only the retrieved
  excerpts and answers with a fixed "not found" sentence otherwise.
- **Works with any LLM.** Anthropic Claude, OpenAI, or a local model through Ollama, chosen with one
  environment variable. Without any API key it still works in an extractive mode that quotes the most
  relevant sentences, so the demo runs offline.
- **Hybrid search.** BM25 keyword search finds exact names and numbers. Optional embeddings (OpenAI or
  Ollama) add semantic matching for paraphrased questions; both rankings are merged with
  Reciprocal Rank Fusion.
- **Page-aware chunking.** Text is split at sentence boundaries with overlap, and chunks never cross a page,
  so citations are always exact.
- **Web page, REST API and CLI.** Drag-and-drop chat page, a FastAPI backend with OpenAPI docs at `/docs`,
  and a command-line tool for scripts.
- **Tested.** 23 tests cover parsing, retrieval, the answer logic (with a fake LLM) and the API.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[anthropic,openai,dev]"

cp .env.example .env           # optional: add an API key
uvicorn pdfqa.api:app --reload # open http://localhost:8000
```

Drop `examples/employee-handbook.pdf` onto the page and ask, for example:
*"How much is the hotel limit in Paris?"* or *"Can I work remotely from abroad?"*

### Docker

```bash
docker build -t pdf-qa-bot .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=... -v pdfqa-data:/data pdf-qa-bot
```

### Command line

```bash
pdfqa index examples/employee-handbook.pdf
pdfqa ask "How many vacation days do full-time employees get?"
# Full-time employees receive 28 vacation days per calendar year. [1] Part-time employees
# receive vacation days in proportion to their working days. [1]
#   [1] employee-handbook.pdf, p. 1
```

## Configuration

| Variable | Values | Default |
|---|---|---|
| `LLM_PROVIDER` | `anthropic`, `openai`, `ollama`, `none` | first provider with an API key, else `none` |
| `LLM_MODEL` | any model name of that provider | a small, inexpensive model per provider |
| `EMBEDDINGS` | `openai`, `ollama`, `none` | `none` (keyword search only) |
| `OLLAMA_HOST` | URL | `http://localhost:11434` |
| `DATA_DIR` | folder for uploaded PDFs and the index | `data` |

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/documents` | upload one or more PDFs (multipart field `files`) |
| `GET` | `/documents` | list indexed documents |
| `DELETE` | `/documents` | remove all documents |
| `POST` | `/ask` | `{"question": "...", "k": 4}` → answer, sources, mode |
| `GET` | `/health` | active model and index size |

## How it works

```
PDF ──► page text ──► sentence chunks (per page, with overlap)
                            │
question ──► BM25 ──┐       ▼
            embed ──┴─► RRF fusion ──► top k excerpts ──► LLM with "cite or say not found" prompt
                                                      └─► (no LLM) best matching sentences
```

| File | Responsibility |
|---|---|
| `pdfqa/ingest.py` | PDF text extraction, cleanup, sentence-based chunking |
| `pdfqa/index.py` | BM25, optional embeddings, Reciprocal Rank Fusion, save/load |
| `pdfqa/qa.py` | prompt, citation handling, extractive fallback |
| `pdfqa/llm.py` | Anthropic, OpenAI and Ollama adapters |
| `pdfqa/api.py` | FastAPI app and upload handling |
| `pdfqa/static/index.html` | the chat page (plain HTML and JavaScript) |

## Limits

- Scanned PDFs without a text layer need OCR first; the API rejects them with a clear message.
- The index is a JSON file, which is fine for hundreds of pages. For large document collections,
  swap `Index` for a vector database such as PostgreSQL with pgvector or Qdrant.
- There is no user management; put it behind your existing login or a reverse proxy.

## Tests

```bash
pytest
```

## License

MIT
