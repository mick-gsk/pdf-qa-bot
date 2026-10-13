import os
import tempfile
from pathlib import Path

import pytest

# The API module builds its app at import time; keep test data out of the repo.
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="pdfqa-test-"))
for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LLM_PROVIDER", "EMBEDDINGS", "API_TOKEN"):
    os.environ.pop(var, None)

SAMPLE = Path(__file__).parent.parent / "examples" / "employee-handbook.pdf"


@pytest.fixture
def sample_pdf() -> Path:
    return SAMPLE
