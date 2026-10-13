import contextlib
import json

import httpx
import pytest

from pdfqa import llm as llm_module
from pdfqa.config import Settings
from pdfqa.llm import OllamaLLM, get_embedder, get_llm


def test_extractive_mode_has_no_llm():
    assert get_llm(Settings()) is None and get_embedder(Settings()) is None


def test_ollama_stream_reads_ndjson(monkeypatch):
    lines = [json.dumps({"message": {"content": piece}}) for piece in ("Up to ", "120 euros [1].")]
    lines.append(json.dumps({"done": True}))
    seen = {}

    @contextlib.contextmanager
    def fake_stream(method, url, json, timeout):
        seen.update(url=url, body=json)
        yield httpx.Response(200, text="\n".join(lines), request=httpx.Request(method, url))

    monkeypatch.setattr(llm_module.httpx, "stream", fake_stream)
    model = OllamaLLM("llama3.1", "http://gpu:11434")
    assert list(model.stream("sys", "prompt")) == ["Up to ", "120 euros [1]."]
    assert model.complete("sys", "prompt") == "Up to 120 euros [1]."
    assert seen["url"] == "http://gpu:11434/api/chat"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"}


def test_ollama_errors_are_raised(monkeypatch):
    @contextlib.contextmanager
    def fake_stream(method, url, json, timeout):
        yield httpx.Response(404, text="model not found", request=httpx.Request(method, url))

    monkeypatch.setattr(llm_module.httpx, "stream", fake_stream)
    with pytest.raises(httpx.HTTPStatusError, match="404"):
        OllamaLLM("missing", "http://localhost:11434").complete("s", "p")
