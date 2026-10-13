"""LLM and embedding providers.

Each LLM adapter implements `complete` (whole answer) and `stream` (answer in pieces,
for the chat page). The SDKs are imported lazily so that only the provider in use
has to be installed.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from typing import Protocol

import httpx

from .config import Settings

MAX_TOKENS = 800


class LLM(Protocol):
    name: str

    def complete(self, system: str, prompt: str) -> str: ...

    def stream(self, system: str, prompt: str) -> Iterator[str]: ...


class AnthropicLLM:
    def __init__(self, model: str, api_key: str | None = None) -> None:
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.name = f"anthropic:{model}"
        self.model = model

    def complete(self, system: str, prompt: str) -> str:
        return "".join(self.stream(system, prompt))

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        with self.client.messages.stream(
            model=self.model, max_tokens=MAX_TOKENS, system=system,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            yield from stream.text_stream


class OpenAILLM:
    def __init__(self, model: str, api_key: str | None = None) -> None:
        import openai

        self.client = openai.OpenAI(api_key=api_key)
        self.name = f"openai:{model}"
        self.model = model

    def complete(self, system: str, prompt: str) -> str:
        return "".join(self.stream(system, prompt))

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        chunks = self.client.chat.completions.create(
            model=self.model, max_tokens=MAX_TOKENS, stream=True,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        for chunk in chunks:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class OllamaLLM:
    def __init__(self, model: str, host: str) -> None:
        self.host = host
        self.name = f"ollama:{model}"
        self.model = model

    def complete(self, system: str, prompt: str) -> str:
        return "".join(self.stream(system, prompt))

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        body = {"model": self.model, "stream": True, "options": {"num_predict": MAX_TOKENS},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
        with httpx.stream("POST", f"{self.host}/api/chat", json=body, timeout=120) as res:
            res.raise_for_status()
            for line in res.iter_lines():
                piece = json.loads(line).get("message", {}).get("content", "") if line else ""
                if piece:
                    yield piece


def get_llm(settings: Settings) -> LLM | None:
    if settings.llm_provider == "anthropic":
        return AnthropicLLM(settings.llm_model, settings.anthropic_api_key)
    if settings.llm_provider == "openai":
        return OpenAILLM(settings.llm_model, settings.openai_api_key)
    if settings.llm_provider == "ollama":
        return OllamaLLM(settings.llm_model, settings.ollama_host)
    return None


def get_embedder(settings: Settings) -> Callable[[Sequence[str]], list[list[float]]] | None:
    model = settings.embedding_model
    if settings.embeddings == "openai":
        import openai

        client = openai.OpenAI(api_key=settings.openai_api_key)

        def embed(texts: Sequence[str]) -> list[list[float]]:
            vectors: list[list[float]] = []
            for start in range(0, len(texts), 256):  # stay well below the API's batch limit
                res = client.embeddings.create(model=model, input=list(texts[start:start + 256]))
                vectors.extend(d.embedding for d in res.data)
            return vectors

        embed.model = f"openai/{model}"  # type: ignore[attr-defined]
        return embed
    if settings.embeddings == "ollama":
        host = settings.ollama_host

        def embed(texts: Sequence[str]) -> list[list[float]]:
            res = httpx.post(f"{host}/api/embed", json={"model": model, "input": list(texts)}, timeout=120)
            res.raise_for_status()
            return res.json()["embeddings"]

        embed.model = f"ollama/{model}"  # type: ignore[attr-defined]
        return embed
    return None
