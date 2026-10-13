"""LLM and embedding providers, picked from environment variables.

LLM_PROVIDER: anthropic | openai | ollama | none   (default: first one with a key, else none)
LLM_MODEL:    model name for the chosen provider    (sensible default per provider)
EMBEDDINGS:   openai | ollama | none                (default: none, keyword search only)
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Protocol

import httpx

DEFAULT_MODELS = {
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.1",
}


class LLM(Protocol):
    name: str

    def complete(self, system: str, prompt: str) -> str: ...


class AnthropicLLM:
    def __init__(self, model: str) -> None:
        import anthropic

        self.client = anthropic.Anthropic()
        self.name = f"anthropic:{model}"
        self.model = model

    def complete(self, system: str, prompt: str) -> str:
        msg = self.client.messages.create(
            model=self.model, max_tokens=800, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in msg.content if block.type == "text")


class OpenAILLM:
    def __init__(self, model: str) -> None:
        import openai

        self.client = openai.OpenAI()
        self.name = f"openai:{model}"
        self.model = model

    def complete(self, system: str, prompt: str) -> str:
        res = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        return res.choices[0].message.content or ""


class OllamaLLM:
    def __init__(self, model: str, host: str | None = None) -> None:
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.name = f"ollama:{model}"
        self.model = model

    def complete(self, system: str, prompt: str) -> str:
        res = httpx.post(
            f"{self.host}/api/chat", timeout=120,
            json={"model": self.model, "stream": False, "messages": [
                {"role": "system", "content": system}, {"role": "user", "content": prompt}]},
        )
        res.raise_for_status()
        return res.json()["message"]["content"]


def get_llm() -> LLM | None:
    provider = os.getenv("LLM_PROVIDER", "").lower()
    if not provider:
        if os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        else:
            provider = "none"
    if provider == "none":
        return None
    model = os.getenv("LLM_MODEL") or DEFAULT_MODELS[provider]
    return {"anthropic": AnthropicLLM, "openai": OpenAILLM, "ollama": OllamaLLM}[provider](model)


def get_embedder() -> Callable[[Sequence[str]], list[list[float]]] | None:
    kind = os.getenv("EMBEDDINGS", "none").lower()
    if kind == "openai":
        import openai

        client = openai.OpenAI()
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

        def embed(texts: Sequence[str]) -> list[list[float]]:
            res = client.embeddings.create(model=model, input=list(texts))
            return [d.embedding for d in res.data]

        embed.model = f"openai/{model}"  # type: ignore[attr-defined]
        return embed
    if kind == "ollama":
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

        def embed(texts: Sequence[str]) -> list[list[float]]:
            res = httpx.post(f"{host}/api/embed", json={"model": model, "input": list(texts)}, timeout=120)
            res.raise_for_status()
            return res.json()["embeddings"]

        embed.model = f"ollama/{model}"  # type: ignore[attr-defined]
        return embed
    return None
