"""Settings read from environment variables (and a .env file if present).

All configuration lives here so that a typo like LLM_PROVIDER=antropic fails at
startup with a clear message instead of on the first question.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

LLM_PROVIDERS = ("anthropic", "openai", "ollama", "none")
EMBEDDING_PROVIDERS = ("openai", "ollama", "none")
GROUNDING_MODES = ("repair", "flag", "off")
DEFAULT_MODELS = {
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.1",
}
DEFAULT_EMBEDDING_MODELS = {"openai": "text-embedding-3-small", "ollama": "nomic-embed-text"}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "none"
    llm_model: str = ""
    embeddings: str = "none"
    embedding_model: str = ""
    ollama_host: str = "http://localhost:11434"
    data_dir: Path = field(default_factory=lambda: Path("data"))
    api_token: str | None = None
    max_upload_mb: int = 20
    chunk_words: int = 120
    top_k: int = 4
    log_level: str = "INFO"
    grounding: str = "repair"
    # kept here because a key that is only in .env is not in os.environ for the SDKs to find
    anthropic_api_key: str | None = field(default=None, repr=False)
    openai_api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Settings:
        env = {**read_dotenv(Path(".env")), **os.environ} if env is None else dict(env)
        provider = env.get("LLM_PROVIDER", "").lower()
        if not provider:
            # Pick the first provider that has a key, so `ANTHROPIC_API_KEY=... uvicorn ...` just works.
            provider = ("anthropic" if env.get("ANTHROPIC_API_KEY")
                        else "openai" if env.get("OPENAI_API_KEY") else "none")
        _check("LLM_PROVIDER", provider, LLM_PROVIDERS)
        embeddings = env.get("EMBEDDINGS", "none").lower()
        _check("EMBEDDINGS", embeddings, EMBEDDING_PROVIDERS)
        grounding = env.get("GROUNDING", "repair").lower()
        _check("GROUNDING", grounding, GROUNDING_MODES)
        return cls(
            llm_provider=provider,
            llm_model=env.get("LLM_MODEL") or DEFAULT_MODELS.get(provider, ""),
            embeddings=embeddings,
            embedding_model=env.get("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODELS.get(embeddings, ""),
            ollama_host=env.get("OLLAMA_HOST", cls.ollama_host).rstrip("/"),
            data_dir=Path(env.get("DATA_DIR", "data")),
            api_token=env.get("API_TOKEN") or None,
            max_upload_mb=_int(env, "MAX_UPLOAD_MB", cls.max_upload_mb, 1, 500),
            chunk_words=_int(env, "CHUNK_WORDS", cls.chunk_words, 30, 1000),
            top_k=_int(env, "TOP_K", cls.top_k, 1, 10),
            log_level=env.get("LOG_LEVEL", cls.log_level).upper(),
            grounding=grounding,
            anthropic_api_key=env.get("ANTHROPIC_API_KEY") or None,
            openai_api_key=env.get("OPENAI_API_KEY") or None,
        )


def read_dotenv(path: Path) -> dict[str, str]:
    """Minimal .env parser: KEY=VALUE lines, # comments, optional quotes. Real env vars win."""
    if not path.is_file():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.split(" #", 1)[0].strip().strip("'\"")
        if value:
            values[key.strip()] = value
    return values


def _check(name: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ConfigError(f"{name}={value!r} is not supported; use one of: {', '.join(allowed)}")


def _int(env: dict[str, str], name: str, default: int, low: int, high: int) -> int:
    raw = env.get(name)
    if not raw:  # unset or empty
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name}={raw!r} must be a whole number") from None
    if not low <= value <= high:
        raise ConfigError(f"{name}={value} must be between {low} and {high}")
    return value
