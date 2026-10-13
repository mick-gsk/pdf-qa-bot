import pytest

from pdfqa.config import ConfigError, Settings, read_dotenv


def test_defaults_without_keys_use_extractive_mode():
    s = Settings.from_env({})
    assert s.llm_provider == "none" and s.embeddings == "none" and s.top_k == 4


def test_provider_follows_available_key():
    assert Settings.from_env({"OPENAI_API_KEY": "x"}).llm_provider == "openai"
    s = Settings.from_env({"ANTHROPIC_API_KEY": "x", "OPENAI_API_KEY": "y"})
    assert s.llm_provider == "anthropic" and s.llm_model.startswith("claude")


def test_explicit_provider_and_model():
    s = Settings.from_env({"LLM_PROVIDER": "Ollama", "LLM_MODEL": "qwen3", "OLLAMA_HOST": "http://gpu:11434/"})
    assert (s.llm_provider, s.llm_model, s.ollama_host) == ("ollama", "qwen3", "http://gpu:11434")


@pytest.mark.parametrize("env, message", [
    ({"LLM_PROVIDER": "antropic"}, "LLM_PROVIDER"),
    ({"EMBEDDINGS": "cohere"}, "EMBEDDINGS"),
    ({"TOP_K": "four"}, "whole number"),
    ({"MAX_UPLOAD_MB": "0"}, "between"),
])
def test_invalid_values_fail_early(env, message):
    with pytest.raises(ConfigError, match=message):
        Settings.from_env(env)


def test_dotenv_parsing(tmp_path):
    env = tmp_path / ".env"
    env.write_text('# comment\nANTHROPIC_API_KEY=\nLLM_PROVIDER="ollama"  # local\nTOP_K=6\n')
    assert read_dotenv(env) == {"LLM_PROVIDER": "ollama", "TOP_K": "6"}
