"""Tests for the inference client configuration helpers."""

import pytest

from nextext.modules import openai_cfg
from nextext.modules.openai_cfg import InferencePipeline


def test_ollama_client_uses_placeholder_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that Ollama client creation uses a placeholder API key.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables and module attributes.
    """
    recorded_kwargs: dict[str, str] = {}

    class DummyClient:
        """Minimal OpenAI client stub for constructor assertions."""

        def __init__(self, **kwargs: str) -> None:
            """Store the client constructor arguments for inspection.

            Args:
                **kwargs (str): Arbitrary client keyword arguments.
            """
            recorded_kwargs.update(kwargs)

    monkeypatch.setattr(openai_cfg, "OpenAIClient", DummyClient)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("TEXT_MODEL", "llama3.1:8b")

    pipeline = InferencePipeline(provider="ollama")

    _ = pipeline.client

    assert recorded_kwargs["api_key"] == "ollama"
    assert recorded_kwargs["base_url"].endswith("/v1")


def test_ollama_base_url_avoids_duplicate_v1(
) -> None:
    """Test that Ollama base URLs do not append ``/v1`` twice."""
    pipeline = InferencePipeline(provider="ollama")
    pipeline.openai_api_base = "http://localhost:11434/v1"

    assert pipeline.base_url == "http://localhost:11434/v1"
    assert pipeline.ollama_api_base == "http://localhost:11434"


def test_openai_provider_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the hosted OpenAI provider requires ``OPENAI_API_KEY``.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("TEXT_MODEL", "gpt-4o")

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        InferencePipeline(provider="openai")


def test_openai_client_uses_environment_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the hosted OpenAI provider uses ``OPENAI_API_KEY``.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables and module attributes.
    """
    recorded_kwargs: dict[str, str] = {}

    class DummyClient:
        """Minimal OpenAI client stub for constructor assertions."""

        def __init__(self, **kwargs: str) -> None:
            """Store the client constructor arguments for inspection.

            Args:
                **kwargs (str): Arbitrary client keyword arguments.
            """
            recorded_kwargs.update(kwargs)

    monkeypatch.setattr(openai_cfg, "OpenAIClient", DummyClient)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("TEXT_MODEL", "gpt-4o")

    pipeline = InferencePipeline(provider="openai")

    _ = pipeline.client

    assert recorded_kwargs["api_key"] == "test-openai-key"


def test_ollama_default_model_uses_text_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that Ollama general LLM selection prefers ``TEXT_MODEL``.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    monkeypatch.setenv("TEXT_MODEL", "llama3.1:8b")

    pipeline = InferencePipeline(provider="ollama")

    assert pipeline.default_model == "llama3.1:8b"


def test_default_model_requires_text_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that text analysis requires ``TEXT_MODEL``.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    monkeypatch.delenv("TEXT_MODEL", raising=False)

    pipeline = InferencePipeline(provider="ollama")

    with pytest.raises(RuntimeError, match="TEXT_MODEL"):
        _ = pipeline.default_model


def test_translation_model_requires_translation_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that translation requires ``TRANSLATION_MODEL``.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    monkeypatch.setenv("TEXT_MODEL", "llama3.1:8b")
    monkeypatch.delenv("TRANSLATION_MODEL", raising=False)

    pipeline = InferencePipeline(provider="ollama")

    with pytest.raises(RuntimeError, match="TRANSLATION_MODEL"):
        _ = pipeline.translation_model
