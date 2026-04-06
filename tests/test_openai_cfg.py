"""Tests for the inference client configuration helpers."""

import pytest

from nextext.modules import openai_cfg
from nextext.modules.openai_cfg import InferencePipeline


def test_client_uses_configured_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the client is constructed with OPENAI_API_KEY and OPENAI_API_BASE.

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
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_BASE", "http://inference-server/v1")
    monkeypatch.setenv("TEXT_MODEL", "llama3.1:8b")

    pipeline = InferencePipeline()
    _ = pipeline.client

    assert recorded_kwargs["api_key"] == "test-key"
    assert recorded_kwargs["base_url"] == "http://inference-server/v1"


def test_client_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that accessing the client raises when OPENAI_API_KEY is not set.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    pipeline = InferencePipeline()

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        _ = pipeline.api_key


def test_base_url_strips_trailing_slash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that base_url strips a trailing slash from OPENAI_API_BASE.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    monkeypatch.setenv("OPENAI_API_BASE", "http://inference-server/v1/")

    pipeline = InferencePipeline()

    assert pipeline.base_url == "http://inference-server/v1"


def test_default_model_uses_text_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that TEXT_MODEL is used for the default model.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    monkeypatch.setenv("TEXT_MODEL", "llama3.1:8b")

    pipeline = InferencePipeline()

    assert pipeline.default_model == "llama3.1:8b"


def test_default_model_requires_text_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that text analysis requires TEXT_MODEL.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    monkeypatch.delenv("TEXT_MODEL", raising=False)

    pipeline = InferencePipeline()

    with pytest.raises(RuntimeError, match="TEXT_MODEL"):
        _ = pipeline.default_model


def test_translation_model_requires_translation_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that translation requires TRANSLATION_MODEL.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    monkeypatch.setenv("TEXT_MODEL", "llama3.1:8b")
    monkeypatch.delenv("TRANSLATION_MODEL", raising=False)

    pipeline = InferencePipeline()

    with pytest.raises(RuntimeError, match="TRANSLATION_MODEL"):
        _ = pipeline.translation_model
