"""Tests for the Translator provider branching."""

from typing import Any

import pytest

from nextext.core.openai_cfg import InferencePipeline
from nextext.core.translation import Translator


class _RecordingPipeline(InferencePipeline):
    """InferencePipeline subclass that records call_model invocations in-memory."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    def call_model(  # type: ignore[override]
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.1,
        seed: int = 42,
        stop: list[str] | None = None,
        num_predict: int | None = None,
        top_p: float | None = None,
        system_prompt: str | None = None,
        include_system_prompt: bool = True,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "model": model,
                "temperature": temperature,
                "system_prompt": system_prompt,
                "include_system_prompt": include_system_prompt,
            }
        )
        return f"translated::{prompt}"


def _make_translator(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    translation_model: str = "translategemma:4b",
) -> tuple[Translator, _RecordingPipeline]:
    """Build a Translator wired to a recording InferencePipeline with env seeded."""
    monkeypatch.setenv("INFERENCE_PROVIDER", provider)
    monkeypatch.setenv("TEXT_MODEL", "text-model-for-test")
    monkeypatch.setenv("TRANSLATION_MODEL", translation_model)
    pipeline = _RecordingPipeline()
    translator = Translator(inference_pipeline=pipeline)
    return translator, pipeline


def test_translator_vllm_builds_delimiter_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """vllm provider produces the delimiter-format prompt with no system message."""
    translator, pipeline = _make_translator(
        monkeypatch,
        provider="vllm",
        translation_model="Infomaniak-AI/vllm-translategemma-4b-it",
    )

    result = translator.translate(trg_lang="de-DE", text="hello world", src_lang="en")

    assert len(pipeline.calls) == 1
    call = pipeline.calls[0]
    assert call["prompt"] == "<<<source>>>en<<<target>>>de-DE<<<text>>>hello world"
    assert call["include_system_prompt"] is False
    assert call["system_prompt"] is None
    assert call["model"] == "Infomaniak-AI/vllm-translategemma-4b-it"
    assert call["temperature"] == 0.0
    assert result == "translated::<<<source>>>en<<<target>>>de-DE<<<text>>>hello world"


def test_translator_vllm_same_language_short_circuit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same source and target language still short-circuits in vllm mode."""
    translator, pipeline = _make_translator(
        monkeypatch,
        provider="vllm",
        translation_model="Infomaniak-AI/vllm-translategemma-4b-it",
    )

    result = translator.translate(trg_lang="en", text="hello", src_lang="en")

    assert result == "hello"
    assert pipeline.calls == []


def test_translator_ollama_uses_templated_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ollama provider keeps the existing templated prompt + translation system prompt."""
    translator, pipeline = _make_translator(monkeypatch, provider="ollama")

    translator.translate(trg_lang="de-DE", text="hello", src_lang="en")

    assert len(pipeline.calls) == 1
    call = pipeline.calls[0]
    assert "<<<source>>>" not in call["prompt"]
    assert "hello" in call["prompt"]
    assert call["include_system_prompt"] is True
    assert call["system_prompt"] == (
        "You are a precise translation engine. Return only the translation text."
    )
    assert call["temperature"] == 0.0


def test_translator_openai_uses_templated_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """openai provider behaves identically to ollama on the templated path."""
    translator, pipeline = _make_translator(
        monkeypatch, provider="openai", translation_model="gpt-4o"
    )

    translator.translate(trg_lang="de-DE", text="hello", src_lang="en")

    assert len(pipeline.calls) == 1
    call = pipeline.calls[0]
    assert "<<<source>>>" not in call["prompt"]
    assert call["include_system_prompt"] is True
    assert call["model"] == "gpt-4o"


def test_translator_vllm_warns_on_non_translategemma_model_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sanity warning fires once when vllm is configured without a TranslateGemma model."""
    import io

    from loguru import logger

    translator, _ = _make_translator(
        monkeypatch, provider="vllm", translation_model="gpt-4o"
    )

    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING")
    try:
        translator.translate(trg_lang="de-DE", text="hello", src_lang="en")
        translator.translate(trg_lang="de-DE", text="again", src_lang="en")
    finally:
        logger.remove(handler_id)

    warnings = [
        line for line in sink.getvalue().splitlines() if "TranslateGemma" in line
    ]
    assert len(warnings) == 1
