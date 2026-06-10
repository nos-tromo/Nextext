"""Tests for the Translator: one templated path across all providers."""

from typing import Any

import pytest

from nextext.core.openai_cfg import InferencePipeline
from nextext.core.translation import Translator

_TRANSLATION_SYSTEM_PROMPT = "You are a precise translation engine. Return only the translation text."


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
) -> tuple[Translator, _RecordingPipeline]:
    """Build a Translator wired to a recording InferencePipeline with env seeded."""
    monkeypatch.setenv("INFERENCE_PROVIDER", provider)
    monkeypatch.setenv("TEXT_MODEL", "text-model-for-test")
    monkeypatch.delenv("TRANSLATION_MODEL", raising=False)
    pipeline = _RecordingPipeline()
    translator = Translator(inference_pipeline=pipeline)
    return translator, pipeline


@pytest.mark.parametrize("provider", ["ollama", "vllm", "openai"])
def test_translator_uses_templated_prompt_for_all_providers(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    """Every provider uses the templated prompt + translation system prompt on TEXT_MODEL."""
    translator, pipeline = _make_translator(monkeypatch, provider=provider)

    result = translator.translate(trg_lang="de-DE", text="hello world", src_lang="en")

    assert len(pipeline.calls) == 1
    call = pipeline.calls[0]
    assert "<<<source>>>" not in call["prompt"]
    assert "hello world" in call["prompt"]
    assert call["include_system_prompt"] is True
    assert call["system_prompt"] == _TRANSLATION_SYSTEM_PROMPT
    assert call["temperature"] == 0.0
    # ``model`` is left unset, so InferencePipeline.call_model defaults to TEXT_MODEL.
    assert call["model"] is None
    assert result == f"translated::{call['prompt']}"


@pytest.mark.parametrize("provider", ["ollama", "vllm", "openai"])
def test_translator_same_language_short_circuit(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    """Same source and target language short-circuits without calling the model."""
    translator, pipeline = _make_translator(monkeypatch, provider=provider)

    result = translator.translate(trg_lang="en", text="hello", src_lang="en")

    assert result == "hello"
    assert pipeline.calls == []
