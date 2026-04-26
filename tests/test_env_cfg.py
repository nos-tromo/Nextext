"""Tests for environment-driven configuration helpers."""

import pytest

from nextext.utils.env_cfg import load_transcription_env


def test_ollama_resolves_to_local_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """INFERENCE_PROVIDER=ollama yields the local Whisper provider."""
    monkeypatch.setenv("INFERENCE_PROVIDER", "ollama")
    monkeypatch.delenv("WHISPER_MODEL", raising=False)

    cfg = load_transcription_env()

    assert cfg.provider == "local"
    assert cfg.whisper_model == ""


def test_openai_defaults_to_whisper_1(monkeypatch: pytest.MonkeyPatch) -> None:
    """INFERENCE_PROVIDER=openai falls back to whisper-1 when WHISPER_MODEL is unset."""
    monkeypatch.setenv("INFERENCE_PROVIDER", "openai")
    monkeypatch.delenv("WHISPER_MODEL", raising=False)

    cfg = load_transcription_env()

    assert cfg.provider == "external"
    assert cfg.whisper_model == "whisper-1"


def test_vllm_defaults_to_openai_large_v3(monkeypatch: pytest.MonkeyPatch) -> None:
    """INFERENCE_PROVIDER=vllm falls back to openai/whisper-large-v3 when WHISPER_MODEL is unset."""
    monkeypatch.setenv("INFERENCE_PROVIDER", "vllm")
    monkeypatch.delenv("WHISPER_MODEL", raising=False)

    cfg = load_transcription_env()

    assert cfg.provider == "external"
    assert cfg.whisper_model == "openai/whisper-large-v3"


@pytest.mark.parametrize("provider", ["openai", "vllm"])
def test_explicit_whisper_model_overrides_default(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    """An explicit WHISPER_MODEL overrides the per-provider default."""
    monkeypatch.setenv("INFERENCE_PROVIDER", provider)
    monkeypatch.setenv("WHISPER_MODEL", "custom/model")

    cfg = load_transcription_env()

    assert cfg.provider == "external"
    assert cfg.whisper_model == "custom/model"


def test_blank_whisper_model_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blank WHISPER_MODEL is treated as unset and the default kicks in."""
    monkeypatch.setenv("INFERENCE_PROVIDER", "openai")
    monkeypatch.setenv("WHISPER_MODEL", "   ")

    cfg = load_transcription_env()

    assert cfg.whisper_model == "whisper-1"
