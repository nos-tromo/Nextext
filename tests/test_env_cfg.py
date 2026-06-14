"""Tests for environment-driven configuration helpers."""

import io

import pytest

from nextext.utils.env_cfg import (
    DEFAULT_DIARIZE_TIMEOUT,
    DEFAULT_NER_TIMEOUT,
    load_diarization_env,
    load_inference_env,
    load_ner_env,
    load_transcription_env,
)


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
def test_explicit_whisper_model_overrides_default(monkeypatch: pytest.MonkeyPatch, provider: str) -> None:
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


# ---------------------------------------------------------------------------
# OLLAMA_THINK / _parse_tristate_bool via load_inference_env
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_value",
    ["1", "true", "yes", "on", "True", " YES "],
)
def test_load_inference_env_parses_ollama_think_truthy(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str,
) -> None:
    """All recognised truthy tokens for OLLAMA_THINK resolve to think=True.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
        raw_value (str): A truthy token to set as OLLAMA_THINK.
    """
    monkeypatch.setenv("OLLAMA_THINK", raw_value)

    cfg = load_inference_env()

    assert cfg.think is True


@pytest.mark.parametrize(
    "raw_value",
    ["0", "false", "no", "off", "FALSE"],
)
def test_load_inference_env_parses_ollama_think_falsy(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str,
) -> None:
    """All recognised falsy tokens for OLLAMA_THINK resolve to think=False.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
        raw_value (str): A falsy token to set as OLLAMA_THINK.
    """
    monkeypatch.setenv("OLLAMA_THINK", raw_value)

    cfg = load_inference_env()

    assert cfg.think is False


def test_load_inference_env_ollama_think_unset_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset OLLAMA_THINK must produce think=None in InferenceConfig.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    monkeypatch.delenv("OLLAMA_THINK", raising=False)

    cfg = load_inference_env()

    assert cfg.think is None


def test_load_inference_env_ollama_think_invalid_warns_and_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unrecognised OLLAMA_THINK value emits a warning and resolves to None.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    from loguru import logger

    monkeypatch.setenv("OLLAMA_THINK", "maybe")

    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING")
    try:
        cfg = load_inference_env()
    finally:
        logger.remove(handler_id)

    assert cfg.think is None
    log_output = sink.getvalue()
    assert "OLLAMA_THINK" in log_output
    assert "maybe" not in log_output


# ---------------------------------------------------------------------------
# load_diarization_env
# ---------------------------------------------------------------------------


def test_load_diarization_env_unset_disables_and_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset DIARIZE_API_BASE disables diarization and uses the default timeout.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    monkeypatch.delenv("DIARIZE_API_BASE", raising=False)
    monkeypatch.delenv("DIARIZE_TIMEOUT", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    cfg = load_diarization_env()

    assert cfg.api_base == ""
    assert cfg.api_key == ""
    assert cfg.timeout == DEFAULT_DIARIZE_TIMEOUT


def test_load_diarization_env_strips_whitespace_and_trailing_slash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DIARIZE_API_BASE is normalised: surrounding whitespace and trailing '/' removed.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    monkeypatch.setenv("DIARIZE_API_BASE", "  http://vllm-router:9000/  ")

    cfg = load_diarization_env()

    assert cfg.api_base == "http://vllm-router:9000"


def test_load_diarization_env_reuses_openai_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bearer token is taken from OPENAI_API_KEY (the shared router credential).

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    monkeypatch.setenv("DIARIZE_API_BASE", "http://vllm-router:9000")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")

    cfg = load_diarization_env()

    assert cfg.api_key == "sk-secret"


def test_load_diarization_env_parses_valid_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid DIARIZE_TIMEOUT is parsed as a float.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    monkeypatch.setenv("DIARIZE_TIMEOUT", "120")

    cfg = load_diarization_env()

    assert cfg.timeout == 120.0


@pytest.mark.parametrize("raw_value", ["not-a-number", "0", "-5"])
def test_load_diarization_env_invalid_timeout_warns_and_defaults(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str,
) -> None:
    """Non-numeric or non-positive DIARIZE_TIMEOUT warns and falls back to the default.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
        raw_value (str): An invalid timeout token.
    """
    from loguru import logger

    monkeypatch.setenv("DIARIZE_TIMEOUT", raw_value)

    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING")
    try:
        cfg = load_diarization_env()
    finally:
        logger.remove(handler_id)

    assert cfg.timeout == DEFAULT_DIARIZE_TIMEOUT
    assert "DIARIZE_TIMEOUT" in sink.getvalue()


# ---------------------------------------------------------------------------
# load_ner_env
# ---------------------------------------------------------------------------


def test_load_ner_env_unset_disables_and_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset NER_API_BASE disables NER and uses the default timeout.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    monkeypatch.delenv("NER_API_BASE", raising=False)
    monkeypatch.delenv("NER_TIMEOUT", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    cfg = load_ner_env()

    assert cfg.api_base == ""
    assert cfg.api_key == ""
    assert cfg.timeout == DEFAULT_NER_TIMEOUT


def test_load_ner_env_strips_whitespace_and_trailing_slash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NER_API_BASE is normalised: surrounding whitespace and trailing '/' removed.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    monkeypatch.setenv("NER_API_BASE", "  http://vllm-router:4000/  ")

    cfg = load_ner_env()

    assert cfg.api_base == "http://vllm-router:4000"


def test_load_ner_env_reuses_openai_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bearer token is taken from OPENAI_API_KEY (the shared router credential).

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    monkeypatch.setenv("NER_API_BASE", "http://vllm-router:4000")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")

    cfg = load_ner_env()

    assert cfg.api_key == "sk-secret"


def test_load_ner_env_parses_valid_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid NER_TIMEOUT is parsed as a float.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    monkeypatch.setenv("NER_TIMEOUT", "45")

    cfg = load_ner_env()

    assert cfg.timeout == 45.0


@pytest.mark.parametrize("raw_value", ["not-a-number", "0", "-5"])
def test_load_ner_env_invalid_timeout_warns_and_defaults(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str,
) -> None:
    """Non-numeric or non-positive NER_TIMEOUT warns and falls back to the default.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
        raw_value (str): An invalid timeout token.
    """
    from loguru import logger

    monkeypatch.setenv("NER_TIMEOUT", raw_value)

    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING")
    try:
        cfg = load_ner_env()
    finally:
        logger.remove(handler_id)

    assert cfg.timeout == DEFAULT_NER_TIMEOUT
    assert "NER_TIMEOUT" in sink.getvalue()
