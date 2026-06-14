"""Tests for environment-driven configuration helpers."""

import io

import pytest

from nextext.utils.env_cfg import (
    DEFAULT_DIARIZE_TIMEOUT,
    DEFAULT_NER_TIMEOUT,
    DEFAULT_VAD_TIMEOUT,
    load_diarization_env,
    load_inference_env,
    load_ner_env,
    load_vad_env,
    load_whisper_env,
)

_ENDPOINT_ENV_VARS = (
    "OPENAI_API_BASE",
    "OPENAI_API_KEY",
    "WHISPER_API_BASE",
    "WHISPER_API_KEY",
    "WHISPER_MODEL",
    "NER_API_BASE",
    "NER_TIMEOUT",
    "DIARIZE_API_BASE",
    "DIARIZE_TIMEOUT",
    "VAD_API_BASE",
    "VAD_TIMEOUT",
)


def _clear_endpoint_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every endpoint-related variable so each test starts hermetic.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    for name in _ENDPOINT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


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
# load_whisper_env
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "expected_model"),
    [("openai", "whisper-1"), ("vllm", "openai/whisper-large-v3")],
)
def test_load_whisper_env_falls_back_to_central_endpoint(
    monkeypatch: pytest.MonkeyPatch, provider: str, expected_model: str
) -> None:
    """Without dedicated overrides the central OPENAI_* pair is used.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
        provider (str): The INFERENCE_PROVIDER under test.
        expected_model (str): The per-provider default Whisper model.
    """
    _clear_endpoint_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_PROVIDER", provider)
    monkeypatch.setenv("OPENAI_API_BASE", "http://vllm-router:4000/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "central-key")

    cfg = load_whisper_env()

    assert cfg.api_base == "http://vllm-router:4000/v1"
    assert cfg.api_key == "central-key"
    assert cfg.model == expected_model


def test_load_whisper_env_dedicated_overrides_win(monkeypatch: pytest.MonkeyPatch) -> None:
    """WHISPER_API_BASE / WHISPER_API_KEY / WHISPER_MODEL beat the central pair.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    _clear_endpoint_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_PROVIDER", "vllm")
    monkeypatch.setenv("OPENAI_API_BASE", "http://vllm-router:4000/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "central-key")
    monkeypatch.setenv("WHISPER_API_BASE", "http://audio:8000/v1")
    monkeypatch.setenv("WHISPER_API_KEY", "audio-key")
    monkeypatch.setenv("WHISPER_MODEL", "custom/model")

    cfg = load_whisper_env()

    assert cfg.api_base == "http://audio:8000/v1"
    assert cfg.api_key == "audio-key"
    assert cfg.model == "custom/model"


def test_load_whisper_env_blank_model_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank WHISPER_MODEL is treated as unset for non-ollama providers.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    _clear_endpoint_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_PROVIDER", "openai")
    monkeypatch.setenv("WHISPER_MODEL", "   ")

    cfg = load_whisper_env()

    assert cfg.model == "whisper-1"


def test_load_whisper_env_ollama_with_explicit_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider ollama works when WHISPER_API_BASE and WHISPER_MODEL are set.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    _clear_endpoint_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_PROVIDER", "ollama")
    monkeypatch.setenv("OPENAI_API_KEY", "central-key")
    monkeypatch.setenv("WHISPER_API_BASE", "http://audio:8000/v1")
    monkeypatch.setenv("WHISPER_MODEL", "openai/whisper-large-v3")

    cfg = load_whisper_env()

    assert cfg.api_base == "http://audio:8000/v1"
    assert cfg.api_key == "central-key"
    assert cfg.model == "openai/whisper-large-v3"


@pytest.mark.parametrize(
    ("base", "model", "expected_missing"),
    [
        ("", "openai/whisper-large-v3", ["WHISPER_API_BASE"]),
        ("http://audio:8000/v1", "", ["WHISPER_MODEL"]),
        ("", "", ["WHISPER_API_BASE", "WHISPER_MODEL"]),
    ],
)
def test_load_whisper_env_ollama_requires_explicit_config(
    monkeypatch: pytest.MonkeyPatch,
    base: str,
    model: str,
    expected_missing: list[str],
) -> None:
    """Provider ollama without explicit Whisper config raises an actionable error.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
        base (str): The WHISPER_API_BASE value under test.
        model (str): The WHISPER_MODEL value under test.
        expected_missing (list[str]): Variable names the error must mention.
    """
    _clear_endpoint_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_PROVIDER", "ollama")
    monkeypatch.setenv("OPENAI_API_BASE", "http://ollama:11434/v1")
    if base:
        monkeypatch.setenv("WHISPER_API_BASE", base)
    if model:
        monkeypatch.setenv("WHISPER_MODEL", model)

    with pytest.raises(RuntimeError) as excinfo:
        load_whisper_env()

    for name in expected_missing:
        assert name in str(excinfo.value)


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


# ---------------------------------------------------------------------------
# load_vad_env
# ---------------------------------------------------------------------------


def test_load_vad_env_unset_disables_and_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset VAD_API_BASE disables the guard and uses the default timeout.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    monkeypatch.delenv("VAD_API_BASE", raising=False)
    monkeypatch.delenv("VAD_TIMEOUT", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    cfg = load_vad_env()

    assert cfg.api_base == ""
    assert cfg.api_key == ""
    assert cfg.timeout == DEFAULT_VAD_TIMEOUT


def test_load_vad_env_strips_whitespace_and_trailing_slash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VAD_API_BASE is normalised: surrounding whitespace and trailing '/' removed.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    monkeypatch.setenv("VAD_API_BASE", "  http://vllm-router:7000/  ")

    cfg = load_vad_env()

    assert cfg.api_base == "http://vllm-router:7000"


def test_load_vad_env_reuses_openai_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bearer token is taken from OPENAI_API_KEY (the shared router credential).

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    monkeypatch.setenv("VAD_API_BASE", "http://vllm-router:7000")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")

    cfg = load_vad_env()

    assert cfg.api_key == "sk-secret"


def test_load_vad_env_parses_valid_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid VAD_TIMEOUT is parsed as a float.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    monkeypatch.setenv("VAD_TIMEOUT", "15")

    cfg = load_vad_env()

    assert cfg.timeout == 15.0


@pytest.mark.parametrize("raw_value", ["not-a-number", "0", "-5"])
def test_load_vad_env_invalid_timeout_warns_and_defaults(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str,
) -> None:
    """Non-numeric or non-positive VAD_TIMEOUT warns and falls back to the default.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
        raw_value (str): An invalid timeout token.
    """
    from loguru import logger

    monkeypatch.setenv("VAD_TIMEOUT", raw_value)

    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING")
    try:
        cfg = load_vad_env()
    finally:
        logger.remove(handler_id)

    assert cfg.timeout == DEFAULT_VAD_TIMEOUT
    assert "VAD_TIMEOUT" in sink.getvalue()
