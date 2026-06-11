"""Tests for environment-driven configuration helpers."""

import io

import pytest

from nextext.utils.env_cfg import (
    load_diarization_client_env,
    load_inference_env,
    load_ner_client_env,
    load_whisper_env,
    openai_api_root,
)

_ENDPOINT_ENV_VARS = (
    "OPENAI_API_BASE",
    "OPENAI_API_KEY",
    "WHISPER_API_BASE",
    "WHISPER_API_KEY",
    "WHISPER_MODEL",
    "NER_API_BASE",
    "NER_API_KEY",
    "NER_THRESHOLD",
    "NER_TIMEOUT",
    "DIARIZATION_API_BASE",
    "DIARIZATION_API_KEY",
    "DIARIZATION_TIMEOUT",
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
# openai_api_root
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        ("http://vllm-router:4000/v1", "http://vllm-router:4000"),
        ("http://vllm-router:4000/v1/", "http://vllm-router:4000"),
        ("http://gliner-only:8000", "http://gliner-only:8000"),
        ("https://api.openai.com/v1", "https://api.openai.com"),
        ("http://host/v1/extra", "http://host/v1/extra"),
    ],
)
def test_openai_api_root_strips_one_trailing_v1(monkeypatch: pytest.MonkeyPatch, base: str, expected: str) -> None:
    """Only a trailing /v1 segment is stripped from OPENAI_API_BASE.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
        base (str): The OPENAI_API_BASE value under test.
        expected (str): The expected derived root URL.
    """
    _clear_endpoint_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_BASE", base)

    assert openai_api_root() == expected


def test_openai_api_root_unset_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unset OPENAI_API_BASE derives an empty root.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    _clear_endpoint_env(monkeypatch)

    assert openai_api_root() == ""


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
# load_ner_client_env
# ---------------------------------------------------------------------------


def test_load_ner_client_env_defaults_to_central_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """NER endpoint derives from OPENAI_API_BASE with /v1 stripped.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    _clear_endpoint_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_BASE", "http://vllm-router:4000/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "central-key")

    cfg = load_ner_client_env()

    assert cfg.api_base == "http://vllm-router:4000"
    assert cfg.api_key == "central-key"
    assert cfg.threshold == 0.3
    assert cfg.timeout == 30.0


def test_load_ner_client_env_dedicated_overrides_win(monkeypatch: pytest.MonkeyPatch) -> None:
    """NER_API_BASE / NER_API_KEY / NER_THRESHOLD / NER_TIMEOUT beat the defaults.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    _clear_endpoint_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_BASE", "http://vllm-router:4000/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "central-key")
    monkeypatch.setenv("NER_API_BASE", "http://gliner-only:8000/")
    monkeypatch.setenv("NER_API_KEY", "ner-key")
    monkeypatch.setenv("NER_THRESHOLD", "0.45")
    monkeypatch.setenv("NER_TIMEOUT", "12.5")

    cfg = load_ner_client_env()

    assert cfg.api_base == "http://gliner-only:8000"
    assert cfg.api_key == "ner-key"
    assert cfg.threshold == 0.45
    assert cfg.timeout == 12.5


def test_load_ner_client_env_without_any_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """No NER_API_KEY and no OPENAI_API_KEY resolves to api_key=None.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    _clear_endpoint_env(monkeypatch)
    monkeypatch.setenv("NER_API_BASE", "http://gliner-only:8000")

    cfg = load_ner_client_env()

    assert cfg.api_base == "http://gliner-only:8000"
    assert cfg.api_key is None


# ---------------------------------------------------------------------------
# load_diarization_client_env
# ---------------------------------------------------------------------------


def test_load_diarization_client_env_defaults_to_central_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Diarization endpoint derives from OPENAI_API_BASE with /v1 stripped.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    _clear_endpoint_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_BASE", "http://vllm-router:4000/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "central-key")

    cfg = load_diarization_client_env()

    assert cfg.api_base == "http://vllm-router:4000"
    assert cfg.api_key == "central-key"
    assert cfg.timeout == 600.0


def test_load_diarization_client_env_dedicated_overrides_win(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DIARIZATION_* variables beat the central fallback.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    _clear_endpoint_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_BASE", "http://vllm-router:4000/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "central-key")
    monkeypatch.setenv("DIARIZATION_API_BASE", "http://diarize:8000")
    monkeypatch.setenv("DIARIZATION_API_KEY", "diarize-key")
    monkeypatch.setenv("DIARIZATION_TIMEOUT", "90")

    cfg = load_diarization_client_env()

    assert cfg.api_base == "http://diarize:8000"
    assert cfg.api_key == "diarize-key"
    assert cfg.timeout == 90.0


def test_load_diarization_client_env_unset_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no endpoint variables at all the base is empty and the key None.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    _clear_endpoint_env(monkeypatch)

    cfg = load_diarization_client_env()

    assert cfg.api_base == ""
    assert cfg.api_key is None
