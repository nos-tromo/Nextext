"""Tests for environment-driven configuration helpers."""

import io

import pytest

from nextext.utils.env_cfg import (
    DEFAULT_DIARIZE_TIMEOUT,
    DEFAULT_JOB_CONCURRENCY,
    DEFAULT_NER_TIMEOUT,
    DEFAULT_SUMMARY_MAX_INPUT_TOKENS,
    DEFAULT_VAD_TIMEOUT,
    load_diarization_env,
    load_diarize_vad_gate_env,
    load_inference_env,
    load_job_concurrency,
    load_language_env,
    load_ner_env,
    load_sentence_restore_env,
    load_summary_env,
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


def test_load_diarization_env_unset_without_central_disables_and_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With neither DIARIZE_API_BASE nor a central endpoint, diarization is disabled.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    _clear_endpoint_env(monkeypatch)

    cfg = load_diarization_env()

    assert cfg.api_base == ""
    assert cfg.api_key == ""
    assert cfg.timeout == DEFAULT_DIARIZE_TIMEOUT


def test_load_diarization_env_falls_back_to_central_with_v1_stripped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset DIARIZE_API_BASE falls back to OPENAI_API_BASE with /v1 stripped.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    _clear_endpoint_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_BASE", "http://vllm-router:4000/v1")

    cfg = load_diarization_env()

    assert cfg.api_base == "http://vllm-router:4000"


def test_load_diarization_env_dedicated_base_wins_over_central(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A set DIARIZE_API_BASE takes precedence over the central fallback.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    _clear_endpoint_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_BASE", "http://central:4000/v1")
    monkeypatch.setenv("DIARIZE_API_BASE", "http://vllm-router:9000")

    cfg = load_diarization_env()

    assert cfg.api_base == "http://vllm-router:9000"


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


def test_load_ner_env_unset_without_central_disables_and_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With neither NER_API_BASE nor a central endpoint, NER is disabled.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    _clear_endpoint_env(monkeypatch)

    cfg = load_ner_env()

    assert cfg.api_base == ""
    assert cfg.api_key == ""
    assert cfg.timeout == DEFAULT_NER_TIMEOUT


@pytest.mark.parametrize(
    ("central", "expected_root"),
    [
        ("http://vllm-router:4000/v1", "http://vllm-router:4000"),
        ("http://vllm-router:4000/v1/", "http://vllm-router:4000"),
        ("  http://vllm-router:4000/v1  ", "http://vllm-router:4000"),
        ("http://vllm-router:4000", "http://vllm-router:4000"),
        ("https://api.openai.com/v1", "https://api.openai.com"),
    ],
)
def test_load_ner_env_falls_back_to_central_stripping_one_v1(
    monkeypatch: pytest.MonkeyPatch,
    central: str,
    expected_root: str,
) -> None:
    """An unset NER_API_BASE falls back to OPENAI_API_BASE with one trailing /v1 removed.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
        central (str): The OPENAI_API_BASE value under test.
        expected_root (str): The service root the NER client should target.
    """
    _clear_endpoint_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_BASE", central)

    assert load_ner_env().api_base == expected_root


def test_load_ner_env_dedicated_base_wins_and_is_not_v1_stripped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A set NER_API_BASE beats the central fallback and is used verbatim (no /v1 strip).

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    _clear_endpoint_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_BASE", "http://central:4000/v1")
    monkeypatch.setenv("NER_API_BASE", "http://dedicated:4000/v1")

    assert load_ner_env().api_base == "http://dedicated:4000/v1"


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


def test_load_vad_env_unset_without_central_disables_and_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With neither VAD_API_BASE nor a central endpoint, the guard is disabled.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    _clear_endpoint_env(monkeypatch)

    cfg = load_vad_env()

    assert cfg.api_base == ""
    assert cfg.api_key == ""
    assert cfg.timeout == DEFAULT_VAD_TIMEOUT


def test_load_vad_env_falls_back_to_central_with_v1_stripped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset VAD_API_BASE falls back to OPENAI_API_BASE with /v1 stripped (guard on).

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    _clear_endpoint_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_BASE", "http://vllm-router:4000/v1")

    cfg = load_vad_env()

    assert cfg.api_base == "http://vllm-router:4000"


def test_load_vad_env_dedicated_base_wins_over_central(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A set VAD_API_BASE takes precedence over the central fallback.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    _clear_endpoint_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_BASE", "http://central:4000/v1")
    monkeypatch.setenv("VAD_API_BASE", "http://vllm-router:7000")

    cfg = load_vad_env()

    assert cfg.api_base == "http://vllm-router:7000"


@pytest.mark.parametrize("off_token", ["off", "false", "no", "0", "OFF", "  Off  "])
def test_load_vad_env_off_token_disables_even_with_central(
    monkeypatch: pytest.MonkeyPatch,
    off_token: str,
) -> None:
    """An explicit falsy VAD_API_BASE switches the guard off despite a central endpoint.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
        off_token (str): A falsy token recognised as the guard's off switch.
    """
    _clear_endpoint_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_BASE", "http://vllm-router:4000/v1")
    monkeypatch.setenv("VAD_API_BASE", off_token)

    assert load_vad_env().api_base == ""


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


# ---------------------------------------------------------------------------
# load_summary_env
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("raw_value", "expected"), [("8000", 8000), ("3000", 3000), ("  4096  ", 4096)])
def test_load_summary_env_parses_valid_budget(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str,
    expected: int,
) -> None:
    """A valid SUMMARY_MAX_INPUT_TOKENS is parsed as a positive integer.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
        raw_value (str): The raw environment value.
        expected (int): The parsed token budget.
    """
    monkeypatch.setenv("SUMMARY_MAX_INPUT_TOKENS", raw_value)

    cfg = load_summary_env()

    assert cfg.max_input_tokens == expected


def test_load_summary_env_unset_returns_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset SUMMARY_MAX_INPUT_TOKENS falls back to the default budget.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    monkeypatch.delenv("SUMMARY_MAX_INPUT_TOKENS", raising=False)

    cfg = load_summary_env()

    assert cfg.max_input_tokens == DEFAULT_SUMMARY_MAX_INPUT_TOKENS


@pytest.mark.parametrize("raw_value", ["not-a-number", "0", "-5", "1.5"])
def test_load_summary_env_invalid_budget_warns_and_defaults(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str,
) -> None:
    """Non-integer or non-positive SUMMARY_MAX_INPUT_TOKENS warns and falls back to the default.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
        raw_value (str): An invalid budget token.
    """
    from loguru import logger

    monkeypatch.setenv("SUMMARY_MAX_INPUT_TOKENS", raw_value)

    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING")
    try:
        cfg = load_summary_env()
    finally:
        logger.remove(handler_id)

    assert cfg.max_input_tokens == DEFAULT_SUMMARY_MAX_INPUT_TOKENS
    assert "SUMMARY_MAX_INPUT_TOKENS" in sink.getvalue()


# ---------------------------------------------------------------------------
# load_language_env
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [("de", "de"), ("en", "en"), ("  DE  ", "de"), ("EN", "en")],
)
def test_load_language_env_parses_supported(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str,
    expected: str,
) -> None:
    """A supported NEXTEXT_RESPONSE_LANGUAGE is normalised to its language code.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
        raw_value (str): The raw environment value.
        expected (str): The resolved two-letter language code.
    """
    monkeypatch.setenv("NEXTEXT_RESPONSE_LANGUAGE", raw_value)

    cfg = load_language_env()

    assert cfg.code == expected


def test_load_language_env_unset_returns_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset NEXTEXT_RESPONSE_LANGUAGE falls back to English.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    monkeypatch.delenv("NEXTEXT_RESPONSE_LANGUAGE", raising=False)

    cfg = load_language_env()

    assert cfg.code == "en"


def test_load_language_env_blank_returns_default_without_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blank NEXTEXT_RESPONSE_LANGUAGE falls back to English without warning.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    from loguru import logger

    monkeypatch.setenv("NEXTEXT_RESPONSE_LANGUAGE", "   ")

    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING")
    try:
        cfg = load_language_env()
    finally:
        logger.remove(handler_id)

    assert cfg.code == "en"
    assert "NEXTEXT_RESPONSE_LANGUAGE" not in sink.getvalue()


@pytest.mark.parametrize("raw_value", ["fr", "xx", "german"])
def test_load_language_env_invalid_warns_and_defaults(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str,
) -> None:
    """An unsupported NEXTEXT_RESPONSE_LANGUAGE warns and falls back to English.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
        raw_value (str): An unsupported language token.
    """
    from loguru import logger

    monkeypatch.setenv("NEXTEXT_RESPONSE_LANGUAGE", raw_value)

    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING")
    try:
        cfg = load_language_env()
    finally:
        logger.remove(handler_id)

    assert cfg.code == "en"
    assert "NEXTEXT_RESPONSE_LANGUAGE" in sink.getvalue()


# ---------------------------------------------------------------------------
# load_job_concurrency
# ---------------------------------------------------------------------------


def test_load_job_concurrency_unset_returns_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset NEXTEXT_JOB_CONCURRENCY falls back to the serial default (1).

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    monkeypatch.delenv("NEXTEXT_JOB_CONCURRENCY", raising=False)

    assert load_job_concurrency() == DEFAULT_JOB_CONCURRENCY == 1


def test_load_job_concurrency_parses_valid_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NEXTEXT_JOB_CONCURRENCY=3 resolves to 3, enabling parallel job workers.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    monkeypatch.setenv("NEXTEXT_JOB_CONCURRENCY", "3")

    assert load_job_concurrency() == 3


def test_load_job_concurrency_invalid_value_warns_and_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-integer NEXTEXT_JOB_CONCURRENCY warns and falls back to the default.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
    """
    from loguru import logger

    monkeypatch.setenv("NEXTEXT_JOB_CONCURRENCY", "abc")

    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING")
    try:
        result = load_job_concurrency()
    finally:
        logger.remove(handler_id)

    assert result == DEFAULT_JOB_CONCURRENCY
    assert "NEXTEXT_JOB_CONCURRENCY" in sink.getvalue()


@pytest.mark.parametrize("raw_value", ["0", "-2"])
def test_load_job_concurrency_clamps_values_below_one(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str,
) -> None:
    """Values below 1 clamp to 1 with a warning instead of yielding a zero/negative semaphore.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching environment variables.
        raw_value (str): An out-of-range concurrency token.
    """
    from loguru import logger

    monkeypatch.setenv("NEXTEXT_JOB_CONCURRENCY", raw_value)

    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING")
    try:
        result = load_job_concurrency()
    finally:
        logger.remove(handler_id)

    assert result == 1
    assert "NEXTEXT_JOB_CONCURRENCY" in sink.getvalue()


# ---------------------------------------------------------------------------
# load_diarize_vad_gate_env
# ---------------------------------------------------------------------------


def test_vad_gate_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset → gating on with the tuned Silero defaults (threshold 0.4, pad 100ms)."""
    for name in ("NEXTEXT_DIARIZE_VAD_GATE", "VAD_GATE_THRESHOLD", "VAD_GATE_PAD_MS"):
        monkeypatch.delenv(name, raising=False)
    cfg = load_diarize_vad_gate_env()
    assert cfg.enabled is True
    assert cfg.threshold == 0.4
    assert cfg.pad_ms == 100


def test_vad_gate_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """NEXTEXT_DIARIZE_VAD_GATE=off disables gating."""
    monkeypatch.setenv("NEXTEXT_DIARIZE_VAD_GATE", "off")
    assert load_diarize_vad_gate_env().enabled is False


def test_vad_gate_threshold_parsed_and_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Threshold in (0, 1] is honored; out-of-range or invalid falls back to the default."""
    monkeypatch.setenv("VAD_GATE_THRESHOLD", "0.3")
    assert load_diarize_vad_gate_env().threshold == 0.3
    monkeypatch.setenv("VAD_GATE_THRESHOLD", "5")
    assert load_diarize_vad_gate_env().threshold == 0.4
    monkeypatch.setenv("VAD_GATE_THRESHOLD", "abc")
    assert load_diarize_vad_gate_env().threshold == 0.4


def test_vad_gate_pad_ms_parsed_and_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    """pad_ms >= 0 is honored; negative or invalid falls back to the default."""
    monkeypatch.setenv("VAD_GATE_PAD_MS", "200")
    assert load_diarize_vad_gate_env().pad_ms == 200
    monkeypatch.setenv("VAD_GATE_PAD_MS", "-5")
    assert load_diarize_vad_gate_env().pad_ms == 100
    monkeypatch.setenv("VAD_GATE_PAD_MS", "x")
    assert load_diarize_vad_gate_env().pad_ms == 100


def test_vad_gate_unrecognized_toggle_defaults_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unrecognized NEXTEXT_DIARIZE_VAD_GATE value falls back to enabled (default on)."""
    monkeypatch.setenv("NEXTEXT_DIARIZE_VAD_GATE", "banana")
    assert load_diarize_vad_gate_env().enabled is True


def test_vad_gate_threshold_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Threshold 0 is rejected (out of (0, 1]) and falls back; 1 is accepted."""
    monkeypatch.setenv("VAD_GATE_THRESHOLD", "0")
    assert load_diarize_vad_gate_env().threshold == 0.4
    monkeypatch.setenv("VAD_GATE_THRESHOLD", "1")
    assert load_diarize_vad_gate_env().threshold == 1.0


def test_vad_gate_pad_ms_zero_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pad_ms of 0 is valid (>= 0) and honored."""
    monkeypatch.setenv("VAD_GATE_PAD_MS", "0")
    assert load_diarize_vad_gate_env().pad_ms == 0


# ---------------------------------------------------------------------------
# load_sentence_restore_env
# ---------------------------------------------------------------------------


def test_load_sentence_restore_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset vars → enabled with the 0.01 default ratio.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    monkeypatch.delenv("NEXTEXT_SENTENCE_RESTORE", raising=False)
    monkeypatch.delenv("SENTENCE_RESTORE_MIN_PUNCT_RATIO", raising=False)
    cfg = load_sentence_restore_env()
    assert cfg.enabled is True
    assert cfg.min_punct_ratio == 0.01


def test_load_sentence_restore_env_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit falsy token disables restoration.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    monkeypatch.setenv("NEXTEXT_SENTENCE_RESTORE", "off")
    assert load_sentence_restore_env().enabled is False


def test_load_sentence_restore_env_custom_ratio(monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid in-range ratio is honoured.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    monkeypatch.delenv("NEXTEXT_SENTENCE_RESTORE", raising=False)
    monkeypatch.setenv("SENTENCE_RESTORE_MIN_PUNCT_RATIO", "0.03")
    assert load_sentence_restore_env().min_punct_ratio == 0.03


def test_load_sentence_restore_env_invalid_ratio_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Out-of-range / non-numeric ratios warn and fall back to the default.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture for patching
            environment variables.
    """
    monkeypatch.delenv("NEXTEXT_SENTENCE_RESTORE", raising=False)
    monkeypatch.setenv("SENTENCE_RESTORE_MIN_PUNCT_RATIO", "5")
    assert load_sentence_restore_env().min_punct_ratio == 0.01
    monkeypatch.setenv("SENTENCE_RESTORE_MIN_PUNCT_RATIO", "abc")
    assert load_sentence_restore_env().min_punct_ratio == 0.01
