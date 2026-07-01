"""Centralized environment-config dataclasses (every env var loader lives here)."""

import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def is_offline() -> bool:
    """Report whether Nextext runs in offline mode (``NEXTEXT_OFFLINE``).

    Offline mode gates the spaCy / NLTK resource downloads — the only
    runtime downloads Nextext still performs; all model inference happens on
    external endpoints. Defaults to offline (``"1"``) so airgapped
    deployments are safe out of the box.

    Returns:
        bool: ``True`` when offline mode is active.
    """
    return str(os.getenv("NEXTEXT_OFFLINE", "1")).lower() in {"1", "true", "yes"}


VALID_INFERENCE_PROVIDERS: frozenset[str] = frozenset({"ollama", "vllm", "openai"})
_TRUE_TOKENS: frozenset[str] = frozenset({"1", "true", "yes", "on"})
_FALSE_TOKENS: frozenset[str] = frozenset({"0", "false", "no", "off"})

DEFAULT_TARGET_LANG: str = "en"


def load_default_target_lang() -> str:
    """Loads the configured default target translation language code.

    The value seeds the frontend's "Target language" dropdown on a fresh
    browser (one without a persisted preference). The ``GET /languages`` route
    validates the result against the supported target mapping and falls back to
    ``"en"`` (then the first supported code) when it is not a recognised code.

    Returns:
        str: The ``NEXTEXT_DEFAULT_TARGET_LANG`` value (stripped), or
            :data:`DEFAULT_TARGET_LANG` (``"en"``) when unset or blank.
    """
    raw = os.getenv("NEXTEXT_DEFAULT_TARGET_LANG", "").strip()
    return raw or DEFAULT_TARGET_LANG


PROMPT_SUPPORTED_LANGUAGES: tuple[str, ...] = ("en", "de")
DEFAULT_PROMPT_LANGUAGE: str = "en"


@dataclass(frozen=True)
class LanguageConfig:
    """Dataclass for the LLM response-language configuration.

    Attributes:
        code: Two-letter language code selecting the localized prompt
            subdirectory (``nextext/utils/prompts/<code>/``) used for the
            system, summary, and hate-speech prompts. One of
            :data:`PROMPT_SUPPORTED_LANGUAGES`.
    """

    code: Literal["en", "de"]


def load_language_env(default: str = DEFAULT_PROMPT_LANGUAGE) -> LanguageConfig:
    """Loads the LLM response-language setting from ``NEXTEXT_RESPONSE_LANGUAGE``.

    The value selects which localized prompt subdirectory
    (``nextext/utils/prompts/<code>/``) supplies the system, summary, and
    hate-speech prompts, so it governs the language of generated summaries and
    hate-speech rationales independently of the transcription source language or
    the translation target. Translation output is unaffected — it is driven by
    explicit source/target language codes. Unrecognised values warn and fall
    back to ``default`` (then ``"en"``) so a typo cannot break bring-up.

    Args:
        default: Fallback language code used when ``NEXTEXT_RESPONSE_LANGUAGE``
            is unset, blank, or invalid. Defaults to
            :data:`DEFAULT_PROMPT_LANGUAGE` (``"en"``).

    Returns:
        LanguageConfig: Dataclass carrying the resolved two-letter ``code``.
    """
    raw = os.getenv("NEXTEXT_RESPONSE_LANGUAGE")
    candidate = (raw if raw is not None else default).strip().lower()
    if candidate not in PROMPT_SUPPORTED_LANGUAGES:
        if raw is not None and raw.strip():
            logger.warning(
                "Unknown NEXTEXT_RESPONSE_LANGUAGE '{}'. Falling back to '{}'.",
                raw,
                default,
            )
        candidate = default.strip().lower()
        if candidate not in PROMPT_SUPPORTED_LANGUAGES:
            candidate = "en"
    code: Literal["en", "de"] = "de" if candidate == "de" else "en"
    return LanguageConfig(code=code)


@dataclass(frozen=True)
class InferenceConfig:
    """Dataclass for inference provider configuration.

    Attributes:
        provider: One of :data:`VALID_INFERENCE_PROVIDERS`.
        think: Tri-state default for the Ollama ``think`` request field. ``None``
            means the field is omitted from outgoing requests (model default);
            ``True``/``False`` is forwarded via ``extra_body`` and is a no-op for
            vLLM and OpenAI providers.
    """

    provider: str
    think: bool | None = None


@dataclass(frozen=True)
class VadConfig:
    """Dataclass for the voice-activity-detection service configuration.

    VAD runs out-of-process against an HTTP ``/vad`` endpoint (e.g.
    ``nos-tromo/vllm-service``); Nextext no longer hosts a local Silero model.

    Attributes:
        api_base: Root URL of the VAD service (e.g. ``http://vllm-router:7000``);
            the client appends ``/vad``. Falls back to the central endpoint when
            ``VAD_API_BASE`` is unset; an explicit falsy ``VAD_API_BASE`` switches
            the guard off. An empty string disables the speech guard, so every
            file is transcribed.
        api_key: Bearer token forwarded to the service, reused from
            ``OPENAI_API_KEY``. An empty string sends no ``Authorization`` header.
        timeout: Per-request timeout in seconds for the ``/vad`` call.
    """

    api_base: str
    api_key: str
    timeout: float


@dataclass(frozen=True)
class DiarizationConfig:
    """Dataclass for the speaker-diarization service configuration.

    Diarization runs out-of-process against an HTTP ``/diarize`` endpoint
    (e.g. ``nos-tromo/vllm-service``); Nextext no longer hosts pyannote.

    Attributes:
        api_base: Root URL of the diarization service (e.g.
            ``http://vllm-router:9000``); the client appends ``/diarize``. Falls
            back to the central endpoint when ``DIARIZE_API_BASE`` is unset. An
            empty string disables diarization entirely.
        api_key: Bearer token forwarded to the service, reused from
            ``OPENAI_API_KEY``. An empty string sends no ``Authorization``
            header (the keyless ``diarize-only`` backend ignores it anyway).
        timeout: Per-request timeout in seconds for the diarization call,
            which can run for minutes on long or CPU-served audio.
    """

    api_base: str
    api_key: str
    timeout: float


@dataclass(frozen=True)
class NerConfig:
    """Dataclass for the named-entity-recognition service configuration.

    NER runs out-of-process against an HTTP ``/gliner`` endpoint (e.g.
    ``nos-tromo/vllm-service`` behind the LiteLLM router); Nextext no longer
    hosts GLiNER in-process.

    Attributes:
        api_base: Root URL of the NER service (e.g. ``http://vllm-router:4000``);
            the client appends ``/gliner``. Falls back to the central endpoint
            when ``NER_API_BASE`` is unset. An empty string disables NER.
        api_key: Bearer token forwarded to the service, reused from
            ``OPENAI_API_KEY``. An empty string sends no ``Authorization`` header.
        timeout: Per-request timeout in seconds for each chunk's ``/gliner`` call.
    """

    api_base: str
    api_key: str
    timeout: float


@dataclass(frozen=True)
class SummaryConfig:
    """Dataclass for the summarization stage's context budget.

    Attributes:
        max_input_tokens: Upper bound on how many tokens of transcript text are
            sent to the chat model in a single summarize request. The
            map-reduce summarizer in :mod:`nextext.pipeline` splits any
            transcript whose estimated size exceeds this budget into chunks,
            summarizes each, then summarizes the combined partial summaries, so
            no single request can overflow the model's context window. Lower it
            for token-dense scripts (e.g. CJK) or small ``max_model_len``
            backends.
    """

    max_input_tokens: int


EXTERNAL_WHISPER_DEFAULTS: dict[str, str] = {
    "openai": "whisper-1",
    "vllm": "openai/whisper-large-v3",
}

DEFAULT_DIARIZE_TIMEOUT: float = 600.0
DEFAULT_NER_TIMEOUT: float = 120.0
DEFAULT_VAD_TIMEOUT: float = 60.0
DEFAULT_SUMMARY_MAX_INPUT_TOKENS: int = 6000
DEFAULT_KEYFRAMES_PER_MINUTE: int = 4
DEFAULT_KEYFRAMES_MAX: int = 20
KEYFRAMES_MAX_CEILING: int = 200


def _parse_tristate_bool(name: str) -> bool | None:
    """Parses a tri-state boolean environment variable.

    Args:
        name: Environment variable name to read.

    Returns:
        ``True`` for ``1``/``true``/``yes``/``on``, ``False`` for
        ``0``/``false``/``no``/``off``, and ``None`` when unset, empty, or
        unrecognised. Unrecognised values emit a ``logger.warning`` and resolve
        to ``None`` so callers fall back to the model default.
    """
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return None
    if raw in _TRUE_TOKENS:
        return True
    if raw in _FALSE_TOKENS:
        return False
    logger.warning("Unknown {} value (length={}). Ignoring.", name, len(raw))
    return None


def _central_endpoint_root() -> str:
    """Normalise ``OPENAI_API_BASE`` to a service root for path-appending clients.

    The central endpoint is an OpenAI-SDK base that conventionally ends in
    ``/v1`` (e.g. ``http://vllm-router:4000/v1``). The out-of-process ``/gliner``,
    ``/diarize``, and ``/vad`` services sit at the service root alongside it, so a
    single trailing ``/v1`` segment is removed before a client appends its own
    path. Surrounding whitespace and any trailing ``/`` are stripped too.

    Returns:
        str: The central endpoint's service root (e.g. ``http://vllm-router:4000``),
            or ``""`` when ``OPENAI_API_BASE`` is unset or blank.
    """
    base = os.getenv("OPENAI_API_BASE", "").strip().rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3].rstrip("/")
    return base


def load_inference_env() -> InferenceConfig:
    """Loads inference provider configuration from environment variables.

    Returns:
        InferenceConfig: Dataclass containing the resolved provider.
        - provider (str): One of ``ollama`` (default), ``vllm``, or ``openai``.
            Unknown values fall back to ``ollama`` with a warning.
        - think (bool | None): Resolved from ``OLLAMA_THINK``. ``None`` when
            unset/invalid, otherwise the parsed boolean. Forwarded by
            ``InferencePipeline.call_model`` via ``extra_body`` for Ollama; a
            no-op for vLLM/OpenAI providers.
    """
    raw = os.getenv("INFERENCE_PROVIDER", "ollama").strip().lower()
    if raw not in VALID_INFERENCE_PROVIDERS:
        logger.warning("Unknown INFERENCE_PROVIDER '{}'. Falling back to 'ollama'.", raw)
        raw = "ollama"
    return InferenceConfig(provider=raw, think=_parse_tristate_bool("OLLAMA_THINK"))


def load_vad_env() -> VadConfig:
    """Loads voice-activity-detection service configuration from environment variables.

    Returns:
        VadConfig: Dataclass containing the resolved settings.
        - api_base (str): The VAD service root, resolved three ways from
          ``VAD_API_BASE``: a URL is used as-is (trailing ``/`` removed); an
          explicit falsy token (``off``/``false``/``no``/``0``) switches the
          guard off (empty string); unset/blank falls back to the central
          endpoint (:func:`_central_endpoint_root`). An empty result disables the
          guard, so callers skip the HTTP call and transcribe every file.
        - api_key (str): ``OPENAI_API_KEY`` (the VAD service shares the inference
          router's credentials). Empty sends no bearer token.
        - timeout (float): ``VAD_TIMEOUT`` seconds. Defaults to
          :data:`DEFAULT_VAD_TIMEOUT`; non-numeric or non-positive values warn
          and fall back to the default.
    """
    raw_base = os.getenv("VAD_API_BASE", "").strip()
    if raw_base.lower() in _FALSE_TOKENS:
        api_base = ""
    elif raw_base:
        api_base = raw_base.rstrip("/")
    else:
        api_base = _central_endpoint_root()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    timeout = DEFAULT_VAD_TIMEOUT
    raw_timeout = os.getenv("VAD_TIMEOUT", "").strip()
    if raw_timeout:
        try:
            parsed = float(raw_timeout)
            if parsed <= 0:
                raise ValueError
            timeout = parsed
        except ValueError:
            logger.warning(
                "Invalid VAD_TIMEOUT '{}'. Falling back to {}s.",
                raw_timeout,
                DEFAULT_VAD_TIMEOUT,
            )

    return VadConfig(api_base=api_base, api_key=api_key, timeout=timeout)


def load_diarization_env() -> DiarizationConfig:
    """Loads speaker-diarization service configuration from environment variables.

    Returns:
        DiarizationConfig: Dataclass containing the resolved settings.
        - api_base (str): ``DIARIZE_API_BASE`` (a service root, trailing ``/``
          removed) when set; otherwise it falls back to the central endpoint
          (:func:`_central_endpoint_root`). Empty — no dedicated override and no
          central endpoint — disables diarization, so callers skip the HTTP call
          and emit no speaker labels.
        - api_key (str): ``OPENAI_API_KEY`` (the diarization service shares the
          inference router's credentials). Empty sends no bearer token.
        - timeout (float): ``DIARIZE_TIMEOUT`` seconds. Defaults to
          :data:`DEFAULT_DIARIZE_TIMEOUT`; non-numeric or non-positive values
          warn and fall back to the default.
    """
    api_base = os.getenv("DIARIZE_API_BASE", "").strip().rstrip("/") or _central_endpoint_root()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    timeout = DEFAULT_DIARIZE_TIMEOUT
    raw_timeout = os.getenv("DIARIZE_TIMEOUT", "").strip()
    if raw_timeout:
        try:
            parsed = float(raw_timeout)
            if parsed <= 0:
                raise ValueError
            timeout = parsed
        except ValueError:
            logger.warning(
                "Invalid DIARIZE_TIMEOUT '{}'. Falling back to {}s.",
                raw_timeout,
                DEFAULT_DIARIZE_TIMEOUT,
            )

    return DiarizationConfig(api_base=api_base, api_key=api_key, timeout=timeout)


def load_ner_env() -> NerConfig:
    """Loads named-entity-recognition service configuration from environment variables.

    Returns:
        NerConfig: Dataclass containing the resolved settings.
        - api_base (str): ``NER_API_BASE`` (a service root, trailing ``/``
          removed) when set; otherwise it falls back to the central endpoint
          (:func:`_central_endpoint_root`). Empty — no dedicated override and no
          central endpoint — disables NER, so callers skip the HTTP call and emit
          no entities.
        - api_key (str): ``OPENAI_API_KEY`` (the NER service shares the inference
          router's credentials). Empty sends no bearer token.
        - timeout (float): ``NER_TIMEOUT`` seconds. Defaults to
          :data:`DEFAULT_NER_TIMEOUT`; non-numeric or non-positive values warn
          and fall back to the default.
    """
    api_base = os.getenv("NER_API_BASE", "").strip().rstrip("/") or _central_endpoint_root()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    timeout = DEFAULT_NER_TIMEOUT
    raw_timeout = os.getenv("NER_TIMEOUT", "").strip()
    if raw_timeout:
        try:
            parsed = float(raw_timeout)
            if parsed <= 0:
                raise ValueError
            timeout = parsed
        except ValueError:
            logger.warning(
                "Invalid NER_TIMEOUT '{}'. Falling back to {}s.",
                raw_timeout,
                DEFAULT_NER_TIMEOUT,
            )

    return NerConfig(api_base=api_base, api_key=api_key, timeout=timeout)


def load_summary_env() -> SummaryConfig:
    """Loads the summarization context-budget configuration from the environment.

    The map-reduce summarizer in :mod:`nextext.pipeline` uses this budget to
    decide how much transcript text fits in one request before it must split
    and summarize hierarchically.

    Returns:
        SummaryConfig: Dataclass containing the resolved settings.
        - max_input_tokens (int): ``SUMMARY_MAX_INPUT_TOKENS`` parsed as a
          positive integer. Defaults to :data:`DEFAULT_SUMMARY_MAX_INPUT_TOKENS`;
          non-integer or non-positive values warn and fall back to the default.
    """
    max_input_tokens = DEFAULT_SUMMARY_MAX_INPUT_TOKENS
    raw_budget = os.getenv("SUMMARY_MAX_INPUT_TOKENS", "").strip()
    if raw_budget:
        try:
            parsed = int(raw_budget)
            if parsed <= 0:
                raise ValueError
            max_input_tokens = parsed
        except ValueError:
            logger.warning(
                "Invalid SUMMARY_MAX_INPUT_TOKENS '{}'. Falling back to {}.",
                raw_budget,
                DEFAULT_SUMMARY_MAX_INPUT_TOKENS,
            )

    return SummaryConfig(max_input_tokens=max_input_tokens)


@dataclass(frozen=True)
class KeyframeDefaults:
    """Operator-set default keyframe sampling, applied when a job omits the fields.

    Attributes:
        per_minute: Default ``JobOptions.keyframes_per_minute`` used when a job
            creation request omits the field.
        max_frames: Default ``JobOptions.keyframes_max`` used when a job
            creation request omits the field.
    """

    per_minute: int
    max_frames: int


def load_keyframe_defaults(
    default_per_minute: int = DEFAULT_KEYFRAMES_PER_MINUTE,
    default_max_frames: int = DEFAULT_KEYFRAMES_MAX,
) -> KeyframeDefaults:
    """Loads operator-configured default keyframe sampling from the environment.

    Reads ``KEYFRAMES_PER_MINUTE`` / ``KEYFRAMES_MAX`` so an operator can set
    the server's default keyframe "size" without a code change; both are
    consulted only by ``JobOptions``' ``default_factory``, so an explicit
    per-request value always overrides them.

    Robust to bad input: unparseable values fall back to the given default
    (with a warning); out-of-range values are clamped rather than raising
    (also with a warning) — so a misconfigured environment can never turn into
    a validation error on every job. ``per_minute`` is floored at 0;
    ``max_frames`` is clamped to ``[0, 200]``, the ``JobOptions`` hard cap.

    Args:
        default_per_minute: Fallback used when ``KEYFRAMES_PER_MINUTE`` is
            unset or unparseable. Defaults to
            :data:`DEFAULT_KEYFRAMES_PER_MINUTE` (``4``).
        default_max_frames: Fallback used when ``KEYFRAMES_MAX`` is unset or
            unparseable. Defaults to :data:`DEFAULT_KEYFRAMES_MAX` (``20``).

    Returns:
        KeyframeDefaults: Dataclass containing the resolved, clamped defaults.
    """
    raw_per_minute = os.getenv("KEYFRAMES_PER_MINUTE", "").strip()
    try:
        parsed_per_minute = int(raw_per_minute) if raw_per_minute else default_per_minute
    except ValueError:
        logger.warning(
            "Invalid KEYFRAMES_PER_MINUTE '{}'. Falling back to {}.",
            raw_per_minute,
            default_per_minute,
        )
        parsed_per_minute = default_per_minute
    per_minute = max(0, parsed_per_minute)
    if per_minute != parsed_per_minute:
        logger.warning(
            "KEYFRAMES_PER_MINUTE {} is negative. Clamping to {}.",
            parsed_per_minute,
            per_minute,
        )

    raw_max_frames = os.getenv("KEYFRAMES_MAX", "").strip()
    try:
        parsed_max_frames = int(raw_max_frames) if raw_max_frames else default_max_frames
    except ValueError:
        logger.warning(
            "Invalid KEYFRAMES_MAX '{}'. Falling back to {}.",
            raw_max_frames,
            default_max_frames,
        )
        parsed_max_frames = default_max_frames
    max_frames = min(KEYFRAMES_MAX_CEILING, max(0, parsed_max_frames))
    if max_frames != parsed_max_frames:
        logger.warning(
            "KEYFRAMES_MAX {} outside [0, {}]. Clamping to {}.",
            parsed_max_frames,
            KEYFRAMES_MAX_CEILING,
            max_frames,
        )

    return KeyframeDefaults(per_minute=per_minute, max_frames=max_frames)


@dataclass(frozen=True)
class WhisperClientConfig:
    """Dataclass for the external Whisper transcription client.

    Attributes:
        api_base: OpenAI-SDK base URL (including ``/v1``) of the endpoint
            serving ``/audio/transcriptions``. Dedicated override via
            ``WHISPER_API_BASE``; falls back to the central
            ``OPENAI_API_BASE``. An empty string lets the SDK use its
            built-in default (``https://api.openai.com/v1``).
        api_key: Bearer token for the Whisper endpoint. Dedicated override
            via ``WHISPER_API_KEY``; falls back to ``OPENAI_API_KEY``.
        model: Model name sent with each request. ``WHISPER_MODEL``
            overrides the per-provider default.
    """

    api_base: str
    api_key: str
    model: str


def load_whisper_env() -> WhisperClientConfig:
    """Loads external Whisper client configuration from environment variables.

    Every transcription call goes to an OpenAI-compatible audio endpoint —
    there is no local fallback. The endpoint defaults to the central
    ``OPENAI_API_BASE``/``OPENAI_API_KEY`` pair and can be re-pointed via the
    dedicated ``WHISPER_API_BASE``/``WHISPER_API_KEY`` overrides. The model
    defaults per provider (``openai`` → ``whisper-1``, ``vllm`` →
    ``openai/whisper-large-v3``); ``WHISPER_MODEL`` overrides.

    ``INFERENCE_PROVIDER=ollama`` serves no transcription API, so both
    ``WHISPER_API_BASE`` and ``WHISPER_MODEL`` must be set explicitly there
    (e.g. pointing at the vllm-service audio container or any standalone
    OpenAI-compatible STT server).

    Returns:
        WhisperClientConfig: The resolved client configuration.

    Raises:
        RuntimeError: When ``INFERENCE_PROVIDER=ollama`` and
            ``WHISPER_API_BASE`` or ``WHISPER_MODEL`` is unset or blank.
    """
    provider = load_inference_env().provider
    dedicated_base = os.getenv("WHISPER_API_BASE", "").strip()
    api_key = os.getenv("WHISPER_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    model_override = os.getenv("WHISPER_MODEL", "").strip()

    if provider == "ollama":
        missing = [
            name
            for name, value in (
                ("WHISPER_API_BASE", dedicated_base),
                ("WHISPER_MODEL", model_override),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "INFERENCE_PROVIDER=ollama has no Whisper endpoint. Set "
                + " and ".join(missing)
                + " to an OpenAI-compatible transcription service"
                " (e.g. the vllm-service audio container)."
            )
        return WhisperClientConfig(api_base=dedicated_base, api_key=api_key, model=model_override)

    api_base = dedicated_base or os.getenv("OPENAI_API_BASE", "").strip()
    model = model_override or EXTERNAL_WHISPER_DEFAULTS[provider]
    return WhisperClientConfig(api_base=api_base, api_key=api_key, model=model)


@dataclass(frozen=True)
class PrincipalConfig:
    """Dataclass for request-principal resolution configuration.

    Attributes:
        header_name: Trusted request header carrying the authenticated
            principal (the caller's owner identifier).
        default_identity: Developer / header-less fallback identity used when
            the trusted header is absent, or ``None`` when unset (the resolver
            then fails closed with HTTP 401).
    """

    header_name: str
    default_identity: str | None


def load_principal_env(
    default_header_name: str = "X-Auth-User",
    default_identity: str | None = None,
) -> PrincipalConfig:
    """Loads request-principal configuration from environment variables.

    The backend has no real authentication: identity is whatever the trusted
    header carries, with an optional env-var fallback so developers (and any
    header-less client) resolve to a single configured identity instead of
    being rejected. The seam is intentionally thin — swapping the header read
    for a verified-token read later leaves ownership queries and routes
    untouched.

    Args:
        default_header_name (str): Default trusted header carrying the
            authenticated principal when ``NEXTEXT_AUTH_HEADER`` is unset.
        default_identity (str | None): Default fallback identity used when
            ``NEXTEXT_DEFAULT_IDENTITY`` is unset.

    Returns:
        PrincipalConfig: Dataclass containing principal configuration.
        - header_name (str): The trusted header carrying the principal.
        - default_identity (str | None): Fallback identity, or ``None`` when
          unset (the resolver then fails closed with HTTP 401).
    """
    raw_identity = os.getenv("NEXTEXT_DEFAULT_IDENTITY")
    if raw_identity is not None and raw_identity.strip():
        resolved_identity: str | None = raw_identity.strip()
    else:
        resolved_identity = default_identity

    return PrincipalConfig(
        header_name=os.getenv("NEXTEXT_AUTH_HEADER", default_header_name),
        default_identity=resolved_identity,
    )
