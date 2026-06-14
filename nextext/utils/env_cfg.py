"""Centralized environment-config dataclasses (every env var loader lives here)."""

import os
from dataclasses import dataclass

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
    """Dataclass for Voice Activity Detection configuration.

    Attributes:
        enabled: Whether Silero VAD pre-screening is active.
    """

    enabled: bool


@dataclass(frozen=True)
class DiarizationConfig:
    """Dataclass for the speaker-diarization service configuration.

    Diarization runs out-of-process against an HTTP ``/diarize`` endpoint
    (e.g. ``nos-tromo/vllm-service``); Nextext no longer hosts pyannote.

    Attributes:
        api_base: Root URL of the diarization service (e.g.
            ``http://vllm-router:9000``); the client appends ``/diarize``. An
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
            the client appends ``/gliner``. An empty string disables NER.
        api_key: Bearer token forwarded to the service, reused from
            ``OPENAI_API_KEY``. An empty string sends no ``Authorization`` header.
        timeout: Per-request timeout in seconds for each chunk's ``/gliner`` call.
    """

    api_base: str
    api_key: str
    timeout: float


EXTERNAL_WHISPER_DEFAULTS: dict[str, str] = {
    "openai": "whisper-1",
    "vllm": "openai/whisper-large-v3",
}

DEFAULT_DIARIZE_TIMEOUT: float = 600.0
DEFAULT_NER_TIMEOUT: float = 120.0


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
    """Loads Voice Activity Detection configuration from environment variables.

    Returns:
        VadConfig: Dataclass containing the VAD toggle.
        - enabled (bool): ``True`` (default) when ``VAD_ENABLED`` is ``1``,
          ``true``, or ``yes``.
    """
    enabled = str(os.getenv("VAD_ENABLED", "1")).lower() in {"1", "true", "yes"}
    return VadConfig(enabled=enabled)


def load_diarization_env() -> DiarizationConfig:
    """Loads speaker-diarization service configuration from environment variables.

    Returns:
        DiarizationConfig: Dataclass containing the resolved settings.
        - api_base (str): ``DIARIZE_API_BASE`` with surrounding whitespace and
          any trailing ``/`` removed. Empty (the default) disables diarization,
          so callers skip the HTTP call and emit no speaker labels.
        - api_key (str): ``OPENAI_API_KEY`` (the diarization service shares the
          inference router's credentials). Empty sends no bearer token.
        - timeout (float): ``DIARIZE_TIMEOUT`` seconds. Defaults to
          :data:`DEFAULT_DIARIZE_TIMEOUT`; non-numeric or non-positive values
          warn and fall back to the default.
    """
    api_base = os.getenv("DIARIZE_API_BASE", "").strip().rstrip("/")
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
        - api_base (str): ``NER_API_BASE`` with surrounding whitespace and any
          trailing ``/`` removed. Empty (the default) disables NER, so callers
          skip the HTTP call and emit no entities.
        - api_key (str): ``OPENAI_API_KEY`` (the NER service shares the inference
          router's credentials). Empty sends no bearer token.
        - timeout (float): ``NER_TIMEOUT`` seconds. Defaults to
          :data:`DEFAULT_NER_TIMEOUT`; non-numeric or non-positive values warn
          and fall back to the default.
    """
    api_base = os.getenv("NER_API_BASE", "").strip().rstrip("/")
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
