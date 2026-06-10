"""Centralized environment-config dataclasses (every env var loader lives here)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def set_offline_env() -> None:
    """Log the current offline mode status.

    The actual env vars (HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE, etc.) are set at
    module level immediately after ``load_dotenv()`` so they are available before
    ``huggingface_hub`` / ``transformers`` cache their values at import time.
    This function re-applies them (idempotent) and emits a log message.
    """
    if str(os.getenv("NEXTEXT_OFFLINE", "1")).lower() in {"1", "true", "yes"}:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        logger.info("Set Hugging Face libraries to offline mode.")
    else:
        logger.info("Hugging Face libraries are in online mode.")

    # Apple Silicon: route MPS-unsupported ops (e.g. sparse_coo paths used
    # by Whisper / pyannote) to CPU instead of crashing.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


set_offline_env()  # Apply offline settings at module load time


@dataclass(frozen=True)
class PathConfig:
    """Dataclass for path configuration."""

    prompts: Path
    hf_hub_cache: Path


@dataclass(frozen=True)
class TranscriptionConfig:
    """Dataclass for transcription provider configuration."""

    provider: str
    whisper_model: str


VALID_INFERENCE_PROVIDERS: frozenset[str] = frozenset({"ollama", "vllm", "openai"})
VALID_RESIDENCY_STRATEGIES: frozenset[str] = frozenset({"offload", "evict"})
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
class MemoryConfig:
    """Dataclass for model residency configuration.

    Attributes:
        default_strategy: Global fallback strategy applied when no per-model
            override is set. One of :data:`VALID_RESIDENCY_STRATEGIES`.
    """

    default_strategy: str


@dataclass(frozen=True)
class VadConfig:
    """Dataclass for Voice Activity Detection configuration.

    Attributes:
        enabled: Whether Silero VAD pre-screening is active.
    """

    enabled: bool


EXTERNAL_WHISPER_DEFAULTS: dict[str, str] = {
    "openai": "whisper-1",
    "vllm": "openai/whisper-large-v3",
}


def load_transcription_env() -> TranscriptionConfig:
    """Loads transcription provider configuration derived from ``INFERENCE_PROVIDER``.

    Returns:
        TranscriptionConfig: Dataclass containing transcription configuration.
        - provider (str): ``"local"`` when ``INFERENCE_PROVIDER=ollama`` (the
          default), otherwise ``"external"``.
        - whisper_model (str): Model name used by the external Whisper API.
          Empty string in local mode. Defaults to ``whisper-1`` for ``openai``
          and ``openai/whisper-large-v3`` for ``vllm``; ``WHISPER_MODEL``
          overrides the default when set to a non-empty value.
    """
    inference_provider = load_inference_env().provider

    if inference_provider == "ollama":
        return TranscriptionConfig(provider="local", whisper_model="")

    default_model = EXTERNAL_WHISPER_DEFAULTS[inference_provider]
    override = os.getenv("WHISPER_MODEL", "").strip()
    return TranscriptionConfig(
        provider="external",
        whisper_model=override or default_model,
    )


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


def load_memory_env() -> MemoryConfig:
    """Loads model residency configuration from environment variables.

    Returns:
        MemoryConfig: Dataclass containing the resolved residency settings.
        - default_strategy (str): ``"offload"`` (default) or ``"evict"``. Read
          from ``MODEL_RESIDENCY_STRATEGY``. Unknown values fall back to
          ``"offload"`` with a warning. Per-model overrides
          (``MODEL_RESIDENCY_<NAME>``) are resolved inside the model registry
          and are not surfaced here.
    """
    raw = os.getenv("MODEL_RESIDENCY_STRATEGY", "offload").strip().lower()
    if raw not in VALID_RESIDENCY_STRATEGIES:
        logger.warning(
            "Unknown MODEL_RESIDENCY_STRATEGY '{}'. Falling back to 'offload'.",
            raw,
        )
        raw = "offload"
    return MemoryConfig(default_strategy=raw)


def load_vad_env() -> VadConfig:
    """Loads Voice Activity Detection configuration from environment variables.

    Returns:
        VadConfig: Dataclass containing the VAD toggle.
        - enabled (bool): ``True`` (default) when ``VAD_ENABLED`` is ``1``,
          ``true``, or ``yes``.
    """
    enabled = str(os.getenv("VAD_ENABLED", "1")).lower() in {"1", "true", "yes"}
    return VadConfig(enabled=enabled)


def load_path_env() -> PathConfig:
    """Loads path configuration from environment variables or defaults.

    Returns:
        PathConfig: Dataclass containing path configuration.
        - prompts (Path): Path to the prompts directory.
        - hf_hub_cache (Path): Path to the Hugging Face Hub cache directory.
    """
    default_cache_cache: Path = Path.home() / ".cache"
    default_hf_hub_cache: Path = default_cache_cache / "huggingface" / "hub"

    utils_dir: Path = Path(__file__).parent.resolve()
    default_prompts_dir: Path = utils_dir / "prompts"

    return PathConfig(
        prompts=default_prompts_dir,
        hf_hub_cache=Path(os.getenv("HF_HUB_CACHE", default_hf_hub_cache)).expanduser(),
    )


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
