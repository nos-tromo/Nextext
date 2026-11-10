"""Centralized environment-config dataclasses (every env var loader lives here)."""

import math
import os
import re
from dataclasses import dataclass, field
from typing import Literal

from dotenv import load_dotenv
from loguru import logger

from nextext.utils.audit_event import ACTIONS
from nextext.utils.audit_runtime import parse_required_actions

load_dotenv()


def is_offline() -> bool:
    """Report whether Nextext runs in offline mode (``NEXTEXT_OFFLINE``).

    Offline mode gates the spaCy resource downloads — the only
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
    """Loads the LLM response-language setting from ``RESPONSE_LANGUAGE``.

    ``RESPONSE_LANGUAGE`` is the uniform federation-wide name for this setting;
    ``NEXTEXT_RESPONSE_LANGUAGE`` is honored as a deprecated fallback (warns
    when it is the only one set) for one release. The value selects which
    localized prompt subdirectory (``nextext/utils/prompts/<code>/``) supplies
    the system, summary, and hate-speech prompts, so it governs the language of
    generated summaries and hate-speech rationales independently of the
    transcription source language or the translation target. Translation
    output is unaffected — it is driven by explicit source/target language
    codes. Unrecognised values warn and fall back to ``default`` (then
    ``"en"``) so a typo cannot break bring-up.

    Args:
        default: Fallback language code used when neither ``RESPONSE_LANGUAGE``
            nor ``NEXTEXT_RESPONSE_LANGUAGE`` is set, or the resolved value is
            blank or invalid. Defaults to :data:`DEFAULT_PROMPT_LANGUAGE`
            (``"en"``).

    Returns:
        LanguageConfig: Dataclass carrying the resolved two-letter ``code``.
    """
    var_name = "RESPONSE_LANGUAGE"
    raw = os.getenv("RESPONSE_LANGUAGE")
    if raw is None or not raw.strip():
        var_name = "NEXTEXT_RESPONSE_LANGUAGE"
        raw = os.getenv("NEXTEXT_RESPONSE_LANGUAGE")
        if raw is not None and raw.strip():
            logger.warning(
                "NEXTEXT_RESPONSE_LANGUAGE is deprecated; use the uniform "
                "RESPONSE_LANGUAGE instead. Honoring the legacy value '{}'.",
                raw,
            )
    candidate = (raw if raw is not None else default).strip().lower()
    if candidate not in PROMPT_SUPPORTED_LANGUAGES:
        if raw is not None and raw.strip():
            logger.warning(
                "Unknown {} '{}'. Falling back to '{}'.",
                var_name,
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


FacesMode = Literal["off", "detect", "group"]


@dataclass(frozen=True)
class FacesConfig:
    """Dataclass for the faces service configuration (``/faces/detect``, ``/faces/group``).

    Face detection and grouping run out-of-process (e.g. ``nos-tromo/vllm-service``);
    Nextext holds no face model and never receives biometric features.

    Attributes:
        api_base: Root URL of the faces service (e.g. ``http://vllm-router:4000``);
            the client appends ``/faces/detect`` or ``/faces/group``. Falls back to
            the central endpoint when ``FACES_API_BASE`` is unset; an explicit
            falsy ``FACES_API_BASE`` switches the client off (empty string).
        api_key: Bearer token forwarded to the service, reused from
            ``OPENAI_API_KEY``. An empty string sends no ``Authorization`` header.
        timeout: Per-request timeout in seconds.
    """

    api_base: str
    api_key: str
    timeout: float


@dataclass(frozen=True)
class FacesSamplingConfig:
    """How densely the face stage samples a video.

    Attributes:
        interval_s: Seconds between sampled frames; a clip longer than
            ``max_frames`` intervals stretches it so the samples span the clip.
        max_frames: Most frames sampled per video.
    """

    interval_s: float
    max_frames: int


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
class SentenceRestoreConfig:
    """Dataclass for LLM sentence restoration on low-punctuation transcripts.

    When enabled and a transcript's terminal-punctuation density is below
    ``min_punct_ratio``, each contiguous speaker run is re-segmented into
    sentences by ``TEXT_MODEL`` so downstream rows are whole sentences rather
    than whole-speaker-turn blobs.

    Attributes:
        enabled: Whether restoration may run (``NEXTEXT_SENTENCE_RESTORE``,
            default ``True``).
        min_punct_ratio: Terminal-punctuation-per-word threshold below which a
            transcript is treated as low-punctuation
            (``SENTENCE_RESTORE_MIN_PUNCT_RATIO``, default 0.01).
    """

    enabled: bool
    min_punct_ratio: float


@dataclass(frozen=True)
class VisualSummaryConfig:
    """Dataclass for captioning the keyframes sampled from a video file.

    This is the operator kill-switch for the captioning half of the keyframe
    step, which the job itself opts into (``JobOptions.keyframes`` / the CLI's
    ``--keyframes``). When enabled, sampled keyframes are captioned by
    ``TEXT_MODEL`` and the timestamped captions become a result in their own
    right; a requested summary is additionally fed them, so it covers what was
    shown as well as what was said. When disabled, the frames are still sampled
    and downloadable — only the descriptions are skipped. Captioning requires a
    vision-capable chat model and is fail-soft, so a text-only model simply
    yields no captions.

    Attributes:
        enabled: Whether captioning may run (``NEXTEXT_VISUAL_SUMMARY``,
            default ``True``).
        max_frames: Upper bound on frames captioned per job
            (``VISUAL_SUMMARY_MAX_FRAMES``). One inference request is issued
            per frame, so this bounds the stage's cost.
        max_side: Longest edge in pixels each frame is downscaled to before
            upload (``VISUAL_SUMMARY_IMAGE_MAX_SIDE``), bounding request size
            and image-token count.
    """

    enabled: bool
    max_frames: int
    max_side: int


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


@dataclass(frozen=True)
class HateSpeechConfig:
    """Dataclass for the hate-speech stage's context-window budgets.

    Attributes:
        window_tokens: Estimated tokens of transcript rows labelled per request
            (the window's core). Every row is labelled exactly once, in the core
            of one window.
        context_tokens: Estimated tokens of read-only neighbouring rows shown on
            each side of the core so the model can resolve who is speaking,
            what is being answered, and what pronouns refer to. ``0`` disables
            the context margins.
    """

    window_tokens: int
    context_tokens: int


EXTERNAL_WHISPER_DEFAULTS: dict[str, str] = {
    "openai": "whisper-1",
    "vllm": "openai/whisper-large-v3",
}

DEFAULT_DIARIZE_TIMEOUT: float = 600.0
DEFAULT_NER_TIMEOUT: float = 120.0
DEFAULT_VAD_TIMEOUT: float = 60.0
DEFAULT_FACES_TIMEOUT: float = 300.0
DEFAULT_FACES_INTERVAL_S: float = 1.0
DEFAULT_FACES_MAX_FRAMES: int = 600
FACES_MAX_FRAMES_CEILING: int = 3600
DEFAULT_SENTENCE_RESTORE_MIN_PUNCT_RATIO: float = 0.01
DEFAULT_SUMMARY_MAX_INPUT_TOKENS: int = 6000
DEFAULT_HATE_SPEECH_WINDOW_TOKENS: int = 1000
DEFAULT_HATE_SPEECH_CONTEXT_TOKENS: int = 300
DEFAULT_KEYFRAMES_PER_MINUTE: int = 4
DEFAULT_KEYFRAMES_MAX: int = 20
KEYFRAMES_MAX_CEILING: int = 200
DEFAULT_VISUAL_SUMMARY_MAX_FRAMES: int = 12
DEFAULT_VISUAL_SUMMARY_IMAGE_MAX_SIDE: int = 1024
DEFAULT_JOB_CONCURRENCY: int = 1
DEFAULT_MEDIA_GRANT_TTL_S: int = 900
MEDIA_GRANT_TTL_CEILING_S: int = 86400


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
    ``/diarize``, ``/vad`` and ``/faces`` services sit at the service root alongside it, so a
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


def _load_positive_float(name: str, default: float) -> float:
    """Reads a positive, finite float from the environment, fail-soft.

    Args:
        name: Environment variable to read.
        default: Value used when unset, unparseable, non-finite, or not positive.

    Returns:
        float: The resolved value, always positive and finite.
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        parsed = float(raw)
        if not math.isfinite(parsed) or parsed <= 0:
            raise ValueError
    except ValueError:
        logger.warning("Invalid {} '{}'. Falling back to {}.", name, raw, default)
        return default
    return parsed


def load_faces_mode() -> FacesMode:
    """Resolve the operator switch for the face stage (``NEXTEXT_FACES``).

    ``detect`` finds faces and cuts crops without any biometric processing;
    ``group`` also groups them per person, which is biometric processing and
    needs the controller's impact assessment first. Read once at startup.

    Returns:
        FacesMode: ``"off"`` (default; also for an off token such as ``0``),
            ``"detect"`` or ``"group"``. An unknown value warns and stays
            ``"off"``.
    """
    raw = os.getenv("NEXTEXT_FACES", "").strip().lower()
    if not raw or raw in _FALSE_TOKENS:
        return "off"
    if raw == "detect":
        return "detect"
    if raw == "group":
        return "group"
    logger.warning("Unknown NEXTEXT_FACES '{}'. Face detection stays off.", raw)
    return "off"


def load_faces_env() -> FacesConfig:
    """Loads the faces service configuration from environment variables.

    Returns:
        FacesConfig: Dataclass containing the resolved settings.
        - api_base (str): The faces service root, resolved three ways from
          ``FACES_API_BASE``: a URL is used as-is (trailing ``/`` removed); an
          explicit falsy token (``off``/``false``/``no``/``0``) switches the
          client off (empty string); unset/blank falls back to the central
          endpoint (:func:`_central_endpoint_root`).
        - api_key (str): ``OPENAI_API_KEY`` (the faces service shares the
          inference router's credentials). Empty sends no bearer token.
        - timeout (float): ``FACES_TIMEOUT`` seconds. Defaults to
          :data:`DEFAULT_FACES_TIMEOUT`; invalid values warn and fall back.
    """
    raw_base = os.getenv("FACES_API_BASE", "").strip()
    if raw_base.lower() in _FALSE_TOKENS:
        api_base = ""
    elif raw_base:
        api_base = raw_base.rstrip("/")
    else:
        api_base = _central_endpoint_root()
    return FacesConfig(
        api_base=api_base,
        api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        timeout=_load_positive_float("FACES_TIMEOUT", DEFAULT_FACES_TIMEOUT),
    )


def load_faces_sampling_env() -> FacesSamplingConfig:
    """Loads how densely the face stage samples a video.

    Returns:
        FacesSamplingConfig: the resolved settings.
        - interval_s (float): ``FACES_INTERVAL_S``; defaults to
          :data:`DEFAULT_FACES_INTERVAL_S`. Values that are not positive and
          finite warn and fall back.
        - max_frames (int): ``FACES_MAX_FRAMES``; defaults to
          :data:`DEFAULT_FACES_MAX_FRAMES` and is clamped to
          :data:`FACES_MAX_FRAMES_CEILING`.
    """
    return FacesSamplingConfig(
        interval_s=_load_positive_float("FACES_INTERVAL_S", DEFAULT_FACES_INTERVAL_S),
        max_frames=_load_positive_int("FACES_MAX_FRAMES", DEFAULT_FACES_MAX_FRAMES, ceiling=FACES_MAX_FRAMES_CEILING),
    )


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


def load_sentence_restore_env() -> SentenceRestoreConfig:
    """Loads the sentence-restoration configuration from environment variables.

    Returns:
        SentenceRestoreConfig: the resolved settings.
        - enabled (bool): ``NEXTEXT_SENTENCE_RESTORE`` (default ``True``; only an
          explicit falsy token — ``0``/``false``/``no``/``off`` — disables it;
          unrecognised values warn and keep the default).
        - min_punct_ratio (float): ``SENTENCE_RESTORE_MIN_PUNCT_RATIO``; values
          outside ``(0, 1)`` or non-numeric warn and fall back to
          :data:`DEFAULT_SENTENCE_RESTORE_MIN_PUNCT_RATIO`.
    """
    parsed = _parse_tristate_bool("NEXTEXT_SENTENCE_RESTORE")
    enabled = True if parsed is None else parsed

    min_punct_ratio = DEFAULT_SENTENCE_RESTORE_MIN_PUNCT_RATIO
    raw_ratio = os.getenv("SENTENCE_RESTORE_MIN_PUNCT_RATIO", "").strip()
    if raw_ratio:
        try:
            value = float(raw_ratio)
            if not 0.0 < value < 1.0:
                raise ValueError
            min_punct_ratio = value
        except ValueError:
            logger.warning(
                "Invalid SENTENCE_RESTORE_MIN_PUNCT_RATIO '{}'. Falling back to {}.",
                raw_ratio,
                DEFAULT_SENTENCE_RESTORE_MIN_PUNCT_RATIO,
            )

    return SentenceRestoreConfig(enabled=enabled, min_punct_ratio=min_punct_ratio)


def _load_positive_int(name: str, default: int, *, ceiling: int | None = None) -> int:
    """Reads a positive integer budget from the environment, fail-soft.

    Args:
        name: Environment variable to read.
        default: Value used when unset, unparseable, or non-positive.
        ceiling: Optional inclusive upper bound; larger values warn and clamp.

    Returns:
        int: The resolved budget, always positive and within ``ceiling``.
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        value = default
    else:
        try:
            parsed = int(raw)
            if parsed <= 0:
                raise ValueError
            value = parsed
        except ValueError:
            logger.warning("Invalid {} '{}'. Falling back to {}.", name, raw, default)
            value = default
    if ceiling is not None and value > ceiling:
        logger.warning("{} {} exceeds the maximum {}. Clamping.", name, value, ceiling)
        value = ceiling
    return value


def _load_non_negative_int(name: str, default: int) -> int:
    """Reads a non-negative integer budget from the environment, fail-soft.

    Unlike :func:`_load_positive_int`, ``0`` is a valid value (it switches the
    budgeted feature off).

    Args:
        name: Environment variable to read.
        default: Value used when unset, unparseable, or negative.

    Returns:
        int: The resolved budget, always ``>= 0``.
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
        if parsed < 0:
            raise ValueError
    except ValueError:
        logger.warning("Invalid {} '{}'. Falling back to {}.", name, raw, default)
        return default
    return parsed


def load_visual_summary_env() -> VisualSummaryConfig:
    """Loads the visual-context (keyframe captioning) configuration.

    Read only when a job asked for keyframes; it gates the captioning half of
    that step, never the sampling.

    Returns:
        VisualSummaryConfig: the resolved settings.
        - enabled (bool): ``NEXTEXT_VISUAL_SUMMARY`` (default ``True``; only an
          explicit falsy token — ``0``/``false``/``no``/``off`` — disables it;
          unrecognised values warn and keep the default).
        - max_frames (int): ``VISUAL_SUMMARY_MAX_FRAMES``; defaults to
          :data:`DEFAULT_VISUAL_SUMMARY_MAX_FRAMES` and is clamped to
          :data:`KEYFRAMES_MAX_CEILING`, since each frame costs one request.
        - max_side (int): ``VISUAL_SUMMARY_IMAGE_MAX_SIDE``; defaults to
          :data:`DEFAULT_VISUAL_SUMMARY_IMAGE_MAX_SIDE`. Non-integer or
          non-positive values warn and fall back.
    """
    parsed = _parse_tristate_bool("NEXTEXT_VISUAL_SUMMARY")
    return VisualSummaryConfig(
        enabled=True if parsed is None else parsed,
        max_frames=_load_positive_int(
            "VISUAL_SUMMARY_MAX_FRAMES",
            DEFAULT_VISUAL_SUMMARY_MAX_FRAMES,
            ceiling=KEYFRAMES_MAX_CEILING,
        ),
        max_side=_load_positive_int(
            "VISUAL_SUMMARY_IMAGE_MAX_SIDE",
            DEFAULT_VISUAL_SUMMARY_IMAGE_MAX_SIDE,
        ),
    )


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


def load_hate_speech_env() -> HateSpeechConfig:
    """Loads the hate-speech context-window budgets from the environment.

    The windowed classifier in :mod:`nextext.pipeline` labels a core of
    transcript rows per request and shows neighbouring rows on both sides as
    read-only context.

    Returns:
        HateSpeechConfig: Dataclass containing the resolved settings.
        - window_tokens (int): ``HATE_SPEECH_WINDOW_TOKENS`` parsed as a
          positive integer. Defaults to
          :data:`DEFAULT_HATE_SPEECH_WINDOW_TOKENS`; invalid values warn and
          fall back to the default.
        - context_tokens (int): ``HATE_SPEECH_CONTEXT_TOKENS`` parsed as a
          non-negative integer (``0`` = no context). Defaults to
          :data:`DEFAULT_HATE_SPEECH_CONTEXT_TOKENS`; invalid values warn and
          fall back to the default.
    """
    return HateSpeechConfig(
        window_tokens=_load_positive_int("HATE_SPEECH_WINDOW_TOKENS", DEFAULT_HATE_SPEECH_WINDOW_TOKENS),
        context_tokens=_load_non_negative_int("HATE_SPEECH_CONTEXT_TOKENS", DEFAULT_HATE_SPEECH_CONTEXT_TOKENS),
    )


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


def load_job_concurrency(default: int = DEFAULT_JOB_CONCURRENCY) -> int:
    """Max jobs the ``JobManager`` runs concurrently (``NEXTEXT_JOB_CONCURRENCY``).

    Defaults to 1 (serial). Unparseable values fall back to the default and
    values below 1 clamp to 1 (both with a warning), so a misconfigured
    environment can never produce a zero/negative semaphore.

    Args:
        default (int): Fallback when the env var is unset/unparseable.

    Returns:
        int: The configured concurrency, at least 1.
    """
    raw = os.getenv("NEXTEXT_JOB_CONCURRENCY", "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid NEXTEXT_JOB_CONCURRENCY {!r}; using {}.", raw, default)
        return default
    if value < 1:
        logger.warning("NEXTEXT_JOB_CONCURRENCY {} is < 1; clamping to 1.", value)
        return 1
    return value


def load_media_grant_ttl() -> int:
    """Seconds a media playback grant stays usable (``NEXTEXT_MEDIA_GRANT_TTL_S``).

    A grant is the bearer capability a media element plays a recording with
    (``POST /jobs/{id}/media-grant``), so it is short-lived: 900 seconds by
    default, at most a day. Unparseable or non-positive values fall back to
    the default and larger ones clamp to the ceiling, both with a warning.

    Returns:
        int: The lifetime in seconds, between 1 and
            :data:`MEDIA_GRANT_TTL_CEILING_S`.
    """
    return _load_positive_int(
        "NEXTEXT_MEDIA_GRANT_TTL_S",
        DEFAULT_MEDIA_GRANT_TTL_S,
        ceiling=MEDIA_GRANT_TTL_CEILING_S,
    )


DEFAULT_CLI_RETENTION_DAYS: int = 365
MAX_CLI_RETENTION_DAYS: int = 3650
DEFAULT_RETENTION_NOTICE_DAYS: int = 30
MAX_RETENTION_NOTICE_DAYS: int = 365


@dataclass(frozen=True)
class CliRetentionConfig:
    """How long what ``make cli`` leaves on the scratch volume is kept.

    Attributes:
        days: Days after it was written at which a CLI run's files are
            deleted; ``None`` keeps them (development only).
        notice_days: Days a new or changed period deletes nothing after it
            took effect.
    """

    days: int | None
    notice_days: int


def _whole_days(name: str, default: int, maximum: int, *, allow_off: bool) -> int | None:
    """Read a period in whole days, fail-closed.

    Args:
        name: The environment variable.
        default: The value when unset or blank.
        maximum: The longest period accepted.
        allow_off: Whether ``off`` (``None``) is accepted.

    Returns:
        int | None: The days, or ``None`` for ``off``.

    Raises:
        ValueError: If the value is neither a whole number from 1 to
            ``maximum`` nor an allowed ``off``; a typo must never pick a
            period nobody chose.
    """
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw == "off" and allow_off:
        return None
    if raw.isascii() and raw.isdigit() and 1 <= int(raw) <= maximum:
        return int(raw)
    off = " or off" if allow_off else ""
    raise ValueError(f"{name} must be whole days from 1 to {maximum}{off}, not {raw!r}")


def load_cli_retention_env() -> CliRetentionConfig:
    """Load ``NEXTEXT_CLI_RETENTION_DAYS`` and ``RETENTION_NOTICE_DAYS``.

    Returns:
        CliRetentionConfig: 365 and 30 days unless configured.

    Raises:
        ValueError: If either is not a valid period.
    """
    days = _whole_days("NEXTEXT_CLI_RETENTION_DAYS", DEFAULT_CLI_RETENTION_DAYS, MAX_CLI_RETENTION_DAYS, allow_off=True)
    notice = _whole_days(
        "RETENTION_NOTICE_DAYS", DEFAULT_RETENTION_NOTICE_DAYS, MAX_RETENTION_NOTICE_DAYS, allow_off=False
    )
    if days is None:
        logger.warning("NEXTEXT_CLI_RETENTION_DAYS=off: what CLI runs leave is never deleted (development only).")
    return CliRetentionConfig(days=days, notice_days=DEFAULT_RETENTION_NOTICE_DAYS if notice is None else notice)


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


EDGE_SECRET_MIN_LENGTH: int = 32
SERVICE_TOKEN_MIN_LENGTH: int = 32
# Service principals live in their own namespace, so no user identity can
# collide with one: a service token resolves to ``service:<name>``, and no
# other source may produce a principal with this prefix.
SERVICE_PRINCIPAL_PREFIX: str = "service:"
_SERVICE_NAME_PATTERN: re.Pattern[str] = re.compile(r"[A-Za-z0-9._-]+")


class PrincipalConfigError(ValueError):
    """Raised when the identity configuration is malformed.

    Messages name the offending variable — and, for ``NEXTEXT_SERVICE_TOKENS``,
    the 1-based position of the offending entry — but never a configured value,
    so the error is safe to log.
    """


@dataclass(frozen=True)
class PrincipalConfig:
    """Dataclass for request-principal resolution configuration.

    Attributes:
        header_name: Trusted request header carrying the authenticated
            principal (the caller's owner identifier).
        default_identity: Developer / header-less fallback identity used when
            the trusted header is absent, or ``None`` when unset (the resolver
            then fails closed with HTTP 401). Never in the service namespace
            (:data:`SERVICE_PRINCIPAL_PREFIX`).
        edge_secret: Secret the edge gateway sends as ``X-Edge-Secret``. When
            set, the trusted header is honoured only on requests carrying it;
            ``None`` keeps the header trusted from any caller. Kept out of
            ``repr``.
        service_tokens: ``(name, token)`` pairs for services that call the API
            directly with ``Authorization: Bearer <token>``; such a request
            resolves to the principal ``service:<name>``. Kept out of
            ``repr``.
    """

    header_name: str
    default_identity: str | None
    edge_secret: str | None = field(default=None, repr=False)
    service_tokens: tuple[tuple[str, str], ...] = field(default=(), repr=False)


def _load_edge_secret() -> str | None:
    """Loads the edge-gateway secret from ``NEXTEXT_EDGE_SECRET``.

    Returns:
        str | None: The stripped secret, or ``None`` when unset or blank.

    Raises:
        PrincipalConfigError: When the secret is shorter than
            :data:`EDGE_SECRET_MIN_LENGTH` characters.
    """
    secret = os.getenv("NEXTEXT_EDGE_SECRET", "").strip()
    if not secret:
        return None
    if len(secret) < EDGE_SECRET_MIN_LENGTH:
        raise PrincipalConfigError(
            f"NEXTEXT_EDGE_SECRET is shorter than {EDGE_SECRET_MIN_LENGTH} characters; "
            "generate one with `make secret` in edge-plane."
        )
    return secret


def _load_service_tokens(edge_secret: str | None) -> tuple[tuple[str, str], ...]:
    """Parses ``NEXTEXT_SERVICE_TOKENS`` into ``(name, token)`` pairs.

    The value is a comma-separated list of ``name=token`` entries; empty
    entries (e.g. a trailing comma) are skipped. Names must match
    ``[A-Za-z0-9._-]+`` and be unique; tokens must be at least
    :data:`SERVICE_TOKEN_MIN_LENGTH` characters, unique, and distinct from the
    edge secret — a shared value would let one holder act as the other.

    Args:
        edge_secret (str | None): The configured edge secret, which no token
            may equal.

    Returns:
        tuple[tuple[str, str], ...]: The configured pairs, in order; empty
            when the variable is unset or blank.

    Raises:
        PrincipalConfigError: When an entry is malformed. The message names
            the entry by position only — never its name or token.
    """
    raw = os.getenv("NEXTEXT_SERVICE_TOKENS", "").strip()
    if not raw:
        return ()
    pairs: list[tuple[str, str]] = []
    name_positions: dict[str, int] = {}
    token_positions: dict[str, int] = {}
    for position, entry in enumerate(raw.split(","), start=1):
        if not entry.strip():
            continue
        raw_name, separator, raw_token = entry.partition("=")
        name = raw_name.strip()
        token = raw_token.strip()
        if not separator:
            raise PrincipalConfigError(f"NEXTEXT_SERVICE_TOKENS entry {position} is not a name=token pair.")
        if not _SERVICE_NAME_PATTERN.fullmatch(name):
            raise PrincipalConfigError(
                f"NEXTEXT_SERVICE_TOKENS entry {position}: the name must be non-empty and use only "
                "letters, digits, '.', '_' or '-'."
            )
        if len(token) < SERVICE_TOKEN_MIN_LENGTH:
            raise PrincipalConfigError(
                f"NEXTEXT_SERVICE_TOKENS entry {position}: the token is shorter than "
                f"{SERVICE_TOKEN_MIN_LENGTH} characters."
            )
        if name in name_positions:
            raise PrincipalConfigError(
                f"NEXTEXT_SERVICE_TOKENS entries {name_positions[name]} and {position} use the same name."
            )
        if token in token_positions:
            raise PrincipalConfigError(
                f"NEXTEXT_SERVICE_TOKENS entries {token_positions[token]} and {position} use the same token."
            )
        if token == edge_secret:
            raise PrincipalConfigError(
                f"NEXTEXT_SERVICE_TOKENS entry {position} reuses NEXTEXT_EDGE_SECRET; give each its own value."
            )
        name_positions[name] = position
        token_positions[token] = position
        pairs.append((name, token))
    return tuple(pairs)


def load_principal_env(
    default_header_name: str = "X-Auth-User",
    default_identity: str | None = None,
) -> PrincipalConfig:
    """Loads request-principal configuration from environment variables.

    Identity is carried by a trusted header, with an optional env-var fallback
    so developers (and any header-less client) resolve to a single configured
    identity instead of being rejected. ``NEXTEXT_EDGE_SECRET`` restricts the
    header to requests from the edge gateway, and ``NEXTEXT_SERVICE_TOKENS``
    lets services that call the API directly authenticate as themselves; see
    :mod:`nextext.api.identity` for the resolution order.

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
        - edge_secret (str | None): The edge-gateway secret, or ``None`` when
          unset or blank.
        - service_tokens (tuple[tuple[str, str], ...]): Configured
          ``(name, token)`` pairs.

    Raises:
        PrincipalConfigError: When ``NEXTEXT_EDGE_SECRET`` or
            ``NEXTEXT_SERVICE_TOKENS`` is malformed, or when
            ``NEXTEXT_DEFAULT_IDENTITY`` lies in the service namespace.
    """
    raw_identity = os.getenv("NEXTEXT_DEFAULT_IDENTITY")
    if raw_identity is not None and raw_identity.strip():
        resolved_identity: str | None = raw_identity.strip()
    else:
        resolved_identity = default_identity
    if resolved_identity is not None and resolved_identity.startswith(SERVICE_PRINCIPAL_PREFIX):
        raise PrincipalConfigError(
            f"NEXTEXT_DEFAULT_IDENTITY must not start with '{SERVICE_PRINCIPAL_PREFIX}', "
            "the prefix reserved for service tokens."
        )

    edge_secret = _load_edge_secret()
    return PrincipalConfig(
        header_name=os.getenv("NEXTEXT_AUTH_HEADER", default_header_name),
        default_identity=resolved_identity,
        edge_secret=edge_secret,
        service_tokens=_load_service_tokens(edge_secret),
    )


@dataclass(frozen=True)
class AuditConfig:
    """Dataclass for the data-protection audit-log settings.

    Attributes:
        output: ``AUDIT_OUTPUT`` — where audit lines go: unset or ``stdout``
            for the process's own stdout, ``container`` for the container's
            main stdout under a container-wide lock
            (:func:`nextext.utils.audit_runtime.open_audit_stream`).
        require_purpose: ``AUDIT_REQUIRE_PURPOSE`` — the actions that need a
            stated purpose; empty by default.
    """

    output: str | None
    require_purpose: frozenset[str]


def load_audit_env() -> AuditConfig:
    """Loads the audit-log settings from environment variables.

    Call it once at startup: an unknown action in ``AUDIT_REQUIRE_PURPOSE``
    stops the process instead of silently switching a requirement off.

    Returns:
        AuditConfig: The output mode and the actions that need a purpose.

    Raises:
        ValueError: When ``AUDIT_REQUIRE_PURPOSE`` names an unknown action.
    """
    required = parse_required_actions(os.getenv("AUDIT_REQUIRE_PURPOSE"), known=ACTIONS - {"lifecycle"})
    output = os.getenv("AUDIT_OUTPUT", "").strip() or None
    return AuditConfig(output=output, require_purpose=required)


def load_cli_audit_user() -> str | None:
    """Loads the operator a command-line run is attributed to (``AUDIT_CLI_USER``).

    ``make cli`` passes the host account that ran it (``SUDO_USER`` or
    ``id -un``) into the container.

    Returns:
        str | None: The stripped account name, or ``None`` when unset or blank.
    """
    return os.getenv("AUDIT_CLI_USER", "").strip() or None
