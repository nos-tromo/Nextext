"""Health and language-listing routes."""

from __future__ import annotations

import asyncio
from importlib.metadata import PackageNotFoundError, version

from fastapi import APIRouter, Request, Response, status
from loguru import logger
from starlette.concurrency import run_in_threadpool

from nextext.api.schemas import (
    ConfigResponse,
    HealthResponse,
    LanguageEntry,
    LanguagesResponse,
    ReadyResponse,
    VersionResponse,
)
from nextext.core.openai_cfg import InferencePipeline
from nextext.utils.env_cfg import DEFAULT_TARGET_LANG, load_default_target_lang, load_language_env
from nextext.utils.log_cfg import error_label, exception_frames
from nextext.utils.mappings_loader import load_mappings

router = APIRouter(tags=["health"])


def _package_version() -> str:
    """Return the installed Nextext package version, or ``"unknown"`` if missing.

    Returns:
        str: Resolved package version string.
    """
    try:
        return version("nextext")
    except PackageNotFoundError:
        return "unknown"


def _inference_reachable() -> bool:
    """Ask the central inference endpoint whether it answers, never raising.

    Returns:
        bool: ``InferencePipeline().get_health()``; ``False`` when the check
            itself raised.
    """
    try:
        return InferencePipeline().get_health()
    except Exception as exc:  # defensive: never let the check make /health or /ready answer 500
        logger.error("Inference health check raised unexpectedly: {}\n{}", error_label(exc), exception_frames(exc))
        return False


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Report API liveness and inference-provider reachability.

    Answers ``200`` while the process lives, whatever ``inference`` says: the
    container healthcheck and compose's ``service_healthy`` depend on it. Whether
    Whisper transcribes is ``GET /ready``'s question.

    Returns:
        HealthResponse: Status payload with ``inference`` set to whether the
            configured OpenAI-compatible endpoint responds.
    """
    return HealthResponse(inference=_inference_reachable(), version=_package_version())


@router.get(
    "/ready",
    response_model=ReadyResponse,
    responses={503: {"model": ReadyResponse, "description": "Inference or transcription is not available."}},
)
async def get_ready(request: Request, response: Response) -> ReadyResponse:
    """Report whether the service can transcribe now. Unauthenticated and not audited, like /health.

    ``transcription`` is a real probe of the Whisper endpoint with synthetic
    silence (:mod:`nextext.api.readiness`), cached for 30 s and shared by
    concurrent requests; ``inference`` is the same check ``/health`` reports.

    Args:
        request: The incoming request, for the app's readiness probe.
        response: Set to ``503`` while either check fails.

    Returns:
        ReadyResponse: ``ready`` with both checks true, else ``unavailable``
            (answered ``503``).
    """
    inference_ok, transcription_ok = await asyncio.gather(
        run_in_threadpool(_inference_reachable),
        request.app.state.readiness.is_ready(),
    )
    ready = inference_ok and transcription_ok
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadyResponse(
        status="ready" if ready else "unavailable",
        inference=inference_ok,
        transcription=transcription_ok,
        version=_package_version(),
    )


@router.get("/version", response_model=VersionResponse)
def get_version() -> VersionResponse:
    """Return the running app version (unauthenticated)."""
    return VersionResponse(version=_package_version())


@router.get("/config", response_model=ConfigResponse)
def get_config(request: Request) -> ConfigResponse:
    """Return the SPA bootstrap config. Unauthenticated and not audited, like /health.

    Args:
        request: The incoming request, for the purpose requirement the
            lifespan parsed from ``AUDIT_REQUIRE_PURPOSE`` and the face switch
            it read from ``NEXTEXT_FACES``.

    Returns:
        ConfigResponse: The UI language, the audit actions that need a
            stated purpose, sorted (empty when none do), and the face mode.
    """
    required = getattr(request.app.state, "audit_required", None)
    faces = getattr(request.app.state, "faces_mode", "off")
    return ConfigResponse(
        language=load_language_env().code,
        require_purpose=sorted(required) if isinstance(required, frozenset) else [],
        faces=faces if faces in ("detect", "group") else "off",
    )


def _mapping_to_entries(mapping: dict[str, str]) -> list[LanguageEntry]:
    """Convert a code->name mapping to a sorted list of ``LanguageEntry``.

    Args:
        mapping: Code-to-name mapping loaded from a JSON file.

    Returns:
        list[LanguageEntry]: Entries sorted by display name.
    """
    return sorted(
        (LanguageEntry(code=code, name=name) for code, name in mapping.items()),
        key=lambda entry: entry.name,
    )


def _resolve_default_target(target: dict[str, str], entries: list[LanguageEntry]) -> str:
    """Resolve the initial target language for the frontend dropdown.

    Resolution order: the configured ``NEXTEXT_DEFAULT_TARGET_LANG`` when it is
    a supported target code, then ``"en"`` when supported, then the first entry
    by display name. The frontend uses this only on a fresh browser; a persisted
    per-browser preference takes precedence client-side.

    Args:
        target: Code-to-name target language mapping.
        entries: The same mapping as a list sorted by display name.

    Returns:
        str: A target language code guaranteed to exist in ``target`` (empty
            only when no target languages are configured).
    """
    configured = load_default_target_lang()
    if configured in target:
        return configured
    if configured != DEFAULT_TARGET_LANG:
        logger.warning(
            "NEXTEXT_DEFAULT_TARGET_LANG '{}' is not a supported target language. Falling back to '{}'.",
            configured,
            DEFAULT_TARGET_LANG,
        )
    if DEFAULT_TARGET_LANG in target:
        return DEFAULT_TARGET_LANG
    return entries[0].code if entries else ""


@router.get("/languages", response_model=LanguagesResponse)
def get_languages() -> LanguagesResponse:
    """Return source and target language mappings for the frontend.

    Returns:
        LanguagesResponse: Two sorted lists — Whisper source languages and
            supported target languages — that the frontend uses to populate its
            dropdowns without bundling the JSON itself, plus ``default_target``,
            the initial target-language selection (configurable via
            ``NEXTEXT_DEFAULT_TARGET_LANG``; defaults to English).
    """
    whisper = load_mappings("whisper_languages.json")
    target = load_mappings("target_languages.json")
    target_entries = _mapping_to_entries(target)
    return LanguagesResponse(
        whisper=_mapping_to_entries(whisper),
        target=target_entries,
        default_target=_resolve_default_target(target, target_entries),
    )
