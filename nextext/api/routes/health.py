"""Health and language-listing routes."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from fastapi import APIRouter
from loguru import logger

from nextext.api.schemas import (
    HealthResponse,
    LanguageEntry,
    LanguagesResponse,
)
from nextext.core.openai_cfg import InferencePipeline
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


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Report API health and inference-provider reachability.

    Returns:
        HealthResponse: Status payload with ``inference`` set to whether the
            configured OpenAI-compatible endpoint responds.
    """
    try:
        inference_ok = InferencePipeline().get_health()
    except Exception:  # pragma: no cover - defensive; never want /health to 500
        logger.exception("Inference health check raised unexpectedly.")
        inference_ok = False
    return HealthResponse(inference=inference_ok, version=_package_version())


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


@router.get("/languages", response_model=LanguagesResponse)
def get_languages() -> LanguagesResponse:
    """Return source and target language mappings for the frontend.

    Returns:
        LanguagesResponse: Two sorted lists — Whisper source languages and
            TranslateGemma target languages — that the frontend uses to
            populate its dropdowns without bundling the JSON itself.
    """
    whisper = load_mappings("whisper_languages.json")
    target = load_mappings("translategemma_languages.json")
    return LanguagesResponse(
        whisper=_mapping_to_entries(whisper),
        target=_mapping_to_entries(target),
    )
