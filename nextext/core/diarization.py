"""HTTP client for the external speaker-diarization service.

Nextext no longer runs pyannote locally — speaker diarization is delegated
to an external HTTP service. vllm-service does not ship one yet; this module
defines the contract such a service must implement:

    POST {DIARIZATION_API_BASE}/diarize
        multipart/form-data:
            file:          the audio bytes (any ffmpeg-decodable format)
            max_speakers:  int form field — upper bound on distinct speakers
        Authorization: Bearer <DIARIZATION_API_KEY>   (optional)

    200 OK
        {"segments": [{"start": 0.0, "end": 5.12, "speaker": "SPEAKER_00"}, ...]}

``start``/``end`` are seconds as floats; ``speaker`` is an opaque label that
is surfaced verbatim in the transcript. The endpoint resolves via
:func:`nextext.utils.env_cfg.load_diarization_client_env` (dedicated
``DIARIZATION_API_BASE`` override, central ``OPENAI_API_BASE``-root
fallback).

Unlike NER — which degrades softly to fewer entities — diarization fails
hard: silently dropping speaker attribution would change analysis results
without anyone noticing, so every failure raises with an actionable message.
"""

from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from nextext.utils.env_cfg import DiarizationClientConfig, load_diarization_client_env

_SETUP_HINT = (
    "Speaker diarization requires an external diarization service. Set "
    "DIARIZATION_API_BASE (and DIARIZATION_API_KEY if it needs auth) to a "
    "server implementing POST /diarize; selecting a single speaker disables "
    "diarization."
)


def diarize_file(
    file_path: Path,
    max_speakers: int,
    cfg: DiarizationClientConfig | None = None,
) -> list[dict[str, Any]]:
    """Run speaker diarization on an audio file via the external service.

    Args:
        file_path (Path): Path to the audio file to upload.
        max_speakers (int): Upper bound on distinct speakers, forwarded as a
            form field.
        cfg (DiarizationClientConfig | None): Override client configuration.
            When ``None``, reads from the environment via
            :func:`nextext.utils.env_cfg.load_diarization_client_env`.

    Returns:
        list[dict[str, Any]]: Speaker turns as
        ``{"start": float, "end": float, "speaker": str}`` dicts.

    Raises:
        RuntimeError: When no endpoint is configured, the service is
            unreachable, responds with an error status (a 404 explicitly
            calls out that the configured service does not implement
            diarization), or returns a malformed payload.
    """
    effective_cfg = cfg if cfg is not None else load_diarization_client_env()
    if not effective_cfg.api_base:
        raise RuntimeError("No diarization endpoint is configured. " + _SETUP_HINT)

    headers: dict[str, str] = {}
    if effective_cfg.api_key:
        headers["Authorization"] = f"Bearer {effective_cfg.api_key}"

    url = f"{effective_cfg.api_base}/diarize"
    try:
        with file_path.open("rb") as audio:
            response = httpx.post(
                url,
                files={"file": (file_path.name, audio)},
                data={"max_speakers": str(max_speakers)},
                headers=headers,
                timeout=effective_cfg.timeout,
            )
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Could not reach the diarization service at {url}: {exc}. " + _SETUP_HINT) from exc

    if response.status_code == 404:
        raise RuntimeError(
            f"{url} does not exist (HTTP 404) — the configured service does not implement diarization. "
            + _SETUP_HINT
        )
    if response.is_error:
        raise RuntimeError(f"Diarization service returned HTTP {response.status_code}: {response.text[:500]}")

    try:
        payload = response.json()
    except Exception as exc:
        raise RuntimeError(f"Diarization service returned a non-JSON payload: {exc}") from exc

    raw_segments = payload.get("segments") if isinstance(payload, dict) else None
    if not isinstance(raw_segments, list):
        raise RuntimeError("Diarization service returned an unexpected payload shape (missing 'segments' list).")

    segments: list[dict[str, Any]] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            raise RuntimeError(f"Diarization segment is malformed: {item!r}")
        try:
            segments.append(
                {
                    "start": float(item["start"]),
                    "end": float(item["end"]),
                    "speaker": str(item["speaker"]),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Diarization segment is malformed: {item!r}") from exc

    logger.info("Diarization service returned {} speaker turns.", len(segments))
    return segments
