"""Voice-activity-detection agent: HTTP client for the out-of-process ``/vad`` service.

Nextext no longer runs a local Silero model. The pre-Whisper speech guard runs
against an HTTP ``/vad`` endpoint (e.g. ``nos-tromo/vllm-service``) that accepts a
media upload and reports whether the audio contains speech, so silent or
noise-only files never reach the (remote, possibly metered) Whisper endpoint.

The endpoint is located via ``VAD_API_BASE`` (see
:func:`nextext.utils.env_cfg.load_vad_env`). The guard is **fail-open**: when the
endpoint is unset, unreachable, errors, or returns a malformed payload,
:func:`has_speech` returns ``True`` so transcription proceeds. Only an explicit
``{"has_speech": false}`` skips the upload — a VAD outage must never silently
drop a transcription.

Request/response contract for the future vllm-service implementation::

    POST {VAD_API_BASE}/vad
        multipart/form-data:
            file: the audio bytes (any ffmpeg-decodable format; the server
                  decodes and resamples it)
        Authorization: Bearer <OPENAI_API_KEY>   (optional)

    200 OK
        {"has_speech": true | false}
"""

from pathlib import Path

import httpx as httpx  # explicit re-export so tests can monkeypatch vad.httpx
from loguru import logger

from nextext.utils.env_cfg import load_vad_env

__all__ = ["has_speech"]


def has_speech(file_path: Path) -> bool:
    """Report whether ``file_path`` contains speech via the ``/vad`` service.

    The service URL is ``{VAD_API_BASE}/vad``. The guard is fail-open: an unset
    ``VAD_API_BASE``, any transport/HTTP error, or a malformed payload all
    resolve to ``True`` so the caller proceeds to transcription. Only an
    explicit ``{"has_speech": false}`` returns ``False``.

    Args:
        file_path (Path): Path to the audio/video file to screen. Sent as-is;
            the server decodes and resamples it.

    Returns:
        bool: ``True`` when speech is present (or could not be ruled out),
            ``False`` only when the service explicitly reports no speech.
    """
    config = load_vad_env()
    if not config.api_base:
        return True

    headers: dict[str, str] = {}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    url = f"{config.api_base}/vad"
    try:
        with open(file_path, "rb") as audio:
            response = httpx.post(
                url,
                files={"file": (file_path.name, audio, "application/octet-stream")},
                headers=headers,
                timeout=config.timeout,
            )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "VAD request to {} failed ({}): {}; assuming speech.",
            url,
            exc.response.status_code,
            exc.response.text[:500],
        )
        return True
    except (httpx.HTTPError, ValueError, OSError) as exc:
        logger.warning("VAD request to {} failed: {}; assuming speech.", url, exc)
        return True

    if not isinstance(payload, dict):
        logger.warning("VAD response from {} was not a JSON object; assuming speech.", url)
        return True

    speech = payload.get("has_speech")
    if not isinstance(speech, bool):
        logger.warning("VAD response from {} lacked a boolean 'has_speech'; assuming speech.", url)
        return True

    if not speech:
        logger.info("VAD service reported no speech in '{}'.", file_path.name)
    return speech
