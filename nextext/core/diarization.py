"""Speaker-diarization agent: HTTP client for the out-of-process ``/diarize`` service.

Nextext no longer hosts pyannote in-process. Diarization runs against an HTTP
``/diarize`` endpoint (e.g. ``nos-tromo/vllm-service``) that accepts a media
upload and returns chronological speaker turns. This module owns both the wire
call and the client-side alignment of those turns onto Whisper's transcript
segments (by maximum temporal overlap), keeping the contract in one place.

The endpoint is resolved by :func:`nextext.utils.env_cfg.load_diarization_env`:
``DIARIZE_API_BASE`` when set, else the central ``OPENAI_API_BASE`` (one trailing
``/v1`` stripped). When neither resolves diarization is disabled and callers
simply receive no speaker labels. Failures are logged and swallowed: a transcript
without speakers is preferable to a failed job.
"""

from pathlib import Path
from typing import Any

import httpx as httpx  # explicit re-export so tests can monkeypatch diarization.httpx
from loguru import logger

from nextext.utils.env_cfg import load_diarization_env

__all__ = [
    "SPEAKER_LABEL_PREFIX",
    "assign_speakers_by_overlap",
    "canonicalize_speaker_labels",
    "diarize_file",
]

SPEAKER_LABEL_PREFIX = "Speaker"


def canonicalize_speaker_labels(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Renumber raw diarization labels to contiguous ``Speaker N`` by first appearance.

    pyannote's ``SPEAKER_00``/``SPEAKER_02`` labels are arbitrary and gap-y.
    This maps them to ``Speaker 1``, ``Speaker 2``, … in the order each label
    is first heard (earliest turn ``start``), so the first voice is always
    ``Speaker 1``. The input order is preserved in the output; only the
    ``speaker`` string changes.

    Args:
        turns (list[dict[str, Any]]): Speaker turns with ``start`` / ``end`` /
            ``speaker`` keys, as returned by :func:`diarize_file`.

    Returns:
        list[dict[str, Any]]: New turn dicts (same order) with canonical labels.
    """
    mapping: dict[str, str] = {}
    for turn in sorted(turns, key=lambda t: float(t["start"])):
        raw = str(turn["speaker"])
        if raw not in mapping:
            mapping[raw] = f"{SPEAKER_LABEL_PREFIX} {len(mapping) + 1}"
    return [{**turn, "speaker": mapping[str(turn["speaker"])]} for turn in turns]


def diarize_file(
    file_path: Path,
    *,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[dict[str, Any]]:
    """Request speaker turns for an audio file from the ``/diarize`` service.

    The service URL is ``{base}/diarize`` for the resolved diarization ``base``
    (``DIARIZE_API_BASE`` or the central ``OPENAI_API_BASE`` with one trailing
    ``/v1`` stripped). When neither resolves diarization is disabled: a warning is
    logged and an empty list is returned so the caller proceeds without speaker
    labels. Any transport or HTTP error is likewise logged and swallowed into an
    empty list.

    ``num_speakers`` (exact count) is mutually exclusive with
    ``min_speakers``/``max_speakers`` on the server side; the frontend's
    "max speakers" control maps to ``max_speakers``.

    Args:
        file_path (Path): Path to the audio/video file to diarize. Sent as-is;
            the server resamples to 16 kHz mono via ffmpeg.
        num_speakers (int | None): Exact number of speakers, if known.
        min_speakers (int | None): Lower bound on the speaker count.
        max_speakers (int | None): Upper bound on the speaker count.

    Returns:
        list[dict[str, Any]]: Chronological speaker turns, each a mapping with
            ``start`` / ``end`` (absolute seconds) and ``speaker`` keys. Empty
            when diarization is disabled or the request fails.
    """
    config = load_diarization_env()
    if not config.api_base:
        logger.warning(
            "Diarization requested but no endpoint is configured (DIARIZE_API_BASE "
            "and OPENAI_API_BASE both unset); returning no speaker turns. Set "
            "DIARIZE_API_BASE or the central OPENAI_API_BASE to enable speaker labels."
        )
        return []

    data: dict[str, int] = {}
    if num_speakers is not None:
        data["num_speakers"] = num_speakers
    if min_speakers is not None:
        data["min_speakers"] = min_speakers
    if max_speakers is not None:
        data["max_speakers"] = max_speakers

    headers: dict[str, str] = {}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    url = f"{config.api_base}/diarize"
    try:
        with open(file_path, "rb") as audio:
            response = httpx.post(
                url,
                files={"file": (file_path.name, audio, "application/octet-stream")},
                data=data,
                headers=headers,
                timeout=config.timeout,
            )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        logger.error(
            "Diarization request to {} failed ({}): {}",
            url,
            exc.response.status_code,
            exc.response.text[:500],
        )
        return []
    except (httpx.HTTPError, ValueError, OSError) as exc:
        logger.error("Diarization request to {} failed: {}", url, exc)
        return []

    if not isinstance(payload, dict):
        logger.error("Diarization response from {} was not a JSON object; ignoring.", url)
        return []

    segments = list(payload.get("segments", []))
    logger.info("Diarization complete: {} speaker turns from '{}'.", len(segments), file_path.name)
    return segments


def assign_speakers_by_overlap(
    transcription_segments: list[dict[str, Any]],
    diarize_segments: list[dict[str, Any]],
) -> None:
    """Label transcript segments with the maximally-overlapping speaker.

    For each transcription segment, the total temporal overlap against every
    diarization turn is accumulated per speaker, and the speaker with the
    greatest overlap wins. Segments that overlap no turn are left untouched
    (they gain no ``speaker`` key). ``transcription_segments`` is mutated in
    place. This mirrors the previous in-process pyannote alignment, but reads
    the speaker turns from the ``/diarize`` response rather than a pyannote
    ``Annotation``.

    Args:
        transcription_segments (list[dict[str, Any]]): Whisper segments with
            float ``start`` / ``end`` keys (seconds). Each gains a ``speaker``
            key when an overlapping turn exists.
        diarize_segments (list[dict[str, Any]]): Speaker turns from
            :func:`diarize_file`, each with ``start`` / ``end`` / ``speaker``.
    """
    for segment in transcription_segments:
        seg_start = float(segment["start"])
        seg_end = float(segment["end"])
        speaker_durations: dict[str, float] = {}
        for turn in diarize_segments:
            overlap_start = max(seg_start, float(turn["start"]))
            overlap_end = min(seg_end, float(turn["end"]))
            if overlap_end > overlap_start:
                speaker = str(turn["speaker"])
                speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + (overlap_end - overlap_start)
        if speaker_durations:
            segment["speaker"] = max(speaker_durations, key=lambda s: speaker_durations[s])
