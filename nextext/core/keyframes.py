"""Extract a small, scene-diverse set of video keyframes as JPEG bytes.

Reuses the PyAV decode already relied on for audio (see
``nextext.core.audio``). To stay cheap and naturally diverse we decode only
keyframes (``skip_frame="NONKEY"``), gather candidates up to a work bound, then
evenly subsample to the rate/cap requested by the job. Audio-only files, files
with no video stream, and decode errors yield ``[]`` (fail-soft) — a clip that
can't be framed must never fail the job.
"""

from __future__ import annotations

import io
from pathlib import Path

import av
from av.error import FFmpegError
from loguru import logger

__all__ = ["extract_keyframes", "subsample"]

# Decode at most this multiple of the cap before subsampling, so a long video
# does not decode end-to-end just to throw most frames away.
_CANDIDATE_BUDGET_FACTOR = 4


def subsample[T](items: list[T], target: int) -> list[T]:
    """Evenly pick ``target`` items across ``items`` (order preserved).

    Args:
        items (list[T]): Source items.
        target (int): Desired count. ``<= 0`` yields an empty list; a target
            at or above ``len(items)`` returns all items.

    Returns:
        list[T]: ``target`` evenly-spaced items (or all, or none).
    """
    if target <= 0:
        return []
    if len(items) <= target:
        return list(items)
    step = len(items) / target
    return [items[int(i * step)] for i in range(target)]


def extract_keyframes(file_path: Path, *, per_minute: int = 4, max_frames: int = 20) -> list[bytes]:
    """Return up to a capped, rate-scaled set of video keyframes as JPEG bytes.

    Args:
        file_path (Path): Path to the media file.
        per_minute (int): Target frames per minute of video.
        max_frames (int): Hard ceiling on returned frames.

    Returns:
        list[bytes]: JPEG-encoded frames in time order; ``[]`` when there is no
            decodable video stream.
    """
    if per_minute <= 0 or max_frames <= 0:
        return []
    candidates: list[bytes] = []
    target = 1
    try:
        with av.open(str(file_path)) as container:
            if not container.streams.video:
                return []
            stream = container.streams.video[0]
            stream.codec_context.skip_frame = "NONKEY"
            duration_sec = float(container.duration / 1_000_000) if container.duration else 0.0
            target = min(max_frames, max(1, round(duration_sec / 60.0 * per_minute)))
            work_budget = max_frames * _CANDIDATE_BUDGET_FACTOR
            for frame in container.decode(stream):
                buffer = io.BytesIO()
                frame.to_image().save(buffer, format="JPEG")
                candidates.append(buffer.getvalue())
                if len(candidates) >= work_budget:
                    break
    except (FFmpegError, ValueError, OSError) as exc:
        logger.warning("Keyframe extraction failed for {}: {}", file_path.name, exc)
    return subsample(candidates, target) if candidates else []
