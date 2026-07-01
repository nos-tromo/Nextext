"""Extract a small, scene-diverse set of video keyframes as JPEG bytes.

Reuses the PyAV decode already relied on for audio (see
``nextext.core.audio``). Sampling spans the *whole* clip in two passes: a
cheap demux pass gathers every keyframe's timestamp without decoding, then we
evenly select timestamps across the full duration and seek to decode only
those. A long video is therefore framed end-to-end rather than only from its
opening minutes. Audio-only files, files with no video stream, and decode
errors yield ``[]`` (fail-soft) — a clip that can't be framed must never fail
the job.
"""

from __future__ import annotations

import io
from pathlib import Path

import av
from av.error import FFmpegError
from loguru import logger

__all__ = ["extract_keyframes", "subsample"]


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
    """Return video keyframes sampled evenly across the clip's full duration.

    Runs two passes over the container: a cheap demux pass collects every
    keyframe's presentation timestamp (no decoding), then ``target`` timestamps
    are chosen evenly across the whole duration and each is seeked and decoded
    to a JPEG. This frames the entire clip rather than only its opening
    minutes. Any decode/seek/parse error is logged and the frames gathered so
    far are returned — the function never raises.

    Args:
        file_path (Path): Path to the media file.
        per_minute (int): Target frames per minute of video; the effective
            count is capped by ``max_frames``.
        max_frames (int): Hard ceiling on returned frames.

    Returns:
        list[bytes]: JPEG-encoded frames in time order; ``[]`` when arguments
            are non-positive, there is no decodable video stream, or extraction
            fails before any frame is produced.
    """
    if per_minute <= 0 or max_frames <= 0:
        return []
    frames: list[bytes] = []
    try:
        with av.open(str(file_path)) as container:
            if not container.streams.video:
                return []
            stream = container.streams.video[0]
            stream.codec_context.skip_frame = "NONKEY"
            duration_sec = container.duration / 1_000_000 if container.duration else 0.0
            target = min(max_frames, max(1, round(duration_sec / 60.0 * per_minute)))
            # Pass 1 — cheap, no decoding: gather keyframe timestamps across the
            # whole file.
            keyframe_pts = [
                packet.pts for packet in container.demux(stream) if packet.is_keyframe and packet.pts is not None
            ]
            if not keyframe_pts:
                return []
            # Evenly select ``target`` keyframes spanning the full duration.
            selected = subsample(sorted(keyframe_pts), target)
            # Pass 2 — decode only the selected keyframes by seeking to each.
            for pts in selected:
                container.seek(pts, stream=stream)
                for frame in container.decode(stream):
                    buffer = io.BytesIO()
                    frame.to_image().save(buffer, format="JPEG")
                    frames.append(buffer.getvalue())
                    break
    except (FFmpegError, ValueError, OSError) as exc:
        logger.warning("Keyframe extraction failed for {}: {}", file_path.name, exc)
    return frames
