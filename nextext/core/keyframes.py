"""Extract a small, scene-diverse set of video keyframes as JPEG bytes.

Reuses the PyAV decode already relied on for audio (see
``nextext.core.audio``). Sampling spans the *whole* clip in two passes: a
cheap demux pass gathers every keyframe's timestamp without decoding, then we
evenly select timestamps across the full duration and seek to decode only
those. A long video is therefore framed end-to-end rather than only from its
opening minutes. Audio-only files, files with no video stream, and decode
errors yield ``[]`` (fail-soft) — a clip that can't be framed must never fail
the job.

The face stage reads a different shape of sample (:func:`sample_frames`): one
upright frame per time interval, held as an image rather than a JPEG, so the
faces it finds can be cut from the source at its own resolution.
"""

from __future__ import annotations

import io
import itertools
import json
import math
import os
import statistics
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import av
from av.error import FFmpegError
from loguru import logger
from PIL import Image

from nextext.utils.env_cfg import load_job_concurrency
from nextext.utils.log_cfg import error_label

__all__ = [
    "Keyframe",
    "SampledFrame",
    "extract_keyframe_samples",
    "extract_keyframes",
    "keyframe_manifest",
    "sample_frames",
    "subsample",
]

# Keyframes at most this many intervals apart are dense enough to sample from
# alone; sparser ones would leave gaps, so every reference frame is decoded.
_KEYFRAME_GAP_FACTOR = 2.0
# Slack for float timestamps that land a hair before their tick.
_TICK_EPSILON = 1e-6

# Which frames the decoder may skip: everything but keyframes, or only the
# frames no other frame refers to.
DecodeMode = Literal["NONKEY", "NONREF"]


@dataclass(frozen=True)
class Keyframe:
    """A single sampled video keyframe and the moment it was taken from.

    Attributes:
        time_sec: Presentation time of the frame in seconds from the start of
            the clip. Used to label visual-context captions so a reader can
            tie what was on screen to what was said at that point.
        jpeg: JPEG-encoded frame bytes.
    """

    time_sec: float
    jpeg: bytes


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


def _frame_time(frame: av.VideoFrame, pts: int, stream: av.VideoStream) -> float:
    """Resolve a decoded frame's presentation time in seconds.

    Prefers PyAV's own ``frame.time`` (pts scaled by the frame's time base).
    Some containers hand back frames without one, so fall back to scaling the
    seek target by the stream's time base, then to the microsecond convention
    ``av`` uses for container-level timestamps.

    Args:
        frame (av.VideoFrame): The decoded frame.
        pts (int): Presentation timestamp the decoder was seeked to.
        stream (av.VideoStream): Stream the frame was decoded from.

    Returns:
        float: Seconds from the start of the clip; never negative.
    """
    if frame.time is not None:
        return max(0.0, float(frame.time))
    if stream.time_base is not None:
        return max(0.0, float(pts * stream.time_base))
    return max(0.0, pts / 1_000_000)


def extract_keyframe_samples(file_path: Path, *, per_minute: int = 4, max_frames: int = 20) -> list[Keyframe]:
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
        list[Keyframe]: Timestamped frames in time order; ``[]`` when arguments
            are non-positive, there is no decodable video stream, or extraction
            fails before any frame is produced.
    """
    if per_minute <= 0 or max_frames <= 0:
        return []
    frames: list[Keyframe] = []
    try:
        with av.open(str(file_path)) as container:
            if not container.streams.video:
                logger.info("No video stream in '{}'; keyframe sampling skipped.", file_path.name)
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
                logger.info("No keyframes found in '{}'.", file_path.name)
                return []
            # Evenly select ``target`` keyframes spanning the full duration.
            selected = subsample(sorted(keyframe_pts), target)
            # Pass 2 — decode only the selected keyframes by seeking to each.
            for pts in selected:
                container.seek(pts, stream=stream)
                for frame in container.decode(stream):
                    buffer = io.BytesIO()
                    frame.to_image().save(buffer, format="JPEG")
                    frames.append(Keyframe(time_sec=_frame_time(frame, pts, stream), jpeg=buffer.getvalue()))
                    break
            logger.info(
                "Sampled {} keyframes from '{}' (duration={:.0f}s, target={}).",
                len(frames),
                file_path.name,
                duration_sec,
                target,
            )
    except (FFmpegError, ValueError, OSError) as exc:
        logger.warning("Keyframe extraction failed for {}: {}", file_path.name, exc)
    return frames


def extract_keyframes(file_path: Path, *, per_minute: int = 4, max_frames: int = 20) -> list[bytes]:
    """Return sampled video keyframes as JPEG bytes, without timestamps.

    Thin wrapper over :func:`extract_keyframe_samples` for the callers that
    only archive the images (the ``keyframes.zip`` artifact).

    Args:
        file_path (Path): Path to the media file.
        per_minute (int): Target frames per minute of video.
        max_frames (int): Hard ceiling on returned frames.

    Returns:
        list[bytes]: JPEG-encoded frames in time order; ``[]`` under the same
            conditions as :func:`extract_keyframe_samples`.
    """
    return [sample.jpeg for sample in extract_keyframe_samples(file_path, per_minute=per_minute, max_frames=max_frames)]


def keyframe_manifest(times: Sequence[float] | None, count: int) -> bytes | None:
    """Render the frame/time index that accompanies a set of keyframes.

    Args:
        times: Sampling time of each frame, in the frames' own order.
        count: Number of frames the manifest must describe.

    Returns:
        bytes | None: UTF-8 JSON naming each frame's file, index and
            ``time_sec``; ``None`` when no times are available or they do not
            pair with the frames, since a manifest that mislabels a frame is
            worse than none.
    """
    if not times or count <= 0 or len(times) != count:
        return None
    frames = [
        {"file": f"frame_{index:03d}.jpg", "index": index, "time_sec": float(time_sec)}
        for index, time_sec in enumerate(times)
    ]
    return json.dumps({"frames": frames}, ensure_ascii=False).encode("utf-8")


@dataclass(frozen=True)
class SampledFrame:
    """One frame of a video, sampled by time for the face stage.

    Attributes:
        index: Position among the samples yielded, counting from 0.
        time_sec: Presentation time in seconds from the stream's start, so it
            matches the player's clock.
        image: The frame as an RGB image, upright (display rotation applied)
            and no larger than the sampler's ``max_side``.
        scale: The image's size over the source frame's (``1.0`` unless the
            sampler scaled it down), so positions found in the image can be
            reported in the source's own pixels.
    """

    index: int
    time_sec: float
    image: Image.Image
    scale: float = 1.0


def _sample_time(frame: av.VideoFrame, packet_index: int, average_rate: float | None, start_sec: float) -> float | None:
    """Resolve a sampled frame's time in seconds from the start of its stream.

    Args:
        frame: The decoded frame.
        packet_index: Position of the packet it was decoded from, the fallback
            clock when the frame carries no timestamp.
        average_rate: The stream's average frame rate in frames per second,
            if known.
        start_sec: The stream's start time in seconds, subtracted so the first
            frame sits at ``0``.

    Returns:
        float | None: Seconds, never negative; ``None`` when the frame has no
            timestamp and the stream no rate to count by.
    """
    if frame.time is not None:
        return max(0.0, float(frame.time) - start_sec)
    if average_rate:
        return max(0.0, packet_index / average_rate)
    return None


def _upright_image(frame: av.VideoFrame, max_side: int) -> tuple[Image.Image, float]:
    """Convert a frame to an upright RGB image whose long side is at most ``max_side``.

    The frame is scaled down by the decoder's scaler before conversion, then
    turned by its display rotation (counter-clockwise degrees, as PyAV reports
    it) so a portrait phone video comes out portrait.

    Args:
        frame: The decoded frame.
        max_side: Largest allowed width or height.

    Returns:
        tuple[Image.Image, float]: The upright RGB image, and its scale
            relative to the source frame.
    """
    long_side = max(frame.width, frame.height)
    scale = min(1.0, max_side / long_side) if long_side else 1.0
    width = max(1, round(frame.width * scale))
    height = max(1, round(frame.height * scale))
    image = frame.reformat(width=width, height=height, format="rgb24").to_image()
    quarter_turns = round(frame.rotation / 90) % 4
    if quarter_turns == 1:
        return image.transpose(Image.Transpose.ROTATE_90), scale
    if quarter_turns == 2:
        return image.transpose(Image.Transpose.ROTATE_180), scale
    if quarter_turns == 3:
        return image.transpose(Image.Transpose.ROTATE_270), scale
    return image, scale


def _keyframe_times(file_path: Path) -> tuple[list[float], float] | None:
    """Demux a video once, without decoding, for its keyframe times and duration.

    Args:
        file_path: The media file.

    The duration comes from the container, else the stream, else — for a
    file that records none, such as a browser recording — the largest packet
    timestamp seen in this same pass.

    Returns:
        tuple[list[float], float] | None: Sorted keyframe times in seconds and
            the clip's duration in seconds (``0.0`` when unknown); ``None``
            when the file has no video stream.
    """
    with av.open(str(file_path)) as container:
        if not container.streams.video:
            return None
        stream = container.streams.video[0]
        time_base = stream.time_base
        keyframe_pts: list[int] = []
        last_pts: int | None = None
        for packet in container.demux(stream):
            if packet.pts is None:
                continue
            last_pts = packet.pts if last_pts is None else max(last_pts, packet.pts)
            if packet.is_keyframe:
                keyframe_pts.append(packet.pts)
        times = sorted(float(pts * time_base) for pts in keyframe_pts) if time_base is not None else []
        if container.duration:
            duration_sec = container.duration / 1_000_000
        elif stream.duration and time_base is not None:
            duration_sec = float(stream.duration * time_base)
        elif last_pts is not None and time_base is not None:
            start_pts = stream.start_time or 0
            duration_sec = max(0.0, float((last_pts - start_pts) * time_base))
        else:
            duration_sec = 0.0
    return times, duration_sec


def _decoder_threads() -> int:
    """Decoder threads for one job: the CPUs this process may use, shared among the job workers.

    A container limited by ``--cpus``/cpuset sees the host's count in
    ``os.cpu_count()``; the affinity mask is what it may actually run on.

    Returns:
        int: At least ``1``.
    """
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    cpus = len(sched_getaffinity(0)) if sched_getaffinity is not None else (os.cpu_count() or 1)
    return max(1, cpus // load_job_concurrency())


def _decode_mode(keyframe_times: Sequence[float], interval_s: float) -> DecodeMode:
    """Choose which frames the decoder may skip.

    Args:
        keyframe_times: Sorted keyframe times in seconds.
        interval_s: The sampling interval.

    Returns:
        DecodeMode: ``"NONKEY"`` when the median keyframe gap is at most twice the
            interval (keyframes alone are dense enough), else ``"NONREF"``.
    """
    gaps = [later - earlier for earlier, later in itertools.pairwise(keyframe_times)]
    if gaps and statistics.median(gaps) <= _KEYFRAME_GAP_FACTOR * interval_s:
        return "NONKEY"
    return "NONREF"


def sample_frames(
    file_path: Path, *, interval_s: float, max_frames: int, max_side: int = 3840
) -> Generator[SampledFrame, None, None]:
    """Yield one upright frame per time interval across the whole video.

    The interval stretches to ``duration / max_frames`` when the clip is longer
    than ``max_frames`` intervals, so the samples span it end to end. A demux
    pass measures the keyframe spacing first: when keyframes are dense enough
    only they are decoded, otherwise every reference frame. Each sample is the
    first frame at or after its tick. Decode errors end the sampling with what
    was yielded so far, logged by class; nothing raises.

    Args:
        file_path: The media file.
        interval_s: Seconds between samples.
        max_frames: Most samples to yield.
        max_side: Largest width or height of a yielded image.

    Yields:
        SampledFrame: The samples, in time order, indexed from ``0``.
    """
    if interval_s <= 0 or max_frames <= 0:
        return
    yielded = 0
    mode: DecodeMode = "NONREF"
    effective_interval = interval_s
    try:
        scan = _keyframe_times(file_path)
        if scan is None:
            logger.info("No video stream in '{}'; frame sampling skipped.", file_path.name)
            return
        keyframe_times, duration_sec = scan
        if duration_sec > 0:
            effective_interval = max(interval_s, duration_sec / max_frames)
        mode = _decode_mode(keyframe_times, effective_interval)
        with av.open(str(file_path)) as container:
            stream = container.streams.video[0]
            codec_context = stream.codec_context
            codec_context.skip_frame = mode
            codec_context.thread_type = "AUTO"
            codec_context.thread_count = _decoder_threads()
            rate = float(stream.average_rate) if stream.average_rate else None
            start_sec = (
                float(stream.start_time * stream.time_base)
                if stream.start_time is not None and stream.time_base is not None
                else 0.0
            )
            next_tick = 0.0
            for packet_index, packet in enumerate(container.demux(stream)):
                for frame in packet.decode():
                    time_sec = _sample_time(frame, packet_index, rate, start_sec)
                    if time_sec is None or time_sec + _TICK_EPSILON < next_tick:
                        continue
                    image, scale = _upright_image(frame, max_side)
                    yield SampledFrame(index=yielded, time_sec=time_sec, image=image, scale=scale)
                    yielded += 1
                    if yielded >= max_frames:
                        break
                    next_tick = (math.floor(time_sec / effective_interval + _TICK_EPSILON) + 1) * effective_interval
                if yielded >= max_frames:
                    break
    except (FFmpegError, ValueError, OSError) as exc:
        logger.warning("Frame sampling failed for '{}' after {} frames: {}", file_path.name, yielded, error_label(exc))
        return
    logger.info(
        "Sampled {} frames from '{}' (interval={:.2f}s, decode={}).",
        yielded,
        file_path.name,
        effective_interval,
        mode,
    )
