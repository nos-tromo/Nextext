"""Visual-context agent: caption sampled video keyframes for the summary.

Transcription only hears a file; for video, what is on screen (slides, scenes,
legible on-screen text) carries meaning the audio never states. This agent
turns the keyframes already sampled by :mod:`nextext.core.keyframes` into
short, timestamped captions via ``TEXT_MODEL``'s vision path
(:meth:`nextext.core.openai_cfg.InferencePipeline.call_vision`), which
:func:`nextext.pipeline.summarization_pipeline` prepends to the transcript.

Everything here is fail-soft: captioning is an enhancement, never a
precondition. A per-frame outage costs that one caption, and a model that
rejects images outright (a text-only ``TEXT_MODEL``) aborts the run after a
single request so the job simply falls back to an audio-only summary.
"""

from __future__ import annotations

import io
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

import openai
from loguru import logger
from PIL import Image

from nextext.core.keyframes import Keyframe, subsample
from nextext.core.openai_cfg import InferencePipeline

__all__ = ["FrameCaption", "describe_keyframes", "format_visual_context", "prepare_frame"]

CAPTION_MAX_OUTPUT_TOKENS: int = 160
"""Output cap per caption: a couple of sentences, never a page."""

_JPEG_QUALITY: int = 85

_REFUSAL_PATTERN = re.compile(
    r"\b(?:can(?:no|')t|cannot|unable to|not able to)\s+(?:see|view|access|process|analyz|analys|display)",
    re.IGNORECASE,
)
"""Matches the stock 'I can't see the image' reply a non-vision model returns.

Such a reply is not a description of the frame, so it must never reach the
summarizer as if it were one.
"""


@dataclass(frozen=True)
class FrameCaption:
    """A model-written description of one video keyframe.

    Attributes:
        time_sec: Seconds from the start of the clip at which the frame was
            sampled.
        caption: Short description of what the frame shows.
    """

    time_sec: float
    caption: str


def prepare_frame(jpeg: bytes, *, max_side: int) -> bytes:
    """Downscale and re-encode a keyframe for upload.

    Keyframes come out of the container at full source resolution, which for
    HD video means a multi-megabyte payload and a large image-token bill per
    caption. Scaling the longest edge down to ``max_side`` bounds both while
    leaving plenty of detail for a one-sentence description.

    Args:
        jpeg (bytes): Encoded frame bytes as produced by
            :func:`nextext.core.keyframes.extract_keyframe_samples`.
        max_side (int): Maximum length in pixels of the longest edge. Frames
            already within budget keep their size (never upscaled).

    Returns:
        bytes: JPEG-encoded frame bytes; the input unchanged when it cannot be
            decoded (fail-soft — let the endpoint be the judge).
    """
    try:
        with Image.open(io.BytesIO(jpeg)) as image:
            image.load()
            if image.mode != "RGB":
                image = image.convert("RGB")
            longest = max(image.size)
            if longest > max_side > 0:
                scale = max_side / longest
                image = image.resize(
                    (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                    Image.Resampling.LANCZOS,
                )
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=_JPEG_QUALITY)
            return buffer.getvalue()
    except (OSError, ValueError) as exc:
        logger.warning("Could not prepare a keyframe for captioning: {}", exc)
        return jpeg


def _is_permanent_rejection(exc: Exception) -> bool:
    """Report whether an error means the model will never caption an image.

    A 4xx other than rate-limiting says the request itself was unacceptable —
    typically a text-only model refusing image content parts. Retrying the
    remaining frames would just repeat the same failure.

    Args:
        exc (Exception): The error raised by the inference call.

    Returns:
        bool: ``True`` when the failure is a permanent client-side rejection.
    """
    return isinstance(exc, openai.APIStatusError) and 400 <= exc.status_code < 500 and exc.status_code != 429


def describe_keyframes(
    samples: Sequence[Keyframe],
    inference_pipeline: InferencePipeline,
    *,
    max_frames: int,
    max_side: int,
) -> list[FrameCaption]:
    """Caption sampled keyframes with the configured vision-capable chat model.

    Issues one request per frame (frames beyond ``max_frames`` are evenly
    subsampled away, preserving full-clip coverage), and drops replies that
    carry no description — blanks and the stock "I can't see the image"
    refusal a text-only model returns.

    Never raises: a frame that fails is skipped, and a first-frame client
    rejection aborts the whole run so a misconfigured (text-only) model costs
    one round-trip rather than ``max_frames`` of them.

    Args:
        samples (Sequence[Keyframe]): Timestamped frames to describe.
        inference_pipeline (InferencePipeline): Client for the chat model.
        max_frames (int): Upper bound on frames captioned.
        max_side (int): Longest edge each frame is downscaled to before upload.

    Returns:
        list[FrameCaption]: Captions in time order; ``[]`` when there are no
            frames, the budget is non-positive, or every request failed.
    """
    if not samples or max_frames <= 0:
        return []

    selected = subsample(list(samples), max_frames)
    try:
        prompt = inference_pipeline.load_prompt("frame_caption")
    except (OSError, FileNotFoundError) as exc:
        logger.warning("Frame-caption prompt unavailable; skipping visual context: {}", exc)
        return []

    captions: list[FrameCaption] = []
    failures = 0
    for position, sample in enumerate(selected):
        try:
            reply = inference_pipeline.call_vision(
                prompt=prompt,
                images=[prepare_frame(sample.jpeg, max_side=max_side)],
                num_predict=CAPTION_MAX_OUTPUT_TOKENS,
            )
        except Exception as exc:
            failures += 1
            if position == 0 and _is_permanent_rejection(exc):
                logger.warning(
                    "The chat model rejected image input ({}); skipping visual context for this job. "
                    "Visual summaries need a vision-capable TEXT_MODEL.",
                    exc,
                )
                return []
            logger.warning("Keyframe caption at {:.1f}s failed: {}", sample.time_sec, exc)
            continue

        text = " ".join(reply.split())
        if not text or _REFUSAL_PATTERN.search(text):
            continue
        captions.append(FrameCaption(time_sec=sample.time_sec, caption=text))

    if failures:
        logger.warning("Captioned {} of {} keyframes ({} failed).", len(captions), len(selected), failures)
    return captions


def _timestamp(seconds: float) -> str:
    """Render a caption timestamp as ``mm:ss`` (minutes grow past 60).

    Args:
        seconds (float): Offset from the start of the clip. Non-finite or
            negative values — which a damaged container can yield — render as
            ``00:00`` rather than raising.

    Returns:
        str: Zero-padded ``mm:ss`` stamp.
    """
    if not math.isfinite(seconds) or seconds < 0:
        seconds = 0.0
    total = int(seconds)
    return f"{total // 60:02d}:{total % 60:02d}"


def format_visual_context(captions: Sequence[FrameCaption]) -> str:
    """Render captions as the timestamped block handed to the summarizer.

    One caption per line, each stamped with its moment in the clip, so the
    model can align what was shown against what was said.

    Args:
        captions (Sequence[FrameCaption]): Captions in time order.

    Returns:
        str: ``"[mm:ss] caption"`` lines joined by newlines; ``""`` when there
            are no captions, so callers can test the block for truthiness.
    """
    return "\n".join(f"[{_timestamp(c.time_sec)}] {' '.join(c.caption.split())}" for c in captions)
