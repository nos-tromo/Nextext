"""Synthetic media for the tests that read video: clips written with PyAV, no fixtures.

Every clip is solid colours with no people, so the tests need no media files and
no network. Shared by the frame sampler's, the keyframe sampler's and the audio
module's tests.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from av.stream import Disposition
from PIL import Image

#: Sample rate of the silent audio track ``make_clip(audio=True)`` writes.
AUDIO_RATE = 16000


def make_clip(
    path: Path,
    *,
    seconds: float,
    fps: int,
    width: int = 32,
    height: int = 32,
    codec: str = "mpeg4",
    gop_size: int | None = None,
    rotation: int | None = None,
    start_pts: int = 0,
    audio: bool = False,
) -> Path:
    """Write a synthetic clip whose stored frames are red on the left, blue on the right.

    Args:
        path: Destination file (container inferred from the suffix).
        seconds: Clip length.
        fps: Frame rate; also the time base, so frame ``i`` sits at ``i / fps``.
        width: Stored frame width.
        height: Stored frame height.
        codec: Encoder; ``mjpeg`` makes every frame a keyframe.
        gop_size: Keyframe interval in frames, for inter-coded codecs.
        rotation: Display rotation in degrees counter-clockwise, if any.
        start_pts: Presentation timestamp of the first frame.
        audio: Also write a silent mono audio track as long as the clip.

    Returns:
        Path: ``path``.
    """
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream(codec, rate=fps)
        assert isinstance(stream, av.VideoStream)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuvj420p" if codec == "mjpeg" else "yuv420p"
        stream.time_base = Fraction(1, fps)
        if gop_size is not None:
            stream.codec_context.gop_size = gop_size
        if rotation is not None:
            stream.set_display_rotation(rotation)
        sound = container.add_stream("aac", rate=AUDIO_RATE) if audio else None
        image = Image.new("RGB", (width, height), (0, 0, 255))
        image.paste((255, 0, 0), (0, 0, width // 2, height))
        for index in range(round(seconds * fps)):
            frame = av.VideoFrame.from_image(image)
            frame.pts = start_pts + index
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        if isinstance(sound, av.AudioStream):
            for offset in range(0, round(seconds * AUDIO_RATE), 1024):
                silence = av.AudioFrame.from_ndarray(np.zeros((1, 1024), dtype="f4"), format="fltp", layout="mono")
                silence.sample_rate = AUDIO_RATE
                silence.pts = offset
                for packet in sound.encode(silence):
                    container.mux(packet)
            for packet in sound.encode():
                container.mux(packet)
    return path


def make_with_cover_art(path: Path, *, video_seconds: int = 0) -> Path:
    """Write an mp4 with a cover image (an attached picture) and, optionally, a real video after it.

    Args:
        path: Destination ``.mp4`` file.
        video_seconds: Length of a real video stream added after the cover; ``0`` for none.

    Returns:
        Path: ``path``.
    """
    with av.open(str(path), mode="w") as container:
        cover = container.add_stream("mjpeg", rate=1)
        assert isinstance(cover, av.VideoStream)
        cover.width = cover.height = 32
        cover.pix_fmt = "yuvj420p"
        cover.disposition = Disposition.attached_pic
        video = None
        if video_seconds:
            video = container.add_stream("mpeg4", rate=4)
            assert isinstance(video, av.VideoStream)
            video.width = video.height = 32
            video.pix_fmt = "yuv420p"
            video.time_base = Fraction(1, 4)
            video.codec_context.gop_size = 4
        picture = av.VideoFrame.from_image(Image.new("RGB", (32, 32), (200, 0, 0)))
        picture.pts = 0
        picture.time_base = Fraction(1, 1)
        for packet in [*cover.encode(picture), *cover.encode()]:
            container.mux(packet)
        if video is not None:
            for index in range(video_seconds * 4):
                frame = av.VideoFrame.from_image(Image.new("RGB", (32, 32), (0, 0, 200)))
                frame.pts = index
                frame.time_base = Fraction(1, 4)
                for packet in video.encode(frame):
                    container.mux(packet)
            for packet in video.encode():
                container.mux(packet)
    return path
