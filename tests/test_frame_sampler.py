"""Tests for the time-based video frame sampler the face stage reads from.

Every clip is synthesized with PyAV (solid colours, no people), so the tests
need no media fixtures and no network.
"""

from __future__ import annotations

import itertools
import wave
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import pytest
from PIL import Image

import nextext.core.keyframes as keyframes_module
from nextext.core.keyframes import SampledFrame, sample_frames


def _make_clip(
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
    return path


def _capture_info(run: Any) -> list[str]:
    """Run a callable while collecting the INFO log lines it writes.

    Args:
        run: Zero-argument callable.

    Returns:
        list[str]: The messages logged.
    """
    from loguru import logger

    lines: list[str] = []
    handler_id = logger.add(lines.append, level="INFO", format="{message}")
    try:
        run()
    finally:
        logger.remove(handler_id)
    return lines


def test_sample_frames_one_per_interval(tmp_path: Path) -> None:
    """A 10 s clip sampled every second yields about ten frames, indexed from 0, in time order."""
    clip = _make_clip(tmp_path / "clip.mp4", seconds=10, fps=24, gop_size=12)

    frames = list(sample_frames(clip, interval_s=1.0, max_frames=600))

    assert [f.index for f in frames] == list(range(len(frames)))
    assert 9 <= len(frames) <= 11
    assert all(b.time_sec > a.time_sec for a, b in itertools.pairwise(frames))
    assert all(isinstance(f, SampledFrame) and f.image.mode == "RGB" for f in frames)


def test_sample_frames_stretch_interval_over_long_clip(tmp_path: Path) -> None:
    """A clip longer than max_frames times the interval is sampled over its whole length."""
    clip = _make_clip(tmp_path / "long.mp4", seconds=60, fps=4, gop_size=4)

    frames = list(sample_frames(clip, interval_s=1.0, max_frames=10))

    assert len(frames) <= 10 and frames[-1].time_sec >= 50.0


def test_sample_frames_keyframe_path_when_gops_are_short(tmp_path: Path) -> None:
    """An all-intra clip is sampled by decoding keyframes only, with the same count."""
    clip = _make_clip(tmp_path / "intra.mkv", seconds=10, fps=8, codec="mjpeg")
    frames: list[SampledFrame] = []

    lines = _capture_info(lambda: frames.extend(sample_frames(clip, interval_s=1.0, max_frames=600)))

    assert 9 <= len(frames) <= 11
    assert any("decode=NONKEY" in line for line in lines)


def test_sample_frames_reference_path_when_gops_are_long(tmp_path: Path) -> None:
    """Keyframes further apart than twice the interval fall back to decoding reference frames."""
    clip = _make_clip(tmp_path / "longgop.mp4", seconds=10, fps=24, gop_size=240)
    frames: list[SampledFrame] = []

    lines = _capture_info(lambda: frames.extend(sample_frames(clip, interval_s=1.0, max_frames=600)))

    assert 9 <= len(frames) <= 11
    assert any("decode=NONREF" in line for line in lines)


def test_sample_frames_applies_display_rotation(tmp_path: Path) -> None:
    """A clip stored landscape with a 90° display rotation comes out upright (portrait).

    Stored frames are red on the left and blue on the right; rotated 90°
    counter-clockwise for display, the red half ends up at the bottom.
    """
    clip = _make_clip(tmp_path / "portrait.mp4", seconds=2, fps=8, width=64, height=32, rotation=90)

    frames = list(sample_frames(clip, interval_s=1.0, max_frames=600))

    assert frames
    image = frames[0].image
    assert image.size == (32, 64)
    top = image.getpixel((16, 8))
    bottom = image.getpixel((16, 56))
    assert isinstance(top, tuple) and isinstance(bottom, tuple)
    assert top[2] > 200 and top[0] < 60
    assert bottom[0] > 200 and bottom[2] < 60


def test_sample_frames_offsets_start_time(tmp_path: Path) -> None:
    """Times count from the stream's start, so they match the player's clock."""
    clip = _make_clip(tmp_path / "offset.mp4", seconds=4, fps=8, start_pts=8 * 5)

    frames = list(sample_frames(clip, interval_s=1.0, max_frames=600))

    assert frames
    assert frames[0].time_sec < 1.0
    assert frames[-1].time_sec < 4.0


def test_sample_frames_audio_only_yields_nothing(tmp_path: Path) -> None:
    """A file with no video stream yields no samples."""
    wav = tmp_path / "silence.wav"
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x00\x00" * 800)

    assert list(sample_frames(wav, interval_s=1.0, max_frames=600)) == []


def test_sample_frames_undecodable_file_yields_nothing(tmp_path: Path) -> None:
    """Bytes that are no media at all yield no samples rather than raising."""
    bogus = tmp_path / "not_media.bin"
    bogus.write_bytes(b"\x00\x01\x02not media")

    assert list(sample_frames(bogus, interval_s=1.0, max_frames=600)) == []


def test_sample_frames_corrupt_file_yields_what_it_has(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A decode error part-way through keeps the samples taken before it, and does not raise."""
    clip = _make_clip(tmp_path / "clip.mp4", seconds=10, fps=8, gop_size=8)
    opened = {"count": 0}
    real_open = av.open

    class _FlakyContainer:
        """Delegates to a real container; the second one opened fails mid-demux."""

        def __init__(self, real: Any, fail: bool) -> None:
            self._real = real
            self._fail = fail

        def __enter__(self) -> _FlakyContainer:
            self._real.__enter__()
            return self

        def __exit__(self, *exc: object) -> object:
            return self._real.__exit__(*exc)

        def demux(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
            for number, packet in enumerate(self._real.demux(*args, **kwargs)):
                if self._fail and number >= 20:
                    raise OSError("injected mid-decode failure")
                yield packet

        def __getattr__(self, name: str) -> object:
            return getattr(self._real, name)

    def _flaky_open(*args: Any, **kwargs: Any) -> _FlakyContainer:
        opened["count"] += 1
        return _FlakyContainer(real_open(*args, **kwargs), fail=opened["count"] >= 2)

    monkeypatch.setattr(keyframes_module.av, "open", _flaky_open)

    frames = list(sample_frames(clip, interval_s=1.0, max_frames=600))

    assert 1 <= len(frames) < 9
    assert [f.index for f in frames] == list(range(len(frames)))


def test_sample_frames_downscales_to_max_side(tmp_path: Path) -> None:
    """Frames larger than max_side are scaled down, keeping their aspect."""
    clip = _make_clip(tmp_path / "wide.mp4", seconds=2, fps=4, width=320, height=180)

    frames = list(sample_frames(clip, interval_s=1.0, max_frames=600, max_side=160))

    assert frames
    assert all(f.image.size == (160, 90) for f in frames)


def test_sample_frames_caps_at_max_frames(tmp_path: Path) -> None:
    """Never more than max_frames samples, whatever the interval."""
    clip = _make_clip(tmp_path / "clip.mp4", seconds=10, fps=8, gop_size=8)

    frames = list(sample_frames(clip, interval_s=0.25, max_frames=3))

    assert len(frames) == 3
