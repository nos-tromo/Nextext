"""Tests for keyframe extraction and even subsampling."""

import io
import wave
from fractions import Fraction
from pathlib import Path

import av
from PIL import Image, ImageStat

from nextext.core.keyframes import extract_keyframes, subsample


def test_subsample_evenly_picks_target() -> None:
    """Even subsampling picks target items spread across the input."""
    assert subsample([1, 2, 3, 4, 5, 6, 7, 8], 4) == [1, 3, 5, 7]


def test_subsample_returns_all_when_fewer_than_target() -> None:
    """Fewer items than the target returns them all unchanged."""
    assert subsample([1, 2], 5) == [1, 2]


def test_subsample_zero_target_is_empty() -> None:
    """A non-positive target yields an empty selection."""
    assert subsample([1, 2, 3], 0) == []


def test_extract_keyframes_failsoft_on_non_video(tmp_path: Path) -> None:
    """A non-video or undecodable file yields [] rather than raising."""
    bogus = tmp_path / "not_a_video.bin"
    bogus.write_bytes(b"\x00\x01\x02not media")
    # No video stream / undecodable -> empty list, never raises.
    assert extract_keyframes(bogus, per_minute=4, max_frames=5) == []


def test_extract_keyframes_failsoft_on_audio_only(tmp_path: Path) -> None:
    """A real audio-only file (no video stream) yields [] via the stream guard.

    Unlike the bogus-bytes test (which trips the ``av.open`` error path), this
    writes a valid WAV with the stdlib ``wave`` module, so ``av`` opens it
    successfully but finds no video stream — exercising the
    ``if not container.streams.video: return []`` guard for real.
    """
    wav = tmp_path / "silence.wav"
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x00\x00" * 800)  # ~0.1s of mono silence
    assert extract_keyframes(wav, per_minute=4, max_frames=5) == []


def test_extract_keyframes_non_positive_args_return_empty(tmp_path: Path) -> None:
    """Non-positive ``per_minute`` or ``max_frames`` short-circuits to []."""
    any_path = tmp_path / "unused.mp4"  # never opened; guard returns first
    assert extract_keyframes(any_path, per_minute=0, max_frames=5) == []
    assert extract_keyframes(any_path, per_minute=4, max_frames=0) == []


def _make_mjpeg_ramp(path: Path, *, n_frames: int, fps: int) -> None:
    """Write a deterministic mjpeg clip whose frames brighten over time.

    Every mjpeg frame is intra-coded (a keyframe), so no GOP or encoder-
    availability assumptions are needed. Frame ``i`` is a solid gray of value
    ``min(255, i * 8)``, giving a monotonic brightness ramp across the clip.

    Args:
        path (Path): Destination file (container format inferred from suffix).
        n_frames (int): Number of frames to encode.
        fps (int): Frame rate; also sets the per-frame timestamp spacing so the
            clip reports a real duration.
    """
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mjpeg", rate=fps)
        stream.width = 64
        stream.height = 64
        stream.pix_fmt = "yuvj420p"
        stream.time_base = Fraction(1, fps)
        for i in range(n_frames):
            value = min(255, i * 8)
            image = Image.new("RGB", (64, 64), (value, value, value))
            frame = av.VideoFrame.from_image(image)
            frame.pts = i
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():  # flush the encoder
            container.mux(packet)


def _mean_brightness(jpeg: bytes) -> float:
    """Return the mean luminance of a JPEG payload.

    Args:
        jpeg (bytes): JPEG-encoded image bytes.

    Returns:
        float: Mean pixel value of the grayscale-converted image.
    """
    with Image.open(io.BytesIO(jpeg)) as image:
        return ImageStat.Stat(image.convert("L")).mean[0]


def test_extract_keyframes_spans_full_duration(tmp_path: Path) -> None:
    """Frames are sampled across the whole clip, not just its opening minutes.

    A synthetic 30-frame / 30-second mjpeg clip is built with a brightness ramp
    (frame ``i`` is a solid gray of value ``~i*8``). Because the sampler now
    selects keyframes evenly across the full duration, the returned frames'
    mean brightness increases strictly from first to last — proving the
    selection spans early->late rather than clustering at the start. The large
    per-frame brightness step (~40) dwarfs JPEG quantization noise, so the
    ordering assertion is deterministic, not flaky.
    """
    video = tmp_path / "ramp.mkv"
    _make_mjpeg_ramp(video, n_frames=30, fps=1)

    # per_minute is high enough that ``target`` saturates ``max_frames`` for a
    # 30s clip, so we deterministically get 6 evenly-spaced frames.
    frames = extract_keyframes(video, per_minute=60, max_frames=6)

    assert 2 <= len(frames) <= 6
    assert len(set(frames)) == len(frames)  # every returned frame is distinct
    brightness = [_mean_brightness(f) for f in frames]
    assert all(brightness[i] < brightness[i + 1] for i in range(len(brightness) - 1))
    # The final selected frame must come from the LATE part of the ramp. The
    # old "gather from the start up to a work bound, then subsample" behavior
    # would top out near frame ~20 (brightness ~160); full-duration sampling
    # reaches frame ~25 (brightness ~200). A threshold between the two turns
    # this into a real regression guard for the rewrite.
    assert brightness[-1] > 180
