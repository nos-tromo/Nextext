"""Tests for the audio-normalization agent (nextext.core.audio)."""

import threading
import wave
from pathlib import Path
from typing import cast

import av
import numpy as np
import pytest

from nextext.core import audio
from nextext.core.cancel import JobCancelled, bind_cancel
from tests.media_factory import make_animated_gif, make_clip, make_still_image, make_with_cover_art


def _write_stereo_wav(path: Path, *, seconds: float = 0.4, rate: int = 44100) -> None:
    """Write a short 44.1 kHz stereo PCM WAV sine clip for decode tests.

    Stereo @ 44.1 kHz forces the normalizer's downmix-to-mono and
    rate-conversion paths to run.

    Args:
        path (Path): Output WAV path; its parent must exist.
        seconds (float): Clip duration in seconds.
        rate (int): Sample rate in Hz.
    """
    n = int(rate * seconds)
    t = np.linspace(0, seconds, n, endpoint=False)
    tone = (0.2 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype("<i2")
    stereo = np.repeat(tone[:, None], 2, axis=1)  # (n, 2) interleaved L == R
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(stereo.tobytes())


def _write_mp3_with_corrupt_tail(path: Path, *, seconds: float = 2.0, rate: int = 44100) -> None:
    """Write a valid MP3 sine clip with a garbage frame appended to its tail.

    Reproduces the real-world failure where an MP3 carries a malformed final
    frame (trailing tag/padding bytes): the bulk of the stream decodes cleanly,
    then the decoder raises an FFmpeg ``InvalidDataError`` on the corrupt tail.

    Args:
        path (Path): Output ``.mp3`` path; its parent must exist.
        seconds (float): Duration of the valid audio body, in seconds.
        rate (int): Source sample rate in Hz.
    """
    with av.open(str(path), mode="w", format="mp3") as container:
        stream = cast(av.AudioStream, container.add_stream("libmp3lame", rate=rate))
        n = int(rate * seconds)
        t = np.linspace(0, seconds, n, endpoint=False)
        tone = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype("f4")
        frame = av.AudioFrame.from_ndarray(tone.reshape(1, -1), format="flt", layout="mono")
        frame.sample_rate = rate
        frame.pts = 0
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    # Append a fake MP3 sync header + zero padding: the valid body decodes, but
    # the final packet raises an FFmpeg InvalidDataError, exactly as in prod.
    with path.open("ab") as handle:
        handle.write(b"\xff\xfb\x90\x00" + b"\x00" * 4096)


def test_normalize_produces_16k_mono_flac(tmp_path: Path) -> None:
    """A stereo 44.1 kHz source yields a 16 kHz mono FLAC, removed on exit.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    src = tmp_path / "sample.wav"
    _write_stereo_wav(src)

    captured: Path | None = None
    with audio.normalize_for_transcription(src) as flac_path:
        captured = flac_path
        assert flac_path.exists()
        assert flac_path.suffix == ".flac"
        with av.open(str(flac_path)) as container:
            stream = container.streams.audio[0]
            assert stream.codec_context.name == "flac"
            frame = next(container.decode(stream))
            assert frame.sample_rate == 16000
            assert frame.layout.name == "mono"

    assert captured is not None
    assert not captured.exists()  # temp file cleaned up on context exit


def test_normalize_recovers_from_corrupt_trailing_frame(tmp_path: Path) -> None:
    """A mostly-valid MP3 with a corrupt final frame still yields a FLAC.

    The transcoder is best-effort: it skips the unreadable trailing frame and
    finishes with the audio that decoded, instead of discarding the whole file.
    Regression — a single bad final frame previously raised AudioDecodeError
    and threw away an entire (potentially hour-long) transcription.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    src = tmp_path / "corrupt_tail.mp3"
    _write_mp3_with_corrupt_tail(src, seconds=2.0)

    with audio.normalize_for_transcription(src) as flac_path:
        assert flac_path.exists()
        assert flac_path.suffix == ".flac"
        with av.open(str(flac_path)) as container:
            stream = container.streams.audio[0]
            assert stream.codec_context.name == "flac"
            samples = sum(frame.samples for frame in container.decode(stream))
    # Most of the ~2.0 s body survived the corrupt trailing frame.
    assert samples / 16000 >= 1.5


def test_normalize_raises_on_undecodable_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-audio bytes raise AudioDecodeError and leave no temp file behind.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
        monkeypatch (pytest.MonkeyPatch): Redirects the tempfile dir so the
            cleanup of the partial FLAC can be observed.
    """
    spool = tmp_path / "spool"
    spool.mkdir()
    monkeypatch.setattr("tempfile.tempdir", str(spool))

    bogus = tmp_path / "broken.ogg"
    bogus.write_bytes(b"NOT-REAL-AUDIO" * 64)
    with pytest.raises(audio.AudioDecodeError):
        with audio.normalize_for_transcription(bogus):
            pass

    # fail-closed must still clean up the temp FLAC it created
    assert list(spool.glob("*.flac")) == []


def test_normalize_writes_its_copy_into_the_scratch_dir(tmp_path: Path) -> None:
    """With a scratch dir the FLAC copy lives there — the API passes the job's directory."""
    src = tmp_path / "upload.wav"
    _write_stereo_wav(src)
    scratch = tmp_path / "job-dir"
    scratch.mkdir()
    with audio.normalize_for_transcription(src, scratch_dir=scratch) as flac_path:
        assert flac_path.parent == scratch
        assert flac_path.exists()
    assert list(scratch.iterdir()) == []


def test_normalize_stops_a_cancelled_job_and_leaves_no_copy(tmp_path: Path) -> None:
    """The per-packet loop is a check point; the partial FLAC is removed on the way out.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    src = tmp_path / "upload.wav"
    _write_stereo_wav(src)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    event = threading.Event()
    event.set()

    with bind_cancel(event), pytest.raises(JobCancelled):
        with audio.normalize_for_transcription(src, scratch):
            pass

    assert list(scratch.iterdir()) == []


def test_video_without_audio_is_true_for_a_clip_with_pictures_only(tmp_path: Path) -> None:
    """A clip with a video stream and no audio stream at all is the case the job skips and frames."""
    clip = make_clip(tmp_path / "screen.mp4", seconds=2, fps=8)

    assert audio.video_without_audio(clip) is True


def test_video_without_audio_is_false_when_the_clip_has_an_audio_stream(tmp_path: Path) -> None:
    """A clip with a (here silent) audio track reaches the normal path; the VAD guard judges its sound."""
    clip = make_clip(tmp_path / "talk.mp4", seconds=2, fps=8, audio=True)

    assert audio.video_without_audio(clip) is False


def test_video_without_audio_is_false_for_audio_only_and_for_cover_art_only(tmp_path: Path) -> None:
    """No video stream, or one that is only cover art, is no video: the file is left to the normalizer."""
    wav = tmp_path / "tone.wav"
    _write_stereo_wav(wav)
    cover = make_with_cover_art(tmp_path / "cover.mp4")

    assert audio.video_without_audio(wav) is False
    assert audio.video_without_audio(cover) is False


def test_video_without_audio_is_false_for_a_file_that_is_not_media_or_is_missing(tmp_path: Path) -> None:
    """It decides only on a positive finding: any error opening the file answers no."""
    not_media = tmp_path / "notes.bin"
    not_media.write_bytes(b"\x00\x01\x02not media")

    assert audio.video_without_audio(not_media) is False
    assert audio.video_without_audio(Path(__file__)) is False
    assert audio.video_without_audio(tmp_path / "missing.mp4") is False


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".webp"])
def test_video_without_audio_is_false_for_a_still_image(tmp_path: Path, suffix: str) -> None:
    """PyAV shows a still image as a one-frame video, but an image demuxer opened it: it is no video.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
        suffix (str): The image format's suffix.
    """
    picture = make_still_image(tmp_path / f"picture{suffix}")

    assert audio.video_without_audio(picture) is False


def test_video_without_audio_is_true_for_an_animated_gif(tmp_path: Path) -> None:
    """An animated GIF is opened by the ``gif`` demuxer, not an image one, so it counts as a silent video."""
    animation = make_animated_gif(tmp_path / "loop.gif", frames=4)

    assert audio.video_without_audio(animation) is True
