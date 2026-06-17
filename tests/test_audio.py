"""Tests for the audio-normalization agent (nextext.core.audio)."""

import wave
from pathlib import Path
from typing import cast

import av
import numpy as np
import pytest

from nextext.core import audio


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
