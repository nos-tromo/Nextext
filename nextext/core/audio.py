"""Audio-normalization agent: re-encode any upload to Whisper-ready audio.

The external Whisper endpoint (vLLM behind LiteLLM) decodes uploads with
``soundfile``/libsndfile only, with no ffmpeg fallback, so it rejects
containers/codecs libsndfile cannot read (Ogg Opus, m4a, mp4, mkv, webm, ...)
with an HTTP 400 "Invalid or unsupported audio file". This agent decodes the
upload with PyAV (bundled ffmpeg) and re-encodes it to 16 kHz mono FLAC --
losslessly compressed, libsndfile-native, on Whisper's accepted-format list, and
far smaller than WAV -- so every upload reaches the endpoint in a form it can
always decode. (The 16 kHz mono downmix is itself a deliberate, lossy reduction;
FLAC then stores that signal without further loss.)

Decoding is **best-effort**: individual packets that fail to decode (e.g. a
corrupt trailing frame in an MP3 carrying trailing tag/padding bytes) are
skipped so the rest of the audio still reaches the endpoint, mirroring the
``ffmpeg`` CLI. Unlike the fail-open VAD guard
(:func:`nextext.core.vad.has_speech`), it stays fail-closed on a *wholly*
unusable input -- no audio stream, or not one frame decodable -- raising
:class:`AudioDecodeError` rather than sending nothing downstream.
"""

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import av
from av.error import FFmpegError
from loguru import logger

__all__ = ["AudioDecodeError", "normalize_for_transcription"]

TARGET_SAMPLE_RATE = 16000
TARGET_LAYOUT = "mono"
TARGET_SAMPLE_FORMAT = "s16"


class AudioDecodeError(ValueError):
    """Raised when an upload cannot be decoded into Whisper-ready audio."""


def _transcode_to_flac(src: Path, dst: Path) -> None:
    """Decode ``src`` and write a 16 kHz mono FLAC to ``dst``.

    Decoding is best-effort: individual packets that fail to decode -- e.g. a
    malformed trailing frame in an MP3 carrying trailing tag/padding bytes -- are
    skipped so the rest of the audio still reaches the endpoint, the way the
    ``ffmpeg`` CLI tolerates a corrupt frame. The transcode is fail-closed only
    when *nothing* usable decodes: no audio stream, or zero frames recovered.

    Args:
        src (Path): Source media in any container/codec PyAV can decode.
        dst (Path): Destination path for the 16 kHz mono FLAC output.

    Raises:
        AudioDecodeError: If ``src`` has no audio stream, or not a single frame
            could be decoded from it.
    """
    resampler = av.AudioResampler(
        format=TARGET_SAMPLE_FORMAT,
        layout=TARGET_LAYOUT,
        rate=TARGET_SAMPLE_RATE,
    )
    frames_decoded = 0
    packets_skipped = 0
    try:
        with av.open(str(src)) as in_container, av.open(str(dst), mode="w") as out_container:
            if not in_container.streams.audio:
                raise AudioDecodeError(f"Could not decode audio file '{src.name}': no audio stream found.")
            in_stream = in_container.streams.audio[0]
            raw_stream = out_container.add_stream("flac", rate=TARGET_SAMPLE_RATE)
            out_stream = cast(av.AudioStream, raw_stream)
            out_stream.codec_context.format = TARGET_SAMPLE_FORMAT
            out_stream.codec_context.layout = TARGET_LAYOUT

            def emit(frame: "av.AudioFrame | None") -> None:
                """Resample one frame (or flush on ``None``) into the FLAC output."""
                for resampled in resampler.resample(frame):
                    resampled.pts = None
                    for packet in out_stream.encode(resampled):
                        out_container.mux(packet)

            packets = in_container.demux(in_stream)
            while True:
                try:
                    packet = next(packets)
                except StopIteration:
                    break
                except FFmpegError:
                    # A demuxer-level error leaves the container unusable; keep
                    # whatever already decoded rather than aborting outright.
                    packets_skipped += 1
                    break
                try:
                    frames = packet.decode()
                except FFmpegError:
                    # A single corrupt packet: skip it and keep going.
                    packets_skipped += 1
                    continue
                for frame in frames:
                    if not isinstance(frame, av.AudioFrame):
                        continue
                    frames_decoded += 1
                    emit(frame)
            emit(None)  # flush the resampler
            for packet in out_stream.encode(None):  # flush the encoder
                out_container.mux(packet)
    except AudioDecodeError:
        raise
    except FFmpegError as exc:
        raise AudioDecodeError(
            f"Could not decode audio file '{src.name}'; it may be corrupt or in an unsupported format."
        ) from exc

    if frames_decoded == 0:
        raise AudioDecodeError(
            f"Could not decode audio file '{src.name}'; it may be corrupt or in an unsupported format."
        )
    if packets_skipped:
        logger.warning(
            "Recovered '{}' after skipping {} undecodable packet(s); transcribing the audio that decoded.",
            src.name,
            packets_skipped,
        )


@contextmanager
def normalize_for_transcription(file_path: Path) -> Iterator[Path]:
    """Yield a 16 kHz mono FLAC re-encode of ``file_path``; delete it on exit.

    Decodes ``file_path`` with PyAV regardless of its container/codec and
    re-encodes it to a temporary 16 kHz mono FLAC that the soundfile-based
    Whisper loader can always read. The temporary file is removed when the
    context exits, whether or not the body raised.

    Args:
        file_path (Path): Path to the source upload (any decodable format).

    Yields:
        Path: Path to the temporary 16 kHz mono FLAC file.

    Raises:
        AudioDecodeError: If ``file_path`` cannot be decoded.
    """
    fd, tmp_name = tempfile.mkstemp(suffix=".flac", prefix=f"{file_path.stem}-")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        _transcode_to_flac(file_path, tmp_path)
        logger.info(
            "Normalized '{}' to 16 kHz mono FLAC ({} bytes).",
            file_path.name,
            tmp_path.stat().st_size,
        )
        yield tmp_path
    finally:
        tmp_path.unlink(missing_ok=True)
