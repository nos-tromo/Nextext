"""Tests for the speaker-diarization agent (HTTP client + overlap alignment)."""

from pathlib import Path
from typing import Any

import httpx
import pytest

from nextext.core import diarization
from nextext.core.diarization import assign_speakers_by_overlap, diarize_file

# ---------------------------------------------------------------------------
# assign_speakers_by_overlap
# ---------------------------------------------------------------------------


def test_assign_speakers_uses_maximum_overlap() -> None:
    """The speaker with the greatest cumulative overlap is assigned to a segment."""
    segments: list[dict[str, Any]] = [{"start": 0.0, "end": 2.0, "text": "hello world"}]
    turns = [
        {"start": 0.0, "end": 1.8, "speaker": "SPEAKER_00"},  # 1.8s overlap
        {"start": 1.8, "end": 2.0, "speaker": "SPEAKER_01"},  # 0.2s overlap
    ]

    assign_speakers_by_overlap(segments, turns)

    assert segments[0]["speaker"] == "SPEAKER_00"


def test_assign_speakers_leaves_non_overlapping_segment_untouched() -> None:
    """A segment that overlaps no diarization turn gains no speaker key."""
    segments: list[dict[str, Any]] = [{"start": 5.0, "end": 6.0, "text": "x"}]
    turns = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]

    assign_speakers_by_overlap(segments, turns)

    assert "speaker" not in segments[0]


def test_assign_speakers_empty_segments_is_noop() -> None:
    """Aligning against zero transcription segments is a safe no-op."""
    segments: list[dict[str, Any]] = []

    assign_speakers_by_overlap(segments, [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}])

    assert segments == []


# ---------------------------------------------------------------------------
# diarize_file
# ---------------------------------------------------------------------------


def test_diarize_file_returns_empty_when_base_unset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """With no diarization endpoint configured (dedicated or central), it issues no request.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
        tmp_path (Path): Temporary directory fixture for the audio file.
    """
    monkeypatch.delenv("DIARIZE_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    def fail_post(url: str, **kwargs: Any) -> httpx.Response:
        raise AssertionError("httpx.post must not be called when no diarization endpoint is configured")

    monkeypatch.setattr(diarization.httpx, "post", fail_post)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"data")

    assert diarize_file(audio, max_speakers=3) == []


def test_diarize_file_posts_correctly_and_parses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The request targets /diarize with the speaker field, bearer auth, and timeout.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
        tmp_path (Path): Temporary directory fixture for the audio file.
    """
    monkeypatch.setenv("DIARIZE_API_BASE", "http://router:9000/")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.delenv("DIARIZE_TIMEOUT", raising=False)
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        captured["url"] = url
        captured.update(kwargs)
        return httpx.Response(
            200,
            json={
                "segments": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}],
                "speakers": ["SPEAKER_00"],
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(diarization.httpx, "post", fake_post)
    audio = tmp_path / "clip.mp4"
    audio.write_bytes(b"bytes")

    segments = diarize_file(audio, max_speakers=4)

    assert segments == [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]
    assert captured["url"] == "http://router:9000/diarize"
    assert captured["data"] == {"max_speakers": 4}
    assert captured["headers"]["Authorization"] == "Bearer sk-secret"
    assert captured["timeout"] == 600.0
    assert captured["files"]["file"][0] == "clip.mp4"


def test_diarize_file_omits_authorization_without_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """No Authorization header is sent when OPENAI_API_KEY is empty.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
        tmp_path (Path): Temporary directory fixture for the audio file.
    """
    monkeypatch.setenv("DIARIZE_API_BASE", "http://router:9000")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        captured.update(kwargs)
        return httpx.Response(200, json={"segments": []}, request=httpx.Request("POST", url))

    monkeypatch.setattr(diarization.httpx, "post", fake_post)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"x")

    diarize_file(audio, max_speakers=2)

    assert "Authorization" not in captured["headers"]


def test_diarize_file_swallows_http_status_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A non-2xx response is logged and yields an empty list.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
        tmp_path (Path): Temporary directory fixture for the audio file.
    """
    monkeypatch.setenv("DIARIZE_API_BASE", "http://router:9000")

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(500, text="boom", request=httpx.Request("POST", url))

    monkeypatch.setattr(diarization.httpx, "post", fake_post)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"x")

    assert diarize_file(audio, max_speakers=2) == []


def test_diarize_file_swallows_transport_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A transport error (e.g. connection refused) is logged and yields an empty list.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
        tmp_path (Path): Temporary directory fixture for the audio file.
    """
    monkeypatch.setenv("DIARIZE_API_BASE", "http://router:9000")

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(diarization.httpx, "post", fake_post)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"x")

    assert diarize_file(audio, max_speakers=2) == []


def test_diarize_file_handles_non_dict_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A JSON payload that is not an object is rejected and yields an empty list.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
        tmp_path (Path): Temporary directory fixture for the audio file.
    """
    monkeypatch.setenv("DIARIZE_API_BASE", "http://router:9000")

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(200, json=["not", "a", "dict"], request=httpx.Request("POST", url))

    monkeypatch.setattr(diarization.httpx, "post", fake_post)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"x")

    assert diarize_file(audio, max_speakers=2) == []
