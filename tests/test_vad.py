"""Tests for the voice-activity-detection agent (external /vad HTTP client).

The guard is fail-open: an unset endpoint, a transport/HTTP error, or a
malformed payload all resolve to ``True`` (assume speech) so a VAD outage
never silently drops a transcription. Only an explicit ``{"has_speech": false}``
skips the Whisper upload.
"""

from pathlib import Path
from typing import Any

import httpx
import pytest

from nextext.core import vad
from nextext.core.vad import has_speech, speech_segments


def test_has_speech_true_when_base_unset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """With no VAD endpoint configured (dedicated or central), the guard issues no request.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
        tmp_path (Path): Temporary directory fixture for the audio file.
    """
    monkeypatch.delenv("VAD_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    def fail_post(url: str, **kwargs: Any) -> httpx.Response:
        raise AssertionError("httpx.post must not be called when no VAD endpoint is configured")

    monkeypatch.setattr(vad.httpx, "post", fail_post)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"data")

    assert has_speech(audio) is True


def test_has_speech_off_token_disables_guard_despite_central(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """VAD_API_BASE=off switches the guard off even when a central endpoint is set.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
        tmp_path (Path): Temporary directory fixture for the audio file.
    """
    monkeypatch.setenv("OPENAI_API_BASE", "http://vllm-router:4000/v1")
    monkeypatch.setenv("VAD_API_BASE", "off")

    def fail_post(url: str, **kwargs: Any) -> httpx.Response:
        raise AssertionError("httpx.post must not be called when VAD is switched off")

    monkeypatch.setattr(vad.httpx, "post", fail_post)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"data")

    assert has_speech(audio) is True


def test_has_speech_posts_correctly_and_parses_true(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The request targets /vad with the file, bearer auth, and timeout; true → True.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
        tmp_path (Path): Temporary directory fixture for the audio file.
    """
    monkeypatch.setenv("VAD_API_BASE", "http://router:7000/")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.delenv("VAD_TIMEOUT", raising=False)
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        captured["url"] = url
        captured.update(kwargs)
        return httpx.Response(200, json={"has_speech": True}, request=httpx.Request("POST", url))

    monkeypatch.setattr(vad.httpx, "post", fake_post)
    audio = tmp_path / "clip.mp4"
    audio.write_bytes(b"bytes")

    result = has_speech(audio)

    assert result is True
    assert captured["url"] == "http://router:7000/vad"
    assert captured["headers"]["Authorization"] == "Bearer sk-secret"
    assert captured["timeout"] == 60.0
    assert captured["files"]["file"][0] == "clip.mp4"


def test_has_speech_parses_false(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An explicit {"has_speech": false} is the one case that returns False.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
        tmp_path (Path): Temporary directory fixture for the audio file.
    """
    monkeypatch.setenv("VAD_API_BASE", "http://router:7000")

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(200, json={"has_speech": False}, request=httpx.Request("POST", url))

    monkeypatch.setattr(vad.httpx, "post", fake_post)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"x")

    assert has_speech(audio) is False


def test_has_speech_omits_authorization_without_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """No Authorization header is sent when OPENAI_API_KEY is empty.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
        tmp_path (Path): Temporary directory fixture for the audio file.
    """
    monkeypatch.setenv("VAD_API_BASE", "http://router:7000")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        captured.update(kwargs)
        return httpx.Response(200, json={"has_speech": True}, request=httpx.Request("POST", url))

    monkeypatch.setattr(vad.httpx, "post", fake_post)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"x")

    has_speech(audio)

    assert "Authorization" not in captured["headers"]


def test_has_speech_true_on_http_status_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A non-2xx response is logged and fails open to True (assume speech).

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
        tmp_path (Path): Temporary directory fixture for the audio file.
    """
    monkeypatch.setenv("VAD_API_BASE", "http://router:7000")

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(500, text="boom", request=httpx.Request("POST", url))

    monkeypatch.setattr(vad.httpx, "post", fake_post)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"x")

    assert has_speech(audio) is True


def test_has_speech_true_on_transport_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A transport error (e.g. connection refused) fails open to True.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
        tmp_path (Path): Temporary directory fixture for the audio file.
    """
    monkeypatch.setenv("VAD_API_BASE", "http://router:7000")

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(vad.httpx, "post", fake_post)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"x")

    assert has_speech(audio) is True


def test_has_speech_true_on_non_dict_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A JSON payload that is not an object fails open to True.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
        tmp_path (Path): Temporary directory fixture for the audio file.
    """
    monkeypatch.setenv("VAD_API_BASE", "http://router:7000")

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(200, json=["not", "a", "dict"], request=httpx.Request("POST", url))

    monkeypatch.setattr(vad.httpx, "post", fake_post)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"x")

    assert has_speech(audio) is True


def test_has_speech_true_on_missing_field(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A response object lacking a boolean has_speech field fails open to True.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
        tmp_path (Path): Temporary directory fixture for the audio file.
    """
    monkeypatch.setenv("VAD_API_BASE", "http://router:7000")

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(200, json={"speech": "maybe"}, request=httpx.Request("POST", url))

    monkeypatch.setattr(vad.httpx, "post", fake_post)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"x")

    assert has_speech(audio) is True


# ---------------------------------------------------------------------------
# speech_segments (VAD-gating source)
# ---------------------------------------------------------------------------


def test_speech_segments_none_when_base_unset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """With no VAD endpoint configured, gating gets None (no request) — never gates.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
        tmp_path (Path): Temporary directory fixture for the audio file.
    """
    monkeypatch.delenv("VAD_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    def fail_post(url: str, **kwargs: Any) -> httpx.Response:
        raise AssertionError("httpx.post must not be called when no VAD endpoint is configured")

    monkeypatch.setattr(vad.httpx, "post", fail_post)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"x")

    assert speech_segments(audio, threshold=0.4, pad_ms=100) is None


def test_speech_segments_posts_params_and_parses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The request targets /vad with threshold + speech_pad_ms and parses segments.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
        tmp_path (Path): Temporary directory fixture for the audio file.
    """
    monkeypatch.setenv("VAD_API_BASE", "http://router:7000/")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        captured["url"] = url
        captured.update(kwargs)
        return httpx.Response(
            200,
            json={"segments": [{"start": 0.0, "end": 1.5}, {"start": 3.0, "end": 4.0}], "has_speech": True},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(vad.httpx, "post", fake_post)
    audio = tmp_path / "c.wav"
    audio.write_bytes(b"x")

    segs = speech_segments(audio, threshold=0.4, pad_ms=100)

    assert segs == [(0.0, 1.5), (3.0, 4.0)]
    assert captured["url"] == "http://router:7000/vad"
    assert captured["data"] == {"threshold": 0.4, "speech_pad_ms": 100}
    assert captured["headers"]["Authorization"] == "Bearer sk-x"


def test_speech_segments_none_on_http_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A non-2xx response yields None (fail-open: no gating).

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
        tmp_path (Path): Temporary directory fixture for the audio file.
    """
    monkeypatch.setenv("VAD_API_BASE", "http://router:7000")

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(500, text="boom", request=httpx.Request("POST", url))

    monkeypatch.setattr(vad.httpx, "post", fake_post)
    audio = tmp_path / "c.wav"
    audio.write_bytes(b"x")

    assert speech_segments(audio, threshold=0.4, pad_ms=100) is None


def test_speech_segments_none_on_malformed_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A payload without a 'segments' list yields None (no gating).

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
        tmp_path (Path): Temporary directory fixture for the audio file.
    """
    monkeypatch.setenv("VAD_API_BASE", "http://router:7000")

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(200, json={"has_speech": True}, request=httpx.Request("POST", url))

    monkeypatch.setattr(vad.httpx, "post", fake_post)
    audio = tmp_path / "c.wav"
    audio.write_bytes(b"x")

    assert speech_segments(audio, threshold=0.4, pad_ms=100) is None


def test_speech_segments_none_on_off_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """VAD_API_BASE=off yields None (no request) even when a central endpoint is set.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
        tmp_path (Path): Temporary directory fixture for the audio file.
    """
    monkeypatch.setenv("OPENAI_API_BASE", "http://vllm-router:4000/v1")
    monkeypatch.setenv("VAD_API_BASE", "off")

    def fail_post(url: str, **kwargs: Any) -> httpx.Response:
        raise AssertionError("httpx.post must not be called when VAD is switched off")

    monkeypatch.setattr(vad.httpx, "post", fail_post)
    audio = tmp_path / "c.wav"
    audio.write_bytes(b"x")

    assert speech_segments(audio, threshold=0.4, pad_ms=100) is None


def test_speech_segments_none_on_non_dict_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A JSON payload that is not an object yields None (no gating).

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
        tmp_path (Path): Temporary directory fixture for the audio file.
    """
    monkeypatch.setenv("VAD_API_BASE", "http://router:7000")

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(200, json=["not", "a", "dict"], request=httpx.Request("POST", url))

    monkeypatch.setattr(vad.httpx, "post", fake_post)
    audio = tmp_path / "c.wav"
    audio.write_bytes(b"x")

    assert speech_segments(audio, threshold=0.4, pad_ms=100) is None


def test_speech_segments_none_on_malformed_segment_entry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A segment entry missing/with a non-numeric bound yields None (no gating).

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching env vars and httpx.
        tmp_path (Path): Temporary directory fixture for the audio file.
    """
    monkeypatch.setenv("VAD_API_BASE", "http://router:7000")

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            200, json={"segments": [{"start": 0.0, "end": "oops"}]}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(vad.httpx, "post", fake_post)
    audio = tmp_path / "c.wav"
    audio.write_bytes(b"x")

    assert speech_segments(audio, threshold=0.4, pad_ms=100) is None
