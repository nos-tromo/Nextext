"""Tests for the external speaker-diarization HTTP client."""

from pathlib import Path

import httpx
import pytest
import respx

from nextext.core.diarization import diarize_file
from nextext.utils.env_cfg import DiarizationClientConfig

_BASE = "http://diarize:8000"


def _make_cfg(api_key: str | None = None) -> DiarizationClientConfig:
    """Build an explicit client configuration for tests.

    Args:
        api_key (str | None): Bearer token, or ``None`` for no auth header.

    Returns:
        DiarizationClientConfig: The assembled configuration.
    """
    return DiarizationClientConfig(api_base=_BASE, api_key=api_key, timeout=5.0)


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    """Provide a small on-disk stand-in for an audio upload.

    Args:
        tmp_path (Path): pytest-provided temporary directory.

    Returns:
        Path: Path to a small binary file.
    """
    file_path = tmp_path / "clip.wav"
    file_path.write_bytes(b"RIFF....WAVEfake")
    return file_path


@respx.mock
def test_diarize_file_parses_segments(audio_file: Path) -> None:
    """A successful response yields float-coerced speaker turns.

    Args:
        audio_file (Path): The fake audio upload.
    """
    route = respx.post(f"{_BASE}/diarize").mock(
        return_value=httpx.Response(
            200,
            json={
                "segments": [
                    {"start": 0, "end": 5.12, "speaker": "SPEAKER_00"},
                    {"start": "5.12", "end": 9.5, "speaker": "SPEAKER_01"},
                ]
            },
        )
    )

    segments = diarize_file(audio_file, max_speakers=2, cfg=_make_cfg())

    assert segments == [
        {"start": 0.0, "end": 5.12, "speaker": "SPEAKER_00"},
        {"start": 5.12, "end": 9.5, "speaker": "SPEAKER_01"},
    ]
    request = route.calls.last.request
    body = request.read()
    assert b'name="max_speakers"' in body
    assert b"2" in body
    assert b'filename="clip.wav"' in body


@respx.mock
def test_diarize_file_sends_bearer_header_when_key_set(audio_file: Path) -> None:
    """A configured API key is carried as a Bearer Authorization header.

    Args:
        audio_file (Path): The fake audio upload.
    """
    route = respx.post(f"{_BASE}/diarize").mock(return_value=httpx.Response(200, json={"segments": []}))

    diarize_file(audio_file, max_speakers=2, cfg=_make_cfg(api_key="sk-diarize"))

    assert route.calls.last.request.headers["Authorization"] == "Bearer sk-diarize"


@respx.mock
def test_diarize_file_omits_auth_header_without_key(audio_file: Path) -> None:
    """Without an API key no Authorization header is sent.

    Args:
        audio_file (Path): The fake audio upload.
    """
    route = respx.post(f"{_BASE}/diarize").mock(return_value=httpx.Response(200, json={"segments": []}))

    diarize_file(audio_file, max_speakers=2, cfg=_make_cfg(api_key=None))

    assert "Authorization" not in route.calls.last.request.headers


def test_diarize_file_unset_base_raises_actionable_error(
    audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty endpoint configuration raises before any network access.

    Args:
        audio_file (Path): The fake audio upload.
        monkeypatch (pytest.MonkeyPatch): Clears the central endpoint
            variables so the env-derived config is empty too.
    """
    cfg = DiarizationClientConfig(api_base="", api_key=None, timeout=5.0)

    with pytest.raises(RuntimeError) as excinfo:
        diarize_file(audio_file, max_speakers=2, cfg=cfg)

    message = str(excinfo.value)
    assert "DIARIZATION_API_BASE" in message
    assert "diarize" in message


@respx.mock
def test_diarize_file_404_names_missing_service(audio_file: Path) -> None:
    """A 404 explains that the configured service lacks diarization.

    Args:
        audio_file (Path): The fake audio upload.
    """
    respx.post(f"{_BASE}/diarize").mock(return_value=httpx.Response(404, text="not found"))

    with pytest.raises(RuntimeError) as excinfo:
        diarize_file(audio_file, max_speakers=2, cfg=_make_cfg())

    message = str(excinfo.value)
    assert "does not implement diarization" in message
    assert "DIARIZATION_API_BASE" in message


@respx.mock
def test_diarize_file_server_error_raises(audio_file: Path) -> None:
    """A 5xx response surfaces the status code and body excerpt.

    Args:
        audio_file (Path): The fake audio upload.
    """
    respx.post(f"{_BASE}/diarize").mock(return_value=httpx.Response(500, text="cuda OOM"))

    with pytest.raises(RuntimeError, match="HTTP 500.*cuda OOM"):
        diarize_file(audio_file, max_speakers=2, cfg=_make_cfg())


@respx.mock
def test_diarize_file_connect_error_raises_actionable_error(audio_file: Path) -> None:
    """Network-level failures raise with the setup hint.

    Args:
        audio_file (Path): The fake audio upload.
    """
    respx.post(f"{_BASE}/diarize").mock(side_effect=httpx.ConnectError("refused"))

    with pytest.raises(RuntimeError) as excinfo:
        diarize_file(audio_file, max_speakers=2, cfg=_make_cfg())

    message = str(excinfo.value)
    assert "Could not reach" in message
    assert "DIARIZATION_API_BASE" in message


@respx.mock
@pytest.mark.parametrize(
    "payload",
    [
        {"unexpected": "shape"},
        {"segments": "not-a-list"},
        ["not", "a", "dict"],
    ],
)
def test_diarize_file_malformed_payload_raises(audio_file: Path, payload: object) -> None:
    """Payloads without a segments list raise a RuntimeError.

    Args:
        audio_file (Path): The fake audio upload.
        payload (object): The malformed response payload under test.
    """
    respx.post(f"{_BASE}/diarize").mock(return_value=httpx.Response(200, json=payload))

    with pytest.raises(RuntimeError, match="unexpected payload shape"):
        diarize_file(audio_file, max_speakers=2, cfg=_make_cfg())


@respx.mock
@pytest.mark.parametrize(
    "segment",
    [
        "not-a-dict",
        {"start": 0.0, "end": 1.0},
        {"start": "x", "end": 1.0, "speaker": "SPEAKER_00"},
    ],
)
def test_diarize_file_malformed_segment_raises(audio_file: Path, segment: object) -> None:
    """Malformed segment entries fail hard instead of being skipped.

    Args:
        audio_file (Path): The fake audio upload.
        segment (object): The malformed segment entry under test.
    """
    respx.post(f"{_BASE}/diarize").mock(return_value=httpx.Response(200, json={"segments": [segment]}))

    with pytest.raises(RuntimeError, match="malformed"):
        diarize_file(audio_file, max_speakers=2, cfg=_make_cfg())


@respx.mock
def test_diarize_file_non_json_payload_raises(audio_file: Path) -> None:
    """A non-JSON body raises a RuntimeError naming the problem.

    Args:
        audio_file (Path): The fake audio upload.
    """
    respx.post(f"{_BASE}/diarize").mock(return_value=httpx.Response(200, text="<html>proxy error</html>"))

    with pytest.raises(RuntimeError, match="non-JSON"):
        diarize_file(audio_file, max_speakers=2, cfg=_make_cfg())
