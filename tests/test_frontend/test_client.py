"""Tests for :class:`nextext.frontend.client.BackendClient`."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from nextext.frontend.client import BackendClient, StageEvent


def _make_client(handler: Any) -> BackendClient:
    """Build a :class:`BackendClient` backed by an ``httpx.MockTransport``.

    Args:
        handler: A function ``(httpx.Request) -> httpx.Response``.

    Returns:
        BackendClient: A client wired to the mock transport.
    """
    transport = httpx.MockTransport(handler)
    return BackendClient(base_url="http://backend.test", transport=transport)


def test_submit_job_sends_multipart_and_returns_id() -> None:
    """``submit_job`` should send the file under ``file`` and options as JSON."""
    seen: dict[str, bytes] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content
        seen["content_type"] = request.headers.get("content-type", "").encode()
        return httpx.Response(
            201,
            json={
                "job_id": "abc",
                "status": "queued",
                "created_at": "2024-01-01T00:00:00Z",
            },
        )

    with _make_client(handler) as client:
        job_id = client.submit_job(
            "clip.wav",
            b"AUDIO",
            {"task": "transcribe", "trg_lang": "de"},
        )

    assert job_id == "abc"
    assert b"clip.wav" in seen["body"]
    assert b'"task": "transcribe"' in seen["body"]
    assert b"multipart/form-data" in seen["content_type"]


def test_get_snapshot_returns_decoded_payload() -> None:
    """``get_snapshot`` should return the parsed JSON body."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/jobs/job-1"
        return httpx.Response(
            200,
            json={
                "job_id": "job-1",
                "status": "completed",
                "file_name": "clip.wav",
                "options": {"task": "transcribe", "trg_lang": "de"},
                "created_at": "2024-01-01T00:00:00Z",
                "result": None,
            },
        )

    with _make_client(handler) as client:
        snapshot = client.get_snapshot("job-1")

    assert snapshot["status"] == "completed"


def test_subscribe_events_parses_sse_stream() -> None:
    """``subscribe_events`` should yield decoded ``StageEvent`` instances."""
    body = (
        "event: stage_started\n"
        'data: {"stage": "Transcribing", "stage_index": 0}\n\n'
        ": ping\n\n"
        "event: stage_completed\n"
        'data: {"stage": "Transcribing", "stage_index": 0}\n\n'
        "event: job_completed\n"
        'data: {"job_id": "job-1"}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=body.encode("utf-8"),
            headers={"content-type": "text/event-stream"},
        )

    with _make_client(handler) as client:
        events = list(client.subscribe_events("job-1"))

    assert [event.name for event in events] == [
        "stage_started",
        "stage_completed",
        "job_completed",
    ]
    assert events[0].data["stage"] == "Transcribing"


def test_download_artifact_returns_bytes_and_content_type() -> None:
    """``download_artifact`` should return raw bytes + content type."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"hello,world\n",
            headers={"content-type": "text/csv"},
        )

    with _make_client(handler) as client:
        payload, content_type = client.download_artifact("job-1", "transcript.csv")

    assert payload == b"hello,world\n"
    assert content_type == "text/csv"


def test_delete_job_tolerates_404() -> None:
    """``delete_job`` should not raise on 204 or 404."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with _make_client(handler) as client:
        client.delete_job("missing")


def test_delete_job_raises_on_unexpected_status() -> None:
    """``delete_job`` should raise for unrelated error codes."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    with _make_client(handler) as client:
        with pytest.raises(httpx.HTTPStatusError):
            client.delete_job("job-1")


def test_get_languages_returns_expected_shape() -> None:
    """``get_languages`` should return both lists from the backend."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "whisper": [{"code": "en", "name": "English"}],
                "target": [{"code": "de-DE", "name": "German (Germany)"}],
            },
        )

    with _make_client(handler) as client:
        body = client.get_languages()

    assert body["whisper"][0]["code"] == "en"
    assert body["target"][0]["code"] == "de-DE"


def test_default_base_url_uses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``BACKEND_HOST`` should win over the hard-coded default."""
    monkeypatch.setenv("BACKEND_HOST", "http://elsewhere:9000/")
    client = BackendClient()
    try:
        assert client.base_url == "http://elsewhere:9000"
    finally:
        client.close()


def test_stage_event_holds_name_and_data() -> None:
    """Sanity check the ``StageEvent`` dataclass."""
    event = StageEvent(name="stage_started", data={"stage": "Transcribing"})
    assert event.name == "stage_started"
    assert event.data["stage"] == "Transcribing"
