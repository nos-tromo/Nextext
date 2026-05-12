"""Tests for :class:`nextext.frontend.client.BackendClient`."""

from __future__ import annotations


import httpx
import pytest

from nextext.frontend.client import OWNER_HEADER, BackendClient, StageEvent

_TEST_OWNER_ID = "a" * 32


def _make_client(handler, *, owner_id: str | None = _TEST_OWNER_ID) -> BackendClient:  # type: ignore[no-untyped-def]
    """Build a :class:`BackendClient` backed by an ``httpx.MockTransport``.

    Args:
        handler: A function ``(httpx.Request) -> httpx.Response``.
        owner_id: Value sent in the ``X-Owner-Id`` header. Defaults to a
            32-char hex sentinel used across the frontend test suite.

    Returns:
        BackendClient: A client wired to the mock transport.
    """
    transport = httpx.MockTransport(handler)
    return BackendClient(
        base_url="http://backend.test",
        transport=transport,
        owner_id=owner_id,
    )


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


def test_list_jobs_returns_jobs_array() -> None:
    """``list_jobs`` should return the backend's ``jobs`` list."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/jobs"
        return httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "job_id": "abc",
                        "status": "completed",
                        "file_name": "clip.wav",
                        "progress": 1.0,
                        "created_at": "2026-05-01T00:00:00Z",
                        "task": "transcribe",
                    }
                ]
            },
        )

    with _make_client(handler) as client:
        jobs = client.list_jobs()

    assert len(jobs) == 1
    assert jobs[0]["job_id"] == "abc"


def test_submit_job_threads_persist_flag_into_options() -> None:
    """The persist flag should reach the backend inside ``options``."""
    seen: dict[str, bytes] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content
        return httpx.Response(
            201,
            json={
                "job_id": "p1",
                "status": "queued",
                "created_at": "2026-05-01T00:00:00Z",
            },
        )

    with _make_client(handler) as client:
        client.submit_job(
            "clip.wav",
            b"AUDIO",
            {"task": "transcribe", "persist": True},
        )

    assert b'"persist": true' in seen["body"]


def test_owner_id_is_sent_as_x_owner_id_header() -> None:
    """Every request must carry the configured ``X-Owner-Id`` header."""
    captured: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.headers.get(OWNER_HEADER))
        return httpx.Response(200, json={"jobs": []})

    with _make_client(handler) as client:
        client.list_jobs()
        client.list_jobs()

    assert captured == [_TEST_OWNER_ID, _TEST_OWNER_ID]


def test_distinct_owner_ids_produce_distinct_headers() -> None:
    """Two clients with different owner_ids send distinct headers."""
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get(OWNER_HEADER))
        return httpx.Response(200, json={"jobs": []})

    with (
        _make_client(handler, owner_id="a" * 32) as alice,
        _make_client(handler, owner_id="b" * 32) as bob,
    ):
        alice.list_jobs()
        bob.list_jobs()

    assert seen == ["a" * 32, "b" * 32]


def test_client_without_owner_id_omits_header() -> None:
    """When the caller skips owner_id, no header is sent (test convenience)."""
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get(OWNER_HEADER))
        return httpx.Response(
            200,
            json={"status": "ok", "inference": False, "version": "x"},
        )

    with _make_client(handler, owner_id=None) as client:
        client.get_health()

    assert seen == [None]
