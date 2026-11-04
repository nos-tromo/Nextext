"""Tests that error bodies are generic and logs name the error without its message.

An exception message can quote request or transcript content, so it reaches
neither the client nor the operational logs; the logs keep the exception
class and traceback frames.
"""

from __future__ import annotations

import io
import json
import time
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from loguru import logger

from nextext.api import main as main_module
from nextext.api.jobs import JobState, PushEvent
from nextext.api.main import create_app
from tests.audit_helpers import AuditCapture

# Matches the backend's NEXTEXT_AUTH_HEADER default; every request below
# carries the owner id under this header.
OWNER_HEADER = "X-Auth-User"
OWNER_ID = "a" * 32

MARKER = "MARKER-SECRET-1234"


def _client_with_owner(app: FastAPI) -> TestClient:
    """Build a TestClient that sends a fixed owner id on every request.

    Args:
        app: The FastAPI application to wrap.

    Returns:
        TestClient: A client preconfigured with the owner header.
    """
    client = TestClient(app, raise_server_exceptions=False)
    client.headers[OWNER_HEADER] = OWNER_ID
    return client


def _capture_logs() -> tuple[list[str], int]:
    """Attach a loguru sink that records every emitted message.

    Returns:
        tuple[list[str], int]: The list records are appended to, and the
            sink id to remove afterward.
    """
    records: list[str] = []
    sink_id = logger.add(lambda m: records.append(str(m)), level="DEBUG")
    return records, sink_id


@pytest.fixture
def app_with_boom() -> FastAPI:
    """Build the real app with a throwaway route that always raises.

    Returns:
        FastAPI: The app with a GET /boom route added.
    """
    app = create_app()
    app.get("/boom")(lambda: (_ for _ in ()).throw(RuntimeError(MARKER)))
    return app


def test_unhandled_error_is_generic_and_logged_without_its_message(app_with_boom: FastAPI) -> None:
    """A raising endpoint returns a generic body; the log names the error but not its message."""
    records, sink_id = _capture_logs()
    try:
        client = TestClient(app_with_boom, raise_server_exceptions=False)
        resp = client.get("/boom")
        assert resp.status_code == 500
        assert resp.json() == {"detail": "Internal server error."}
        assert MARKER not in resp.text
        logged = "\n".join(records)
        assert "Unhandled error on GET /boom: RuntimeError" in logged
        assert "Traceback (most recent call last):" in logged
        assert MARKER not in logged
    finally:
        logger.remove(sink_id)


def test_malformed_options_json_returns_generic_detail() -> None:
    """Malformed ``options`` JSON should yield a generic 4xx body, no exc fragment."""
    app = create_app()
    with _client_with_owner(app) as client:
        files = {"file": ("clip.wav", io.BytesIO(b"x"), "audio/wav")}
        response = client.post(
            "/api/v1/jobs",
            files=files,
            data={"options": "not-json"},
        )
    assert response.status_code in (400, 422)
    assert response.json() == {"detail": "Invalid request."}


def _boom_pipeline_runner(state: JobState, push: PushEvent) -> dict[str, Any]:
    """Pipeline stand-in that always raises with a marker in the message.

    Args:
        state: The job being processed.
        push: Event sink for SSE delivery (unused).

    Returns:
        dict[str, Any]: Never returns; always raises.
    """
    raise RuntimeError(MARKER)


@pytest.fixture
def boom_app_client(
    monkeypatch: pytest.MonkeyPatch, audit_capture: AuditCapture
) -> Iterator[tuple[TestClient, list[str]]]:
    """Spin up the FastAPI app with a pipeline runner that always raises.

    Args:
        monkeypatch: Pytest patcher; keeps the app's logging setup from
            removing the log capture.
        audit_capture: Where the app's audit events go.

    Yields:
        tuple[TestClient, list[str]]: The HTTP client and the captured log
            records list.
    """
    monkeypatch.setattr(main_module, "setup_logging", lambda: None)
    records, sink_id = _capture_logs()
    app = create_app(audit_emitter_factory=audit_capture.factory, pipeline_runner=_boom_pipeline_runner)
    try:
        with _client_with_owner(app) as client:
            yield client, records
    finally:
        logger.remove(sink_id)


def _wait_for_status(client: TestClient, job_id: str, target: str, timeout: float = 5.0) -> dict[str, Any]:
    """Poll ``GET /jobs/{id}`` until the status matches ``target``.

    Args:
        client: TestClient to poll with.
        job_id: Job identifier.
        target: Desired status string.
        timeout: Max seconds to wait.

    Returns:
        dict[str, Any]: The final snapshot body.

    Raises:
        AssertionError: If the deadline elapses without seeing ``target``.
    """
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}")
        if response.status_code == 200:
            last = response.json()
            if last["status"] == target:
                return last
        time.sleep(0.05)
    raise AssertionError(f"Job '{job_id}' never reached status '{target}'. Last seen: {last}")


def test_job_failure_error_is_generic_and_logged_without_its_message(
    boom_app_client: tuple[TestClient, list[str]],
) -> None:
    """A pipeline failure surfaces a static error and logs its class and frames, not its message."""
    client, records = boom_app_client
    options = {
        "task": "transcribe",
        "trg_lang": "de",
        "diarize": True,
        "words": False,
        "summarization": False,
        "hate_speech": False,
    }
    files = {"file": ("clip.wav", io.BytesIO(b"audio-bytes"), "audio/wav")}
    data = {"options": json.dumps(options)}

    create_response = client.post("/api/v1/jobs", files=files, data=data)
    assert create_response.status_code == 201
    job_id = create_response.json()["job_id"]

    snapshot = _wait_for_status(client, job_id, "failed")
    assert snapshot["error"] == "Job failed."
    assert MARKER not in json.dumps(snapshot)
    logged = "\n".join(records)
    assert f"Job {job_id} failed: RuntimeError" in logged
    assert "_boom_pipeline_runner" in logged
    assert MARKER not in logged
