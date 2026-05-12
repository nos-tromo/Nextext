"""Tests for the ``/api/v1/jobs`` lifecycle endpoints."""

from __future__ import annotations

import io
import json
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from nextext.api.jobs import JobManager


def _wait_for_status(
    client: TestClient,
    job_id: str,
    target: str,
    timeout: float = 5.0,
) -> dict[str, Any]:
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
    raise AssertionError(
        f"Job '{job_id}' never reached status '{target}'. Last seen: {last}"
    )


def test_post_jobs_creates_and_runs_a_stub_pipeline(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """Submitting a job should run the stub and surface a completed snapshot."""
    client, _ = stub_app_client
    options = {
        "task": "transcribe",
        "trg_lang": "de",
        "speakers": 1,
        "words": False,
        "summarization": True,
        "hate_speech": False,
    }
    files = {"file": ("clip.wav", io.BytesIO(b"audio-bytes"), "audio/wav")}
    data = {"options": json.dumps(options)}

    create_response = client.post("/api/v1/jobs", files=files, data=data)

    assert create_response.status_code == 201
    job_id = create_response.json()["job_id"]

    snapshot = _wait_for_status(client, job_id, "completed")
    assert snapshot["status"] == "completed"
    assert snapshot["progress"] == 1.0
    assert snapshot["result"] is not None
    assert snapshot["result"]["summary"] == "A short summary."
    transcript = snapshot["result"]["transcript"]
    assert len(transcript) == 2
    assert transcript[0]["text"] == "Hello world."


def test_post_jobs_rejects_invalid_options_json(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """Bad JSON in the options field should yield a 422."""
    client, _ = stub_app_client
    files = {"file": ("clip.wav", io.BytesIO(b"x"), "audio/wav")}
    response = client.post(
        "/api/v1/jobs",
        files=files,
        data={"options": "not-json"},
    )
    assert response.status_code == 422


def test_get_jobs_returns_404_for_unknown_id(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """Unknown job ids should yield a 404."""
    client, _ = stub_app_client
    response = client.get("/api/v1/jobs/missing")
    assert response.status_code == 404


def test_delete_jobs_removes_state(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """DELETE should evict the job and return 204."""
    client, manager = stub_app_client
    options = {
        "task": "transcribe",
        "trg_lang": "de",
        "speakers": 1,
        "words": False,
        "summarization": False,
        "hate_speech": False,
    }
    files = {"file": ("clip.wav", io.BytesIO(b"x"), "audio/wav")}
    create_response = client.post(
        "/api/v1/jobs",
        files=files,
        data={"options": json.dumps(options)},
    )
    job_id = create_response.json()["job_id"]
    _wait_for_status(client, job_id, "completed")

    delete_response = client.delete(f"/api/v1/jobs/{job_id}")
    assert delete_response.status_code == 204

    second_delete = client.delete(f"/api/v1/jobs/{job_id}")
    assert second_delete.status_code == 404


def test_oversized_upload_is_rejected(
    stub_app_client: tuple[TestClient, JobManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Uploads exceeding ``NEXTEXT_MAX_UPLOAD_MB`` should yield a 413."""
    monkeypatch.setenv("NEXTEXT_MAX_UPLOAD_MB", "1")
    client, _ = stub_app_client
    huge_payload = b"\x00" * (2 << 20)  # 2 MiB > 1 MiB cap
    files = {"file": ("big.wav", io.BytesIO(huge_payload), "audio/wav")}
    options = {
        "task": "transcribe",
        "trg_lang": "de",
        "speakers": 1,
        "words": False,
        "summarization": False,
        "hate_speech": False,
    }
    response = client.post(
        "/api/v1/jobs",
        files=files,
        data={"options": json.dumps(options)},
    )
    assert response.status_code == 413
