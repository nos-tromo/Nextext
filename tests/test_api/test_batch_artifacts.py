"""Tests for the batch artifact endpoints ``/api/v1/jobs/batch/{name}``."""

from __future__ import annotations

import io
import json
import time
import zipfile
from typing import Any, cast

from fastapi.testclient import TestClient

from nextext.api.jobs import JobManager

from .conftest import BOB_OWNER_ID, OWNER_HEADER

_OPTIONS: dict[str, Any] = {
    "task": "transcribe",
    "trg_lang": "de",
    "speakers": 1,
    "words": False,
    "summarization": False,
    "hate_speech": False,
}


def _submit(client: TestClient, name: str, headers: dict[str, str] | None = None) -> str:
    """Submit a stub job and return its id.

    Args:
        client: TestClient bound to the stub-app fixture.
        name: Upload filename.
        headers: Optional per-request header overrides (e.g. a different owner).

    Returns:
        str: The new job id.
    """
    response = client.post(
        "/api/v1/jobs",
        files={"file": (name, io.BytesIO(b"x"), "audio/wav")},
        data={"options": json.dumps(_OPTIONS)},
        headers=headers,
    )
    assert response.status_code == 201, response.text
    return cast(str, response.json()["job_id"])


def _wait(client: TestClient, job_id: str, headers: dict[str, str] | None = None) -> None:
    """Block until a job completes.

    Args:
        client: TestClient.
        job_id: Job identifier.
        headers: Optional per-request header overrides.
    """
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        snapshot = client.get(f"/api/v1/jobs/{job_id}", headers=headers).json()
        if snapshot["status"] == "completed":
            return
        time.sleep(0.05)
    raise AssertionError(f"Job {job_id} did not complete in time.")


def _jsonl_records(content: bytes) -> list[dict[str, Any]]:
    """Decode an NDJSON response body into records.

    Args:
        content: Raw response bytes.

    Returns:
        list[dict[str, Any]]: One dict per non-empty line.
    """
    return [json.loads(line) for line in content.decode("utf-8").splitlines() if line]


def test_batch_docint_concatenates_completed_jobs(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """The combined JSONL spans every completed job's segments."""
    client, _ = stub_app_client
    first = _submit(client, "a.wav")
    second = _submit(client, "b.wav")
    _wait(client, first)
    _wait(client, second)

    response = client.get("/api/v1/jobs/batch/docint.jsonl")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")

    records = _jsonl_records(response.content)
    # Each stub transcript has two segments; two jobs => four lines.
    assert len(records) == 4
    assert {r["source_file"] for r in records} == {"a.wav", "b.wav"}
    for record in records:
        assert record["language"] == "en"


def test_batch_archive_nests_jobs_in_distinct_folders(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """Two same-named uploads land in distinct top-level folders."""
    client, _ = stub_app_client
    first = _submit(client, "clip.wav")
    second = _submit(client, "clip.wav")
    _wait(client, first)
    _wait(client, second)

    response = client.get("/api/v1/jobs/batch/archive.zip")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    names = archive.namelist()
    folders = {name.split("/", 1)[0] for name in names}
    assert folders == {"clip", "clip_2"}
    for folder in folders:
        assert any(name.startswith(f"{folder}/") and name.endswith("_transcript.csv") for name in names)


def test_batch_docint_returns_404_when_no_completed_jobs(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """Combined JSONL with no completed jobs is a 404, not an empty body."""
    client, _ = stub_app_client
    response = client.get("/api/v1/jobs/batch/docint.jsonl")
    assert response.status_code == 404


def test_batch_archive_returns_404_when_no_completed_jobs(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """Combined ZIP with no completed jobs is a 404, not an empty body."""
    client, _ = stub_app_client
    response = client.get("/api/v1/jobs/batch/archive.zip")
    assert response.status_code == 404


def test_batch_unknown_artifact_returns_404(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """An unsupported batch artifact name yields a 404."""
    client, _ = stub_app_client
    response = client.get("/api/v1/jobs/batch/secrets.tar")
    assert response.status_code == 404


def test_batch_docint_is_owner_scoped(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """Each owner's combined JSONL contains only their own jobs."""
    client, _ = stub_app_client
    bob_headers: dict[str, str] = {OWNER_HEADER: BOB_OWNER_ID}

    alice_first = _submit(client, "alice1.wav")
    alice_second = _submit(client, "alice2.wav")
    bob_only = _submit(client, "bob1.wav", headers=bob_headers)
    _wait(client, alice_first)
    _wait(client, alice_second)
    _wait(client, bob_only, headers=bob_headers)

    alice_records = _jsonl_records(client.get("/api/v1/jobs/batch/docint.jsonl").content)
    bob_records = _jsonl_records(client.get("/api/v1/jobs/batch/docint.jsonl", headers=bob_headers).content)

    assert {r["source_file"] for r in alice_records} == {"alice1.wav", "alice2.wav"}
    assert {r["source_file"] for r in bob_records} == {"bob1.wav"}
