"""Integration tests for ownership-scoped job routes."""

from __future__ import annotations

import io
import json
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nextext.api.identity import IdentityMiddleware
from nextext.api.jobs import JobManager
from nextext.api.main import create_app
from nextext.api.persistence import init_repository

from tests.test_api.conftest import stub_pipeline_runner


@pytest.fixture
def persistent_app_clients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, TestClient]]:
    """Spin up a backend with a real SQLite repository and two clients.

    The two clients model two separate browsers, each holding its own
    ``nextext_session`` cookie.

    Args:
        tmp_path: Per-test temp directory used as ``NEXTEXT_DATA_DIR``.
        monkeypatch: Used to point persistence at ``tmp_path``.

    Yields:
        tuple[TestClient, TestClient]: ``(alice, bob)`` clients.
    """
    monkeypatch.setenv("NEXTEXT_DATA_DIR", str(tmp_path))

    app = create_app()
    # Replace the JobManager wired by the lifespan with one bound to our
    # stub runner so we don't need real ML deps.
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def _patched_lifespan(_app):  # type: ignore[no-untyped-def]
        repository = init_repository(db_path=tmp_path / "jobs.db")
        manager = JobManager(
            pipeline_runner=stub_pipeline_runner,
            ttl_seconds=3600,
            repository=repository,
            data_root=tmp_path,
        )
        await manager.start()
        _app.state.job_manager = manager
        _app.state.repository = repository
        try:
            yield
        finally:
            await manager.stop()
            repository.close()

    app.router.lifespan_context = _patched_lifespan
    try:
        with TestClient(app) as alice, TestClient(app) as bob:
            yield alice, bob
    finally:
        app.router.lifespan_context = original_lifespan


def _submit(client: TestClient, *, persist: bool, name: str = "clip.wav") -> str:
    """Submit a stub job and return its id.

    Args:
        client: TestClient bound to the persistent-app fixture.
        persist: Whether to opt in to durable storage.
        name: Filename used in the upload form.

    Returns:
        str: The new job id.
    """
    options = {
        "task": "transcribe",
        "trg_lang": "de",
        "speakers": 1,
        "words": False,
        "summarization": False,
        "hate_speech": False,
        "persist": persist,
    }
    files = {"file": (name, io.BytesIO(b"x"), "audio/wav")}
    response = client.post(
        "/api/v1/jobs",
        files=files,
        data={"options": json.dumps(options)},
    )
    assert response.status_code == 201, response.text
    return response.json()["job_id"]


def _wait_for_completed(client: TestClient, job_id: str) -> dict:
    """Poll ``GET /jobs/{id}`` until completion.

    Args:
        client: TestClient.
        job_id: Job identifier.

    Returns:
        dict: Final snapshot body.
    """
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}")
        if response.status_code == 200 and response.json()["status"] == "completed":
            return response.json()
        time.sleep(0.05)
    raise AssertionError(f"Job {job_id} never completed.")


def test_persistent_job_writes_db_row_and_artifacts(
    persistent_app_clients: tuple[TestClient, TestClient], tmp_path: Path
) -> None:
    """A persist=true job leaves a DB row + per-job artifact directory."""
    alice, _ = persistent_app_clients
    job_id = _submit(alice, persist=True)
    _wait_for_completed(alice, job_id)

    db_file = tmp_path / "jobs.db"
    assert db_file.is_file()
    job_dir = tmp_path / "jobs" / job_id
    assert job_dir.is_dir()
    assert (job_dir / "transcript.parquet").is_file()
    assert (job_dir / "meta.json").is_file()


def test_ephemeral_job_leaves_no_disk_trace(
    persistent_app_clients: tuple[TestClient, TestClient], tmp_path: Path
) -> None:
    """A persist=false job stores nothing under NEXTEXT_DATA_DIR."""
    alice, _ = persistent_app_clients
    job_id = _submit(alice, persist=False)
    _wait_for_completed(alice, job_id)
    job_dir = tmp_path / "jobs" / job_id
    assert not job_dir.exists()


def test_list_jobs_only_returns_owner_persistent_rows(
    persistent_app_clients: tuple[TestClient, TestClient],
) -> None:
    """Cross-owner visibility must not leak."""
    alice, bob = persistent_app_clients
    alice_id = _submit(alice, persist=True, name="alice.wav")
    bob_id = _submit(bob, persist=True, name="bob.wav")
    _wait_for_completed(alice, alice_id)
    _wait_for_completed(bob, bob_id)

    alice_list = alice.get("/api/v1/jobs").json()["jobs"]
    bob_list = bob.get("/api/v1/jobs").json()["jobs"]

    assert [j["job_id"] for j in alice_list] == [alice_id]
    assert [j["job_id"] for j in bob_list] == [bob_id]


def test_cross_owner_get_returns_404(
    persistent_app_clients: tuple[TestClient, TestClient],
) -> None:
    """One owner must never see another owner's job by id."""
    alice, bob = persistent_app_clients
    alice_id = _submit(alice, persist=True)
    _wait_for_completed(alice, alice_id)
    assert bob.get(f"/api/v1/jobs/{alice_id}").status_code == 404
    assert bob.delete(f"/api/v1/jobs/{alice_id}").status_code == 404


def test_ephemeral_jobs_do_not_appear_in_listing(
    persistent_app_clients: tuple[TestClient, TestClient],
) -> None:
    """The listing endpoint must skip in-flight ephemeral jobs."""
    alice, _ = persistent_app_clients
    persistent_id = _submit(alice, persist=True)
    ephemeral_id = _submit(alice, persist=False)
    _wait_for_completed(alice, persistent_id)
    _wait_for_completed(alice, ephemeral_id)
    listed = alice.get("/api/v1/jobs").json()["jobs"]
    ids = [j["job_id"] for j in listed]
    assert persistent_id in ids
    assert ephemeral_id not in ids


def test_persistent_artifact_download_round_trips(
    persistent_app_clients: tuple[TestClient, TestClient],
) -> None:
    """Downloading a CSV artifact after completion succeeds."""
    alice, _ = persistent_app_clients
    job_id = _submit(alice, persist=True)
    _wait_for_completed(alice, job_id)
    response = alice.get(f"/api/v1/jobs/{job_id}/artifacts/transcript.csv")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert b"Hello world." in response.content


def test_delete_removes_db_row_and_disk_dir(
    persistent_app_clients: tuple[TestClient, TestClient], tmp_path: Path
) -> None:
    """DELETE on a persistent job clears the database and disk artifacts."""
    alice, _ = persistent_app_clients
    job_id = _submit(alice, persist=True)
    _wait_for_completed(alice, job_id)
    job_dir = tmp_path / "jobs" / job_id
    assert job_dir.is_dir()
    response = alice.delete(f"/api/v1/jobs/{job_id}")
    assert response.status_code == 204
    # Allow a brief grace period for the async tempdir removal.
    time.sleep(0.05)
    assert not job_dir.exists()
    assert alice.get(f"/api/v1/jobs/{job_id}").status_code == 404


def test_rehydration_marks_running_rows_as_interrupted(tmp_path: Path) -> None:
    """A backend restart must surface stuck rows as ``interrupted``."""
    from datetime import datetime, timezone

    from nextext.api.persistence import (
        JobRecord,
        SqliteJobRepository,
        init_repository,
    )
    from nextext.api.schemas import JobOptions, JobStatus

    # Simulate "previous boot" — a row that was running when the container died.
    db_path = tmp_path / "jobs.db"
    first_repo = init_repository(db_path=db_path)
    first_repo.create(
        JobRecord(
            job_id="stuck-1",
            owner_id="alice",
            status=JobStatus.RUNNING,
            stage="Transcribing",
            stage_index=1,
            progress=0.2,
            error=None,
            file_name="long.wav",
            source_file_hash=None,
            options=JobOptions(persist=True),
            created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            started_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            finished_at=None,
            artifact_dir="jobs/stuck-1",
        )
    )
    first_repo.create(
        JobRecord(
            job_id="done-1",
            owner_id="alice",
            status=JobStatus.COMPLETED,
            stage=None,
            stage_index=4,
            progress=1.0,
            error=None,
            file_name="ok.wav",
            source_file_hash=None,
            options=JobOptions(persist=True),
            created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            started_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            finished_at=datetime(2026, 5, 1, 1, tzinfo=timezone.utc),
            artifact_dir="jobs/done-1",
        )
    )
    first_repo.close()

    # "New boot" — rehydrate from the same SQLite file.
    second_repo = SqliteJobRepository(db_path)
    second_repo.init()
    try:
        from nextext.api.jobs import JobManager

        manager = JobManager(repository=second_repo, data_root=tmp_path)
        # Drive the rehydration path on the loop the production app uses.
        import asyncio as _asyncio

        async def _bootstrap():  # type: ignore[no-untyped-def]
            await manager.start()
            return await manager.list_persistent("alice")

        states = _asyncio.run(_bootstrap())
        # Both rows are visible, but the running one is now interrupted.
        by_id = {state.job_id: state for state in states}
        assert by_id["stuck-1"].status.value == "interrupted"
        assert by_id["stuck-1"].error is not None
        assert "restart" in by_id["stuck-1"].error.lower()
        assert by_id["done-1"].status.value == "completed"
    finally:
        second_repo.close()
