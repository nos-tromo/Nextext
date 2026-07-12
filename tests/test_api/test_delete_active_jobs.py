"""Tests for deleting jobs that are still queued or running.

``DELETE /jobs/{id}`` used to remove the per-job upload tempdir immediately
while leaving the worker task alive. With ``NEXTEXT_JOB_CONCURRENCY=1`` a
queued job's worker would later acquire the semaphore and run the pipeline
against the vanished upload, failing with ``FileNotFoundError`` →
``AudioDecodeError``. These tests drive ``JobManager`` directly to pin the
fixed semantics: a deleted queued job never runs its pipeline, and deleting a
running job defers tempdir removal until the pipeline thread has finished.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from nextext.api.jobs import JobManager, JobState, PushEvent
from nextext.api.schemas import JobOptions, JobStatus

from .conftest import ALICE_OWNER_ID


async def _create_job(manager: JobManager, tmp_path: Path, name: str) -> JobState:
    """Register a job whose upload lives in its own per-job directory.

    Args:
        manager: The manager under test.
        tmp_path: Pytest temp dir used as the parent for job dirs.
        name: Distinct file name / dir component.

    Returns:
        JobState: The newly created job state.
    """
    job_dir = tmp_path / name
    job_dir.mkdir(parents=True, exist_ok=True)
    file_path = job_dir / "upload.mp4"
    file_path.write_bytes(b"x")
    return await manager.create_job(
        owner_id=ALICE_OWNER_ID,
        file_name=f"{name}.mp4",
        file_path=file_path,
        source_file_hash="sha256:0",
        options=JobOptions(),
    )


async def _wait_for(predicate: Any, timeout: float = 5.0) -> None:
    """Poll ``predicate`` on the event loop until truthy or ``timeout``.

    Args:
        predicate: Zero-arg callable evaluated between short sleeps.
        timeout: Max seconds to wait.

    Raises:
        AssertionError: If the deadline elapses first.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("Condition never became true.")


@pytest.mark.asyncio
async def test_delete_queued_job_never_runs_its_pipeline(tmp_path: Path) -> None:
    """Deleting a queued job cancels its worker before the pipeline touches disk."""
    first_job_started = threading.Event()
    release_first_job = threading.Event()
    ran_job_ids: list[str] = []

    def runner(state: JobState, push: PushEvent) -> dict[str, Any]:
        """Record the job, prove its upload exists, and block until released."""
        ran_job_ids.append(state.job_id)
        assert state.file_path.exists(), "pipeline ran against a deleted upload"
        first_job_started.set()
        assert release_first_job.wait(timeout=5.0)
        return {}

    manager = JobManager(pipeline_runner=runner, concurrency=1)
    try:
        job_a = await _create_job(manager, tmp_path, "a")
        await asyncio.to_thread(first_job_started.wait, 5.0)

        job_b = await _create_job(manager, tmp_path, "b")
        assert job_b.status is JobStatus.QUEUED

        assert await manager.delete(job_b.job_id) is True
        assert not job_b.file_path.parent.exists()

        release_first_job.set()
        await _wait_for(lambda: job_a.status is JobStatus.COMPLETED)
        # Give the (buggy) queued worker a chance to run before asserting.
        await asyncio.sleep(0.1)

        assert ran_job_ids == [job_a.job_id]
        assert job_b.status is not JobStatus.FAILED
    finally:
        release_first_job.set()
        await manager.stop()


@pytest.mark.asyncio
async def test_delete_running_job_defers_dir_cleanup_until_worker_finishes(
    tmp_path: Path,
) -> None:
    """Deleting a running job keeps its upload alive for the pipeline thread."""
    job_started = threading.Event()
    release_job = threading.Event()
    pipeline_errors: list[BaseException] = []

    def runner(state: JobState, push: PushEvent) -> dict[str, Any]:
        """Block mid-pipeline, then re-read the upload after the delete."""
        job_started.set()
        assert release_job.wait(timeout=5.0)
        try:
            state.file_path.read_bytes()
        except OSError as exc:  # pragma: no cover — the bug being pinned
            pipeline_errors.append(exc)
            raise
        return {}

    manager = JobManager(pipeline_runner=runner, concurrency=1)
    try:
        job = await _create_job(manager, tmp_path, "running")
        await asyncio.to_thread(job_started.wait, 5.0)
        assert job.status is JobStatus.RUNNING

        assert await manager.delete(job.job_id) is True
        # The upload must survive while the pipeline thread still uses it.
        assert job.file_path.exists()

        release_job.set()
        await _wait_for(lambda: not job.file_path.parent.exists())
        assert pipeline_errors == []
    finally:
        release_job.set()
        await manager.stop()
