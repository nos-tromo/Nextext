"""Tests for deleting jobs that are still queued or running.

``DELETE /jobs/{id}`` used to remove the per-job upload tempdir immediately
while leaving the worker task alive. With ``NEXTEXT_JOB_CONCURRENCY=1`` a
queued job's worker would later acquire the semaphore and run the pipeline
against the vanished upload, failing with ``FileNotFoundError`` →
``AudioDecodeError``. These tests drive ``JobManager`` directly to pin the
fixed semantics: a deleted queued job never runs its pipeline, and deleting a
running job defers tempdir removal until its pipeline thread has stopped — at
its next check point (:mod:`nextext.core.cancel`), which also frees the
worker slot instead of holding it until the pipeline would have ended.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from prometheus_client import REGISTRY

from nextext.api.jobs import JobManager, JobState, PushEvent
from nextext.api.schemas import JobOptions, JobStatus
from nextext.core.cancel import check
from tests.audit_helpers import AuditCapture, edge_origin

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
        origin=edge_origin(ALICE_OWNER_ID),
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

    manager = JobManager(pipeline_runner=runner, concurrency=1, emitter=AuditCapture().started())
    try:
        job_a = await _create_job(manager, tmp_path, "a")
        await asyncio.to_thread(first_job_started.wait, 5.0)

        job_b = await _create_job(manager, tmp_path, "b")
        assert job_b.status is JobStatus.QUEUED

        assert await manager.delete(job_b.job_id, origin=edge_origin(ALICE_OWNER_ID)) is True
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


def _outcome_count(outcome: str) -> float:
    """Read ``nextext_jobs_total{outcome=...}`` from the default registry.

    Args:
        outcome: The ``outcome`` label value.

    Returns:
        float: The counter's value, ``0.0`` while the series does not exist.
    """
    return REGISTRY.get_sample_value("nextext_jobs_total", {"outcome": outcome}) or 0.0


@pytest.mark.asyncio
async def test_delete_running_job_keeps_the_upload_until_its_next_check_point(
    tmp_path: Path,
) -> None:
    """Deleting a running job leaves the upload to the pipeline thread, which stops at its next check point."""
    job_started = threading.Event()
    release_job = threading.Event()
    ran_past_check_point = threading.Event()
    pipeline_errors: list[BaseException] = []

    def runner(state: JobState, push: PushEvent) -> dict[str, Any]:
        """Block mid-call, re-read the upload after the delete, then reach a check point."""
        job_started.set()
        assert release_job.wait(timeout=5.0)
        try:
            state.file_path.read_bytes()
        except OSError as exc:  # pragma: no cover — the bug being pinned
            pipeline_errors.append(exc)
            raise
        check()
        ran_past_check_point.set()
        return {}

    manager = JobManager(pipeline_runner=runner, concurrency=1, emitter=AuditCapture().started())
    try:
        job = await _create_job(manager, tmp_path, "running")
        await asyncio.to_thread(job_started.wait, 5.0)
        assert job.status is JobStatus.RUNNING

        assert await manager.delete(job.job_id, origin=edge_origin(ALICE_OWNER_ID)) is True
        # The upload must survive while the pipeline thread still uses it.
        assert job.file_path.exists()
        assert job.cancel.is_set()

        release_job.set()
        await _wait_for(lambda: not job.file_path.parent.exists())
        assert pipeline_errors == []
        assert not ran_past_check_point.is_set()
        assert job.status is JobStatus.CANCELLED
        assert job.finished_at is not None
    finally:
        release_job.set()
        await manager.stop()


@pytest.mark.asyncio
async def test_delete_running_job_frees_its_worker_slot_at_the_next_check_point(
    tmp_path: Path,
) -> None:
    """A job created after the delete starts before the deleted job's pipeline would have ended."""
    first_started = threading.Event()
    in_flight_call_returns = threading.Event()
    ran_past_check_point = threading.Event()
    rest_of_pipeline = threading.Event()  # never set: the work the deleted job must not do
    second_started = threading.Event()

    def runner(state: JobState, push: PushEvent) -> dict[str, Any]:
        """Run the first job as a long pipeline; the second returns at once."""
        if state.file_name != "first.mp4":
            second_started.set()
            return {}
        first_started.set()
        assert in_flight_call_returns.wait(timeout=5.0)
        check()
        ran_past_check_point.set()
        rest_of_pipeline.wait(timeout=5.0)
        return {}

    cancelled_before = _outcome_count("cancelled")
    failed_before = _outcome_count("failed")
    completed_before = _outcome_count("completed")
    manager = JobManager(pipeline_runner=runner, concurrency=1, emitter=AuditCapture().started())
    try:
        first = await _create_job(manager, tmp_path, "first")
        await asyncio.to_thread(first_started.wait, 5.0)
        assert await manager.delete(first.job_id, origin=edge_origin(ALICE_OWNER_ID)) is True

        second = await _create_job(manager, tmp_path, "second")
        assert second.status is JobStatus.QUEUED
        assert not second_started.is_set()

        in_flight_call_returns.set()
        assert await asyncio.to_thread(second_started.wait, 2.0), "the deleted job still holds the worker slot"
        await _wait_for(lambda: second.status is JobStatus.COMPLETED)

        assert not ran_past_check_point.is_set()
        assert first.status is JobStatus.CANCELLED
        assert not first.file_path.parent.exists()
        assert _outcome_count("cancelled") == cancelled_before + 1
        assert _outcome_count("failed") == failed_before
        assert _outcome_count("completed") == completed_before + 1
    finally:
        in_flight_call_returns.set()
        rest_of_pipeline.set()
        await manager.stop()


@pytest.mark.asyncio
async def test_a_cancelled_job_pushes_no_terminal_event_and_records_no_combine(tmp_path: Path) -> None:
    """The cancelled job's owner has already been told: no frame, no joined analyses."""
    job_started = threading.Event()
    release_job = threading.Event()

    def runner(state: JobState, push: PushEvent) -> dict[str, Any]:
        job_started.set()
        assert release_job.wait(timeout=5.0)
        check()
        return {}  # pragma: no cover — cancelled at the check point

    capture = AuditCapture()
    manager = JobManager(pipeline_runner=runner, concurrency=1, emitter=capture.started())
    try:
        job = await _create_job(manager, tmp_path, "cancelled")
        await asyncio.to_thread(job_started.wait, 5.0)
        await manager.delete(job.job_id, origin=edge_origin(ALICE_OWNER_ID))
        release_job.set()
        assert job.task is not None
        await job.task

        assert job.status is JobStatus.CANCELLED
        assert job.error is None
        assert job.error_code is None
        assert [name for name, _ in job.event_history] == []
        assert capture.of("combine") == []
    finally:
        release_job.set()
        await manager.stop()


@pytest.mark.asyncio
async def test_stop_cancels_a_running_job(tmp_path: Path) -> None:
    """Shutdown sets the cancel event, so the pipeline thread exits instead of running over removed uploads."""
    job_started = threading.Event()
    release_job = threading.Event()
    ran_past_check_point = threading.Event()

    def runner(state: JobState, push: PushEvent) -> dict[str, Any]:
        job_started.set()
        assert release_job.wait(timeout=5.0)
        check()
        ran_past_check_point.set()  # pragma: no cover — cancelled at the check point
        return {}

    manager = JobManager(pipeline_runner=runner, concurrency=1, emitter=AuditCapture().started())
    job = await _create_job(manager, tmp_path, "stopping")
    try:
        await asyncio.to_thread(job_started.wait, 5.0)
        assert not job.cancel.is_set()

        await manager.stop()
        assert job.cancel.is_set()

        release_job.set()
        assert job.task is not None
        await job.task
        assert not ran_past_check_point.is_set()
        assert job.status is JobStatus.CANCELLED
    finally:
        release_job.set()
