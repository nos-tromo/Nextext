"""Tests for the owner-multiplexed SSE fan-out (``JobManager.subscribe_owner``).

One browser opens a single ``GET /jobs/events`` stream that carries events for
every job it owns, so a large batch uses one connection instead of one per job.
These tests drive the manager directly (async) for precise control over event
ordering — replay, live multiplexing, owner scoping, and jobs created after the
stream is already open.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from nextext.api.jobs import JobManager, JobState
from nextext.api.main import create_app
from nextext.api.schemas import JobOptions, JobStatus

from .conftest import ALICE_OWNER_ID, BOB_OWNER_ID, stub_pipeline_runner


def _parse_frames(buffer: bytes) -> list[tuple[str, dict[str, Any]]]:
    """Decode accumulated SSE bytes into ``(event, payload)`` pairs.

    Args:
        buffer: Raw bytes accumulated from an SSE stream.

    Returns:
        list[tuple[str, dict[str, Any]]]: Parsed events in arrival order,
            skipping comment/ping frames.
    """
    events: list[tuple[str, dict[str, Any]]] = []
    for chunk in buffer.decode("utf-8").split("\n\n"):
        chunk = chunk.strip()
        if not chunk or chunk.startswith(":"):
            continue
        name = ""
        for line in chunk.splitlines():
            if line.startswith("event:"):
                name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                events.append((name, json.loads(line[len("data:") :].strip())))
    return events


async def _create_job(manager: JobManager, owner_id: str, tmp_path: Path, name: str) -> JobState:
    """Register a stub job for ``owner_id`` and return its state.

    Args:
        manager: The manager under test.
        owner_id: Owner principal for the job.
        tmp_path: Pytest temp dir for the (unused) upload placeholder.
        name: Distinct file name / dir component.

    Returns:
        JobState: The newly created job state.
    """
    job_dir = tmp_path / name
    job_dir.mkdir(parents=True, exist_ok=True)
    file_path = job_dir / "upload.wav"
    file_path.write_bytes(b"x")
    return await manager.create_job(
        owner_id=owner_id,
        file_name=f"{name}.wav",
        file_path=file_path,
        source_file_hash="sha256:0",
        options=JobOptions(),
    )


async def _collect(agen: Any, want_completed: int, timeout: float = 5.0) -> list[tuple[str, dict[str, Any]]]:
    """Read an owner SSE async-generator until ``want_completed`` terminals.

    Args:
        agen: The ``subscribe_owner`` async generator.
        want_completed: Stop once this many ``job_completed`` frames arrive.
        timeout: Overall wall-clock budget before giving up.

    Returns:
        list[tuple[str, dict[str, Any]]]: All parsed events seen so far.
    """

    async def _run() -> list[tuple[str, dict[str, Any]]]:
        buffer = bytearray()
        async for frame in agen:
            buffer.extend(frame)
            events = _parse_frames(bytes(buffer))
            if sum(1 for name, _ in events if name == "job_completed") >= want_completed:
                return events
        return _parse_frames(bytes(buffer))

    try:
        return await asyncio.wait_for(_run(), timeout)
    finally:
        await agen.aclose()


@pytest.mark.asyncio
async def test_owner_stream_multiplexes_and_stays_open_across_completions(tmp_path: Path) -> None:
    """One owner stream carries events for two jobs and does not close on the first completion."""
    manager = JobManager(pipeline_runner=stub_pipeline_runner, concurrency=2)
    try:
        job_a = await _create_job(manager, ALICE_OWNER_ID, tmp_path, "a")
        job_b = await _create_job(manager, ALICE_OWNER_ID, tmp_path, "b")

        events = await _collect(manager.subscribe_owner(ALICE_OWNER_ID), want_completed=2)

        job_ids = {payload["job_id"] for _, payload in events}
        assert job_ids == {job_a.job_id, job_b.job_id}
        completed = {payload["job_id"] for name, payload in events if name == "job_completed"}
        assert completed == {job_a.job_id, job_b.job_id}
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_owner_stream_is_owner_scoped(tmp_path: Path) -> None:
    """Bob's stream never carries Alice's job events."""
    manager = JobManager(pipeline_runner=stub_pipeline_runner, concurrency=2)
    try:
        alice_job = await _create_job(manager, ALICE_OWNER_ID, tmp_path, "alice")
        bob_job = await _create_job(manager, BOB_OWNER_ID, tmp_path, "bob")

        events = await _collect(manager.subscribe_owner(BOB_OWNER_ID), want_completed=1)

        seen = {payload["job_id"] for _, payload in events}
        assert bob_job.job_id in seen
        assert alice_job.job_id not in seen
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_owner_stream_replays_history_for_existing_jobs(tmp_path: Path) -> None:
    """A stream opened after a job finished still replays that job's full history."""
    manager = JobManager(pipeline_runner=stub_pipeline_runner, concurrency=2)
    try:
        job = await _create_job(manager, ALICE_OWNER_ID, tmp_path, "done")
        # Let the worker run to completion before subscribing.
        for _ in range(1000):
            if job.status is JobStatus.COMPLETED:
                break
            await asyncio.sleep(0.005)
        assert job.status is JobStatus.COMPLETED

        events = await _collect(manager.subscribe_owner(ALICE_OWNER_ID), want_completed=1)
        names = [name for name, _ in events]
        assert names.count("stage_started") >= 1
        assert "job_completed" in names
    finally:
        await manager.stop()


def test_owner_events_route_streams_via_subscribe_owner() -> None:
    """``GET /jobs/events`` resolves to the owner stream and streams its frames.

    The literal ``events`` path must beat the ``/{job_id}`` pattern (which
    would 404 for a job named 'events'). The real ``subscribe_owner`` never
    closes on its own — which deadlocks the in-process test transports — so
    here it is swapped for a bounded generator to exercise pure route wiring;
    the fan-out behaviour itself is covered by the async manager-level tests.
    """
    app = create_app()
    manager = JobManager(pipeline_runner=stub_pipeline_runner)

    async def _bounded(owner_id: str) -> AsyncIterator[bytes]:
        yield b'event: stage_started\ndata: {"job_id": "j0", "stage": "Transcribing"}\n\n'
        yield b'event: job_completed\ndata: {"job_id": "j0", "skipped": false}\n\n'

    manager.subscribe_owner = _bounded  # type: ignore[method-assign]
    app.state.job_manager = manager

    client = TestClient(app)
    client.headers["X-Auth-User"] = ALICE_OWNER_ID
    with client.stream("GET", "/api/v1/jobs/events") as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        body = b"".join(response.iter_bytes())

    assert b"event: stage_started" in body
    assert b"event: job_completed" in body


@pytest.mark.asyncio
async def test_owner_stream_delivers_job_created_after_connect(tmp_path: Path) -> None:
    """A job created *after* the owner stream opens still streams its events."""
    manager = JobManager(pipeline_runner=stub_pipeline_runner, concurrency=2)
    try:
        agen = manager.subscribe_owner(ALICE_OWNER_ID)
        collector = asyncio.create_task(_collect(agen, want_completed=1))

        # Wait until the subscriber is registered, then create a new job.
        for _ in range(1000):
            if manager._owner_subscribers.get(ALICE_OWNER_ID):
                break
            await asyncio.sleep(0)
        assert manager._owner_subscribers.get(ALICE_OWNER_ID)

        late_job = await _create_job(manager, ALICE_OWNER_ID, tmp_path, "late")
        events = await collector

        assert late_job.job_id in {payload["job_id"] for _, payload in events}
    finally:
        await manager.stop()
