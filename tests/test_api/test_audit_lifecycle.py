"""Tests for the job manager's audit events: owner records, sweeps, purges, deletes.

The job → owner mapping lives in memory, so each job directory also holds an
owner record; the startup sweep reads it to attribute its deletion of the
uploads a previous process left behind, and shutdown records the removal of
every remaining job.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

import pytest

from nextext.api import jobs as jobs_module
from nextext.api.job_record import RECORD_NAME, JobRecord, read_job_record, write_job_record
from nextext.api.jobs import JobManager, JobState, PushEvent
from nextext.api.schemas import JobOptions, JobStatus
from nextext.audit import Origin
from nextext.core.inference_trace import correlation_headers
from nextext.utils.audit_event import Actor
from tests.audit_helpers import AuditCapture, edge_origin

from .conftest import ALICE_OWNER_ID, BOB_OWNER_ID, stub_pipeline_runner

JOB_A, JOB_B, JOB_C = "a" * 32, "b" * 32, "c" * 32


def _leftover(root: Path, name: str, record: JobRecord | None) -> Path:
    """Create a job directory like a crashed process leaves it.

    Args:
        root: The job root.
        name: The directory name.
        record: Its owner record, or ``None`` for none.

    Returns:
        Path: The directory.
    """
    job_dir = root / name
    job_dir.mkdir()
    (job_dir / "upload.wav").write_bytes(b"synthetic audio")
    (job_dir / "upload-abc.flac").write_bytes(b"synthetic re-encode")
    if record is not None:
        write_job_record(job_dir, record)
    return job_dir


async def _new_job(manager: JobManager, root: Path, name: str, origin: Origin) -> JobState:
    """Create a job over an upload in a fresh job directory.

    Args:
        manager: The manager.
        root: Where the job directory goes.
        name: The directory name.
        origin: The creating request.

    Returns:
        JobState: The job.
    """
    job_dir = root / name
    job_dir.mkdir()
    upload = job_dir / "upload.wav"
    upload.write_bytes(b"synthetic audio")
    return await manager.create_job(
        owner_id=str(origin.actor.user),
        file_name="clip.wav",
        file_path=upload,
        source_file_hash="sha256:00",
        options=JobOptions(),
        origin=origin,
    )


@pytest.mark.asyncio
async def test_startup_sweep_removes_leftovers_and_records_whose_they_were(tmp_path: Path) -> None:
    """Leftover job directories are removed; the deletion names each job, grouped by owner."""
    root = tmp_path / "jobs"
    root.mkdir()
    created = "2026-01-31T12:00:00+00:00"
    _leftover(root, "job-1", JobRecord(JOB_A, "alice.example", created))
    _leftover(root, "job-2", JobRecord(JOB_B, "alice.example", created))
    _leftover(root, "job-3", JobRecord(JOB_C, "service:docint", created))
    _leftover(root, "job-4", None)
    (root / "not-a-job").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "job-link").symlink_to(outside)
    capture = AuditCapture()
    manager = JobManager(tmp_root=root, emitter=capture.started())

    await manager.start()

    assert sorted(path.name for path in root.iterdir()) == [".lock", "job-link", "not-a-job"]
    assert outside.exists()
    sweeps = capture.of("delete", "job.sweep_orphans")
    by_container = {event["target"].get("container"): event for event in sweeps}
    assert set(by_container) == {"owner:alice.example", "owner:service:docint", None}
    alice = by_container["owner:alice.example"]
    assert alice["target"] == {"kind": "job", "ids": [JOB_A, JOB_B], "count": 2, "container": "owner:alice.example"}
    assert by_container["owner:service:docint"]["target"]["ids"] == [JOB_C]
    unrecorded = by_container[None]
    assert unrecorded["target"] == {"kind": "upload", "count": 1}
    assert unrecorded["criteria"] == {"owner_recorded": False, "trigger": "startup"}
    for event in sweeps:
        assert event["actor"] == {"user": None, "via": "system"}
        assert event["outcome"] == "ok"
        assert "purpose" not in event
    await manager.stop()
    capture.emitters[0].close()
    capture.assert_chain_ok()


@pytest.mark.asyncio
async def test_a_second_manager_on_the_same_directory_sweeps_nothing(tmp_path: Path) -> None:
    """While one manager holds the job directory, another never removes its live uploads."""
    capture = AuditCapture()
    first = JobManager(tmp_root=tmp_path, emitter=capture.started())
    await first.start()
    live = _leftover(tmp_path, "job-live", JobRecord(JOB_A, "alice.example", "2026-01-31T12:00:00+00:00"))
    second = JobManager(tmp_root=tmp_path, emitter=capture.started())

    await second.start()

    assert live.exists()
    assert capture.of("delete") == []
    await second.stop()
    await first.stop()


@pytest.mark.asyncio
async def test_job_creation_writes_the_owner_record_and_records_the_upload(tmp_path: Path) -> None:
    """The owner record lands next to the upload before the job's collect event."""
    capture = AuditCapture()
    manager = JobManager(tmp_root=tmp_path / "jobs", pipeline_runner=stub_pipeline_runner, emitter=capture.started())
    state = await _new_job(manager, tmp_path, "job-x", edge_origin(ALICE_OWNER_ID))

    record = read_job_record(state.file_path.parent)
    assert record is not None
    assert (record.job_id, record.owner_id) == (state.job_id, ALICE_OWNER_ID)
    raw = json.loads((state.file_path.parent / RECORD_NAME).read_text())
    assert set(raw) == {"v", "job_id", "owner_id", "created_at"}
    collect = capture.one("collect", "job.create")
    assert collect["target"]["ids"] == [state.job_id]
    assert state.audit.collect_event_id == collect["event_id"]
    assert state.task is not None
    await state.task
    await manager.stop()


@pytest.mark.asyncio
async def test_job_creation_fails_closed_without_an_owner_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If the owner record cannot be written, the upload is removed and no job starts."""

    def fail(job_dir: Path, record: JobRecord) -> None:
        raise OSError("read-only volume")

    monkeypatch.setattr(jobs_module, "write_job_record", fail)
    capture = AuditCapture()
    manager = JobManager(tmp_root=tmp_path / "jobs", emitter=capture.started())
    with pytest.raises(OSError):
        await _new_job(manager, tmp_path, "job-y", edge_origin(ALICE_OWNER_ID))
    assert not (tmp_path / "job-y").exists()
    assert manager._jobs == {}
    assert capture.of() == []


@pytest.mark.asyncio
async def test_shutdown_records_the_removal_of_every_job_by_owner(tmp_path: Path) -> None:
    """stop() removes every job directory and records one purge per owner, by the system."""
    capture = AuditCapture()
    manager = JobManager(tmp_root=tmp_path / "jobs", pipeline_runner=stub_pipeline_runner, emitter=capture.started())
    alice = await _new_job(manager, tmp_path, "job-a", edge_origin(ALICE_OWNER_ID))
    bob = await _new_job(manager, tmp_path, "job-b", edge_origin(BOB_OWNER_ID))
    for state in (alice, bob):
        assert state.task is not None
        await state.task

    await manager.stop()

    assert not alice.file_path.parent.exists()
    assert not bob.file_path.parent.exists()
    purges = {event["target"]["container"]: event for event in capture.of("delete", "job.purge")}
    assert purges[f"owner:{ALICE_OWNER_ID}"]["target"]["ids"] == [alice.job_id]
    assert purges[f"owner:{BOB_OWNER_ID}"]["target"]["ids"] == [bob.job_id]
    assert all(event["actor"] == {"user": None, "via": "system"} for event in purges.values())
    assert all(event["criteria"] == {"trigger": "shutdown"} for event in purges.values())
    capture.emitters[0].close()
    capture.assert_chain_ok()


@pytest.mark.asyncio
async def test_delete_of_a_running_job_is_recorded_with_its_status(tmp_path: Path) -> None:
    """Deleting a running job is recorded at once; the upload goes when its pipeline step returns."""
    release = threading.Event()

    def blocking(state: JobState, push: PushEvent) -> dict[str, Any]:
        assert release.wait(5.0)
        return stub_pipeline_runner(state, push)

    capture = AuditCapture()
    manager = JobManager(tmp_root=tmp_path / "jobs", pipeline_runner=blocking, emitter=capture.started())
    state = await _new_job(manager, tmp_path, "job-r", edge_origin(ALICE_OWNER_ID))
    while state.status is not JobStatus.RUNNING:
        await asyncio.sleep(0.01)
    origin = Origin(actor=Actor(ALICE_OWNER_ID, "edge"), request_id="req-delete-1")

    assert await manager.delete(state.job_id, origin=origin, owner_id=ALICE_OWNER_ID) is True

    event = capture.one("delete", "job.delete")
    assert event["criteria"] == {"status": "running"}
    assert event["request_id"] == "req-delete-1"
    release.set()
    assert state.task is not None
    await state.task
    assert not state.file_path.parent.exists()
    assert capture.of("combine") == []
    await manager.stop()


@pytest.mark.asyncio
async def test_inference_requests_carry_the_creating_request_id(tmp_path: Path) -> None:
    """The pipeline thread sees the creating request's id as the X-Request-ID to forward."""
    seen: list[dict[str, str]] = []

    def runner(state: JobState, push: PushEvent) -> dict[str, Any]:
        seen.append(correlation_headers())
        return stub_pipeline_runner(state, push)

    manager = JobManager(tmp_root=tmp_path / "jobs", pipeline_runner=runner, emitter=AuditCapture().started())
    origin = Origin(actor=Actor(ALICE_OWNER_ID, "edge"), request_id="req-create-1")
    state = await _new_job(manager, tmp_path, "job-c", origin)
    assert state.task is not None
    await state.task
    assert seen == [{"X-Request-ID": "req-create-1"}]
    assert correlation_headers() == {}
    await manager.stop()


@pytest.mark.asyncio
async def test_job_events_after_shutdown_are_refused(tmp_path: Path) -> None:
    """Once stop() began, a job's late event raises instead of following the chain's end."""
    capture = AuditCapture()
    manager = JobManager(tmp_root=tmp_path / "jobs", pipeline_runner=stub_pipeline_runner, emitter=capture.started())
    state = await _new_job(manager, tmp_path, "job-s", edge_origin(ALICE_OWNER_ID))
    assert state.task is not None
    await state.task
    await manager.stop()
    with pytest.raises(RuntimeError):
        state.audit.combine(["transcript", "diarization"], speaker_attributed=True)
