"""A job's pipeline runs under its owner's cache scope.

The chat model's prefix cache is partitioned per ``X-Cache-Scope`` by the
router, so every chat call a job makes must carry the scope of the job's
owner: the same scope for every job of one owner, another one for every other
owner. These tests drive the manager directly with a runner that records the
scope it runs under.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest

from nextext.api.jobs import JobManager, JobState, PushEvent
from nextext.api.schemas import JobOptions
from nextext.core.inference_trace import cache_scope, cache_scope_headers
from tests.audit_helpers import AuditCapture, edge_origin

from .conftest import ALICE_OWNER_ID, BOB_OWNER_ID, stub_pipeline_runner


class _ScopeRecorder:
    """Pipeline runner that records the cache-scope headers of each job's run."""

    def __init__(self, expected_runs: int) -> None:
        """Start empty.

        Args:
            expected_runs: How many runs complete the recording.
        """
        self.seen: dict[str, dict[str, str]] = {}
        self._expected = expected_runs
        self._lock = threading.Lock()
        self.done = threading.Event()

    def __call__(self, state: JobState, push: PushEvent) -> dict[str, Any]:
        """Record the scope in effect, then run the stub pipeline.

        Args:
            state: The job.
            push: The event sink.

        Returns:
            dict[str, Any]: The stub pipeline's result.
        """
        with self._lock:
            self.seen[state.job_id] = cache_scope_headers()
            if len(self.seen) == self._expected:
                self.done.set()
        return stub_pipeline_runner(state, push)


async def _create_job(manager: JobManager, owner_id: str, tmp_path: Path, name: str) -> JobState:
    """Register a stub job for ``owner_id`` and return its state.

    Args:
        manager: The manager under test.
        owner_id: Owner principal for the job.
        tmp_path: Pytest temp dir for the upload placeholder.
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
        origin=edge_origin(owner_id),
    )


def _scope_of(owner_id: str) -> dict[str, str]:
    """Return the cache-scope headers bound for ``owner_id``.

    Args:
        owner_id: The owner.

    Returns:
        dict[str, str]: The headers a chat call of that owner carries.
    """
    with cache_scope(owner_id):
        return cache_scope_headers()


@pytest.mark.asyncio
async def test_each_job_runs_under_its_owners_cache_scope(tmp_path: Path) -> None:
    """Two jobs of Alice share her scope; Bob's job runs under his own."""
    recorder = _ScopeRecorder(expected_runs=3)
    manager = JobManager(pipeline_runner=recorder, concurrency=2, emitter=AuditCapture().started())
    try:
        alice_one = await _create_job(manager, ALICE_OWNER_ID, tmp_path, "alice-one")
        alice_two = await _create_job(manager, ALICE_OWNER_ID, tmp_path, "alice-two")
        bob = await _create_job(manager, BOB_OWNER_ID, tmp_path, "bob")
        assert await asyncio.to_thread(recorder.done.wait, 5.0)
    finally:
        await manager.stop()

    assert recorder.seen[alice_one.job_id] == _scope_of(ALICE_OWNER_ID)
    assert recorder.seen[alice_two.job_id] == _scope_of(ALICE_OWNER_ID)
    assert recorder.seen[bob.job_id] == _scope_of(BOB_OWNER_ID)
    assert _scope_of(ALICE_OWNER_ID) != _scope_of(BOB_OWNER_ID)
    assert cache_scope_headers() == {}
