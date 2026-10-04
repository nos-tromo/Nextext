"""Tests for the ``/api/v1/jobs/{id}/events`` SSE endpoint."""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

import nextext.api.jobs as jobs_module
from nextext.api.jobs import PIPELINE_STAGE_LABELS, JobManager, JobState, PushEvent
from nextext.api.schemas import JobOptions

from .conftest import ALICE_OWNER_ID


def _submit(client: TestClient) -> str:
    """Submit a deterministic stub job and return its id.

    Args:
        client: TestClient bound to the stubbed app.

    Returns:
        str: The new job id.
    """
    options = {
        "task": "transcribe",
        "trg_lang": "de",
        "diarize": True,
        "words": False,
        "summarization": False,
        "hate_speech": False,
    }
    response = client.post(
        "/api/v1/jobs",
        files={"file": ("clip.wav", io.BytesIO(b"x"), "audio/wav")},
        data={"options": json.dumps(options)},
    )
    assert response.status_code == 201
    return cast(str, response.json()["job_id"])


def _parse_sse(stream: bytes) -> list[tuple[str, dict[str, Any]]]:
    """Decode an SSE byte payload into ``(event, payload)`` pairs.

    Args:
        stream: Raw bytes from the SSE response body.

    Returns:
        list[tuple[str, dict[str, Any]]]: Pairs in arrival order.
    """
    events: list[tuple[str, dict[str, Any]]] = []
    event_name = ""
    for chunk in stream.decode("utf-8").split("\n\n"):
        chunk = chunk.strip()
        if not chunk or chunk.startswith(":"):
            continue
        for line in chunk.splitlines():
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data = line[len("data:") :].strip()
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    payload = {"raw": data}
                events.append((event_name, payload))
                event_name = ""
    return events


def test_sse_stream_emits_one_pair_per_stage_then_job_completed(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """The SSE stream must produce stage_started/completed pairs and a terminal frame."""
    client, _ = stub_app_client
    job_id = _submit(client)

    # ``TestClient.stream`` keeps the response open until iteration ends, so
    # we wait for the terminal event and then break out.
    with client.stream("GET", f"/api/v1/jobs/{job_id}/events") as response:
        assert response.status_code == 200
        buffer = bytearray()
        for chunk in response.iter_bytes():
            buffer.extend(chunk)
            if b"event: job_completed" in buffer or b"event: job_failed" in buffer:
                break

    events = _parse_sse(bytes(buffer))
    stage_starts = [e for e in events if e[0] == "stage_started"]
    stage_completes = [e for e in events if e[0] == "stage_completed"]
    terminal = [e for e in events if e[0] in {"job_completed", "job_failed"}]

    assert len(stage_starts) == len(PIPELINE_STAGE_LABELS)
    assert len(stage_completes) == len(PIPELINE_STAGE_LABELS)
    assert terminal and terminal[-1][0] == "job_completed"


def test_every_event_carries_its_job_id(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """Every emitted frame (stage + terminal) must identify its job.

    The multiplexed owner stream carries events for many jobs over one
    connection, so each frame has to be self-identifying for a client to
    route it. Stage events historically omitted ``job_id``; this pins that
    every event now carries it.
    """
    client, _ = stub_app_client
    job_id = _submit(client)

    with client.stream("GET", f"/api/v1/jobs/{job_id}/events") as response:
        assert response.status_code == 200
        buffer = bytearray()
        for chunk in response.iter_bytes():
            buffer.extend(chunk)
            if b"event: job_completed" in buffer or b"event: job_failed" in buffer:
                break

    events = _parse_sse(bytes(buffer))
    assert events  # sanity: we captured something
    for name, payload in events:
        assert payload.get("job_id") == job_id, f"{name} event missing job_id"


def _completing_runner(state: JobState, push: PushEvent) -> dict[str, Any]:
    """Emit one stage pair, then finish the job normally.

    Args:
        state: The job being processed.
        push: Event sink for SSE delivery.

    Returns:
        dict[str, Any]: A minimal non-skipped result.
    """
    push("stage_started", {"stage": "stub", "stage_index": 0, "progress": 0.0})
    push("stage_completed", {"stage": "stub", "stage_index": 0, "progress": 1.0})
    return {"skipped": False}


def _failing_runner(state: JobState, push: PushEvent) -> dict[str, Any]:
    """Emit one stage start, then fail the job.

    Args:
        state: The job being processed.
        push: Event sink for SSE delivery.

    Raises:
        RuntimeError: Always, to drive the worker's failure branch.
    """
    push("stage_started", {"stage": "stub", "stage_index": 0, "progress": 0.0})
    raise RuntimeError("stub pipeline failure")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("runner", "metrics_hook", "expected_events"),
    [
        pytest.param(
            _completing_runner,
            "record_completed",
            ["stage_started", "stage_completed", "job_completed"],
            id="completed",
        ),
        pytest.param(
            _failing_runner,
            "record_failed",
            ["stage_started", "job_failed"],
            id="failed",
        ),
    ],
)
async def test_subscriber_attaching_as_job_finishes_still_gets_terminal_frame(
    runner: Any,
    metrics_hook: str,
    expected_events: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A subscriber attaching in the loop step a job finishes in gets its terminal frame.

    The worker flips ``state.status`` and emits the terminal frame in one
    step, and ``subscribe`` closes a finished job's stream once its history
    is drained. Were the frame dispatched a step later, a subscriber attaching
    in between would see a finished job, replay a history without the frame,
    and close — the intermittent terminal-less stream CI hit. The outcome
    metrics call runs inside that step, ahead of the emit, so wrapping it
    starts a subscriber there deterministically instead of by timing.
    """
    manager = JobManager(pipeline_runner=runner)
    record = getattr(jobs_module, metrics_hook)
    streams: list[asyncio.Task[bytes]] = []

    async def collect(state: JobState) -> bytes:
        """Drain one subscription to the end of its stream.

        Args:
            state: The job to subscribe to.

        Returns:
            bytes: Every frame the subscription yielded, concatenated.
        """
        return b"".join([frame async for frame in manager.subscribe(state)])

    def record_then_subscribe(*args: Any) -> None:
        """Record the outcome as usual, then attach a subscriber in this step.

        Args:
            *args: Forwarded to the real metrics hook.
        """
        record(*args)
        streams.append(asyncio.create_task(collect(job)))

    monkeypatch.setattr(jobs_module, metrics_hook, record_then_subscribe)
    # Cleanup removes the upload's parent directory, so give it its own.
    upload = tmp_path / "job" / "upload.wav"
    upload.parent.mkdir()
    upload.write_bytes(b"x")
    try:
        job = await manager.create_job(
            owner_id=ALICE_OWNER_ID,
            file_name="upload.wav",
            file_path=upload,
            source_file_hash="sha256:0",
            options=JobOptions(),
        )
        assert job.task is not None
        await job.task
        assert len(streams) == 1
        stream = await asyncio.wait_for(streams[0], timeout=5.0)
    finally:
        await manager.stop()

    assert [name for name, _ in _parse_sse(stream)] == expected_events


# The owner-multiplexed endpoint (``GET /jobs/events``) is a never-closing
# stream, which deadlocks the *sync* TestClient portal. Its route + wire
# behaviour are covered by async tests in ``test_owner_stream.py`` (driven
# through ``httpx.AsyncClient`` + ``ASGITransport``).
