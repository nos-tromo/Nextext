"""Tests for the ``/api/v1/jobs/{id}/events`` SSE endpoint."""

from __future__ import annotations

import io
import json
from typing import Any, cast

from fastapi.testclient import TestClient

from nextext.api.jobs import PIPELINE_STAGE_LABELS, JobManager


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


# The owner-multiplexed endpoint (``GET /jobs/events``) is a never-closing
# stream, which deadlocks the *sync* TestClient portal. Its route + wire
# behaviour are covered by async tests in ``test_owner_stream.py`` (driven
# through ``httpx.AsyncClient`` + ``ASGITransport``).
