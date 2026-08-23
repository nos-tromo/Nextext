"""Tests for surfacing "no speech / not processed" outcomes to API clients.

A job whose upload holds no processable speech completes normally — it is not
a failure — so the only way a user learns nothing was transcribed is the typed
skip code on the snapshot, the job list, and the terminal SSE event. An
undecodable upload does fail, and carries a typed failure code so the frontend
can say "this file could not be decoded" instead of "unknown error", while the
human-readable detail stays in the logs.
"""

from __future__ import annotations

import io
import json
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from loguru import logger

from nextext.api import jobs as jobs_module
from nextext.api.jobs import JobManager, JobState, PushEvent, _run_pipeline_blocking, _serialize_result
from nextext.api.main import create_app
from nextext.api.schemas import JobOptions, JobStatus
from nextext.core.audio import AudioDecodeError
from nextext.pipeline import TranscriptionOutcome

OWNER_HEADER = "X-Auth-User"
OWNER_ID = "c" * 32
MARKER = "MARKER-DECODE-9876"


def _options_payload() -> dict[str, Any]:
    """Return a minimal valid job-options payload.

    Returns:
        dict[str, Any]: Options with every optional stage switched off.
    """
    return {
        "task": "transcribe",
        "trg_lang": "de",
        "diarize": False,
        "words": False,
        "summarization": False,
        "hate_speech": False,
    }


def _post_job(client: TestClient) -> str:
    """Create a job and return its id.

    Args:
        client (TestClient): Client carrying the owner header.

    Returns:
        str: The new job's identifier.
    """
    response = client.post(
        "/api/v1/jobs",
        files={"file": ("clip.wav", io.BytesIO(b"audio-bytes"), "audio/wav")},
        data={"options": json.dumps(_options_payload())},
    )
    assert response.status_code == 201
    return str(response.json()["job_id"])


def _wait_for_status(client: TestClient, job_id: str, target: str, timeout: float = 5.0) -> dict[str, Any]:
    """Poll ``GET /jobs/{id}`` until the status matches ``target``.

    Args:
        client (TestClient): TestClient to poll with.
        job_id (str): Job identifier.
        target (str): Desired status string.
        timeout (float): Max seconds to wait.

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
    raise AssertionError(f"Job '{job_id}' never reached status '{target}'. Last seen: {last}")


def _app_client_with_runner(runner: Any) -> Iterator[tuple[TestClient, list[str]]]:
    """Yield a client whose job manager uses ``runner``, plus captured logs.

    Args:
        runner (Any): Pipeline stand-in invoked by the job worker.

    Yields:
        tuple[TestClient, list[str]]: The HTTP client and log records.
    """
    records: list[str] = []
    sink_id = logger.add(lambda m: records.append(str(m)), level="DEBUG")
    app = create_app()
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def _patched_lifespan(_app: FastAPI) -> AsyncIterator[None]:
        manager = JobManager(pipeline_runner=runner)
        await manager.start()
        _app.state.job_manager = manager
        try:
            yield
        finally:
            await manager.stop()

    app.router.lifespan_context = _patched_lifespan
    client = TestClient(app, raise_server_exceptions=False)
    client.headers[OWNER_HEADER] = OWNER_ID
    try:
        with client:
            yield client, records
    finally:
        app.router.lifespan_context = original_lifespan
        logger.remove(sink_id)


def _skipping_runner(state: JobState, push: PushEvent) -> dict[str, Any]:
    """Pipeline stand-in returning the skip payload the real worker builds.

    Args:
        state (JobState): The job being processed.
        push (PushEvent): Event sink for SSE delivery (unused).

    Returns:
        dict[str, Any]: A skipped result payload.
    """
    return {
        "transcript": pd.DataFrame({"start": [], "end": [], "text": []}),
        "summary": None,
        "word_counts": None,
        "named_entities": None,
        "wordcloud": None,
        "hate_speech_findings": None,
        "resolved_src_lang": "en",
        "transcript_language": "en",
        "skipped": True,
        "skip_reason": "No speech detected in the audio.",
        "skip_reason_code": "vad_no_speech",
        "task": "transcribe",
        "keyframes": [],
    }


def _decode_failure_runner(state: JobState, push: PushEvent) -> dict[str, Any]:
    """Pipeline stand-in that fails the way an undecodable upload does.

    Args:
        state (JobState): The job being processed.
        push (PushEvent): Event sink for SSE delivery (unused).

    Returns:
        dict[str, Any]: Never returns; always raises.

    Raises:
        AudioDecodeError: Always.
    """
    raise AudioDecodeError(f"Could not decode audio file '{MARKER}'.")


@pytest.fixture
def skipping_app_client() -> Iterator[tuple[TestClient, list[str]]]:
    """Client whose pipeline always reports a skipped job.

    Yields:
        tuple[TestClient, list[str]]: The HTTP client and log records.
    """
    yield from _app_client_with_runner(_skipping_runner)


@pytest.fixture
def decode_failure_app_client() -> Iterator[tuple[TestClient, list[str]]]:
    """Client whose pipeline always raises ``AudioDecodeError``.

    Yields:
        tuple[TestClient, list[str]]: The HTTP client and log records.
    """
    yield from _app_client_with_runner(_decode_failure_runner)


# ---------------------------------------------------------------------------
# The worker records the typed cause
# ---------------------------------------------------------------------------


def test_pipeline_records_skip_reason_code_from_transcription(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The worker must carry the transcription outcome's code into the result.

    Args:
        monkeypatch (pytest.MonkeyPatch): Overrides the transcription stage.
        tmp_path (Path): Temporary upload location.
    """
    media = tmp_path / "clip.wav"
    media.write_bytes(b"audio")
    empty = pd.DataFrame({"start": [], "end": [], "speaker": [], "text": []})
    monkeypatch.setattr(
        "nextext.pipeline.transcription_pipeline",
        lambda **kwargs: TranscriptionOutcome(transcript=empty, src_lang="en", skip_reason="vad_no_speech"),
    )
    monkeypatch.setattr(jobs_module, "extract_keyframes", lambda path, **kw: [])

    state = JobState(
        job_id="j-skip",
        owner_id=OWNER_ID,
        file_name="clip.wav",
        file_path=media,
        source_file_hash="sha256:x",
        options=JobOptions.model_validate({"task": "transcribe"}),
        status=JobStatus.QUEUED,
    )

    result = _run_pipeline_blocking(state, lambda *a, **k: None)

    assert result["skipped"] is True
    assert result["skip_reason_code"] == "vad_no_speech"
    assert result["skip_reason"] == "No speech detected in the audio."
    assert _serialize_result(result).skip_reason_code == "vad_no_speech"


def test_pipeline_logs_the_skipped_outcome(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A skipped job must leave a job-scoped log line for the operator.

    Args:
        monkeypatch (pytest.MonkeyPatch): Overrides the transcription stage.
        tmp_path (Path): Temporary upload location.
    """
    media = tmp_path / "clip.wav"
    media.write_bytes(b"audio")
    empty = pd.DataFrame({"start": [], "end": [], "speaker": [], "text": []})
    monkeypatch.setattr(
        "nextext.pipeline.transcription_pipeline",
        lambda **kwargs: TranscriptionOutcome(
            transcript=empty, src_lang="en", skip_reason="asr_all_segments_filtered"
        ),
    )
    monkeypatch.setattr(jobs_module, "extract_keyframes", lambda path, **kw: [])

    state = JobState(
        job_id="j-log",
        owner_id=OWNER_ID,
        file_name="clip.wav",
        file_path=media,
        source_file_hash="sha256:x",
        options=JobOptions.model_validate({"task": "transcribe"}),
        status=JobStatus.QUEUED,
    )

    records: list[str] = []
    sink_id = logger.add(lambda m: records.append(str(m)), level="DEBUG")
    try:
        _run_pipeline_blocking(state, lambda *a, **k: None)
    finally:
        logger.remove(sink_id)

    assert any("j-log" in r and "asr_all_segments_filtered" in r for r in records)


# ---------------------------------------------------------------------------
# The API surfaces it
# ---------------------------------------------------------------------------


def test_snapshot_exposes_skip_fields_at_top_level(
    skipping_app_client: tuple[TestClient, list[str]],
) -> None:
    """A skipped job's snapshot must carry ``skipped`` and its code.

    Args:
        skipping_app_client (tuple[TestClient, list[str]]): Client + logs.
    """
    client, _ = skipping_app_client
    job_id = _post_job(client)

    snapshot = _wait_for_status(client, job_id, "completed")

    assert snapshot["skipped"] is True
    assert snapshot["skip_reason_code"] == "vad_no_speech"
    assert snapshot["error_code"] is None
    assert snapshot["result"]["skip_reason_code"] == "vad_no_speech"


def test_job_list_exposes_skip_fields(skipping_app_client: tuple[TestClient, list[str]]) -> None:
    """``GET /jobs`` must carry the skip flag so it survives a browser reload.

    The list is the only source the SPA has after a reload — without the
    flag here, a skipped job reads as a plain "Done".

    Args:
        skipping_app_client (tuple[TestClient, list[str]]): Client + logs.
    """
    client, _ = skipping_app_client
    job_id = _post_job(client)
    _wait_for_status(client, job_id, "completed")

    listing = client.get("/api/v1/jobs")

    assert listing.status_code == 200
    item = next(job for job in listing.json()["jobs"] if job["job_id"] == job_id)
    assert item["skipped"] is True
    assert item["skip_reason_code"] == "vad_no_speech"


def test_terminal_event_carries_the_skip_code(skipping_app_client: tuple[TestClient, list[str]]) -> None:
    """The ``job_completed`` frame must name the reason, not just the flag.

    Args:
        skipping_app_client (tuple[TestClient, list[str]]): Client + logs.
    """
    client, _ = skipping_app_client
    job_id = _post_job(client)
    _wait_for_status(client, job_id, "completed")

    with client.stream("GET", f"/api/v1/jobs/{job_id}/events") as stream:
        body = ""
        for chunk in stream.iter_text():
            body += chunk
            if "job_completed" in body:
                break

    assert '"skipped": true' in body
    assert '"skip_reason_code": "vad_no_speech"' in body


def test_skipped_job_archive_is_not_offered(skipping_app_client: tuple[TestClient, list[str]]) -> None:
    """``archive.zip`` must 404 for a skipped job like every sibling artifact.

    Args:
        skipping_app_client (tuple[TestClient, list[str]]): Client + logs.
    """
    client, _ = skipping_app_client
    job_id = _post_job(client)
    _wait_for_status(client, job_id, "completed")

    assert client.get(f"/api/v1/jobs/{job_id}/artifacts/transcript.csv").status_code == 404
    assert client.get(f"/api/v1/jobs/{job_id}/artifacts/archive.zip").status_code == 404


# ---------------------------------------------------------------------------
# Typed failure code for undecodable media
# ---------------------------------------------------------------------------


def test_decode_failure_reports_typed_code_but_generic_message(
    decode_failure_app_client: tuple[TestClient, list[str]],
) -> None:
    """An undecodable upload is a user-actionable failure with a typed code.

    The message stays the static ``"Job failed."`` — the filename and the
    decoder's detail belong in the logs only.

    Args:
        decode_failure_app_client (tuple[TestClient, list[str]]): Client + logs.
    """
    client, records = decode_failure_app_client
    job_id = _post_job(client)

    snapshot = _wait_for_status(client, job_id, "failed")

    assert snapshot["error"] == "Job failed."
    assert snapshot["error_code"] == "undecodable_media"
    assert MARKER not in json.dumps(snapshot)
    assert any(MARKER in r for r in records)

    listing = client.get("/api/v1/jobs")
    item = next(job for job in listing.json()["jobs"] if job["job_id"] == job_id)
    assert item["error_code"] == "undecodable_media"


def test_decode_failure_event_carries_the_code(
    decode_failure_app_client: tuple[TestClient, list[str]],
) -> None:
    """The ``job_failed`` frame must name the typed cause.

    Args:
        decode_failure_app_client (tuple[TestClient, list[str]]): Client + logs.
    """
    client, _ = decode_failure_app_client
    job_id = _post_job(client)
    _wait_for_status(client, job_id, "failed")

    with client.stream("GET", f"/api/v1/jobs/{job_id}/events") as stream:
        body = ""
        for chunk in stream.iter_text():
            body += chunk
            if "job_failed" in body:
                break

    assert '"error_code": "undecodable_media"' in body


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_metrics_count_skipped_jobs_by_reason(
    skipping_app_client: tuple[TestClient, list[str]],
) -> None:
    """A skipped job increments its outcome and reason counters.

    Args:
        skipping_app_client (tuple[TestClient, list[str]]): Client + logs.
    """
    client, _ = skipping_app_client
    job_id = _post_job(client)
    _wait_for_status(client, job_id, "completed")

    body = client.get("/metrics").text

    assert 'nextext_jobs_total{outcome="skipped"}' in body
    assert 'nextext_jobs_skipped_total{reason="vad_no_speech"}' in body


def test_metrics_count_failed_jobs_by_code(
    decode_failure_app_client: tuple[TestClient, list[str]],
) -> None:
    """A failed job increments its outcome and typed-code counters.

    Args:
        decode_failure_app_client (tuple[TestClient, list[str]]): Client + logs.
    """
    client, _ = decode_failure_app_client
    job_id = _post_job(client)
    _wait_for_status(client, job_id, "failed")

    body = client.get("/metrics").text

    assert 'nextext_jobs_total{outcome="failed"}' in body
    assert 'nextext_jobs_failed_total{code="undecodable_media"}' in body
