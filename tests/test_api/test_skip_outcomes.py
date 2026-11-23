"""Tests for surfacing "no speech / not processed" outcomes to API clients.

A job whose upload holds no processable speech completes normally — it is not
a failure — so the only way a user learns nothing was transcribed is the typed
skip code on the snapshot, the job list, and the terminal SSE event. An
undecodable upload does fail, and carries a typed failure code so the frontend
can say "this file could not be decoded" instead of "unknown error", while the
human-readable detail stays in the logs.
"""

from __future__ import annotations

import contextlib
import io
import json
import time
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx2
import openai
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from loguru import logger
from prometheus_client import REGISTRY

from nextext.api import jobs as jobs_module
from nextext.api import main as main_module
from nextext.api.jobs import JobState, PushEvent, _run_pipeline_blocking, _serialize_result
from nextext.api.main import create_app
from nextext.api.schemas import JobOptions, JobStatus
from nextext.core import transcription as transcription_module
from nextext.core.audio import AudioDecodeError
from nextext.pipeline import TranscriptionOutcome
from tests.audit_helpers import AuditCapture, job_audit
from tests.media_factory import make_clip, make_with_cover_art

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
    app = create_app(audit_emitter_factory=AuditCapture().factory, pipeline_runner=runner)
    client = TestClient(app, raise_server_exceptions=False)
    client.headers[OWNER_HEADER] = OWNER_ID
    try:
        # The app's logging setup would remove the log capture.
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(main_module, "setup_logging", lambda: None)
            with client:
                yield client, records
    finally:
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


def test_pipeline_records_skip_reason_code_from_transcription(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    monkeypatch.setattr(jobs_module, "extract_keyframe_samples", lambda path, **kw: [])

    state = JobState(
        job_id="j-skip",
        owner_id=OWNER_ID,
        file_name="clip.wav",
        file_path=media,
        source_file_hash="sha256:x",
        options=JobOptions.model_validate({"task": "transcribe"}),
        status=JobStatus.QUEUED,
        audit=job_audit(),
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
        lambda **kwargs: TranscriptionOutcome(transcript=empty, src_lang="en", skip_reason="asr_all_segments_filtered"),
    )
    monkeypatch.setattr(jobs_module, "extract_keyframe_samples", lambda path, **kw: [])

    state = JobState(
        job_id="j-log",
        owner_id=OWNER_ID,
        file_name="clip.wav",
        file_path=media,
        source_file_hash="sha256:x",
        options=JobOptions.model_validate({"task": "transcribe"}),
        status=JobStatus.QUEUED,
        audit=job_audit(),
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


# ---------------------------------------------------------------------------
# Typed failure code for an inference outage
# ---------------------------------------------------------------------------

ECHOED = "invented upstream detail 0427"
_INFERENCE_REQUEST = httpx2.Request("POST", "http://router.invalid/v1/audio/transcriptions")


def _status_error(status_code: int) -> openai.APIStatusError:
    """Build the SDK error for an HTTP error status whose message echoes a body.

    Args:
        status_code (int): The response status.

    Returns:
        openai.APIStatusError: A status error quoting ``ECHOED``.
    """
    response = httpx2.Response(status_code, request=_INFERENCE_REQUEST, text=f"rejected: {ECHOED}")
    return openai.APIStatusError(f"Error code: {status_code} - {ECHOED}", response=response, body={"error": ECHOED})


def _connection_refused_runner(state: JobState, push: PushEvent) -> dict[str, Any]:
    """Fail as a Whisper call does when the endpoint cannot be reached.

    Args:
        state (JobState): The job being processed.
        push (PushEvent): Event sink for SSE delivery (unused).

    Raises:
        openai.APIConnectionError: Always.
    """
    raise openai.APIConnectionError(request=_INFERENCE_REQUEST)


def _wrapped_503_runner(state: JobState, push: PushEvent) -> dict[str, Any]:
    """Fail as a stage does that wraps the SDK's 503 in an error of its own.

    Args:
        state (JobState): The job being processed.
        push (PushEvent): Event sink for SSE delivery (unused).

    Raises:
        RuntimeError: Always, ``from`` a 503 ``APIStatusError``.
    """
    try:
        raise _status_error(503)
    except openai.APIStatusError as exc:
        raise RuntimeError("the stage failed") from exc


def _handled_503_runner(state: JobState, push: PushEvent) -> dict[str, Any]:
    """Fail with an error raised while handling a 503, with no explicit cause.

    Args:
        state (JobState): The job being processed.
        push (PushEvent): Event sink for SSE delivery (unused).

    Raises:
        RuntimeError: Always; the 503 is its ``__context__``.
    """
    try:
        raise _status_error(503)
    except openai.APIStatusError:
        raise RuntimeError("the stage failed")  # noqa: B904 - the implicit context is the point


def _unrelated_runner(state: JobState, push: PushEvent) -> dict[str, Any]:
    """Fail on a 503 that the raised error declares is not its cause (``from None``).

    Args:
        state (JobState): The job being processed.
        push (PushEvent): Event sink for SSE delivery (unused).

    Raises:
        RuntimeError: Always.
    """
    try:
        raise _status_error(503)
    except openai.APIStatusError:
        raise RuntimeError("the stage failed") from None


def _raising(exc_factory: Callable[[], BaseException]) -> Callable[[JobState, PushEvent], dict[str, Any]]:
    """Make a pipeline stand-in that raises what ``exc_factory`` builds.

    Args:
        exc_factory (Callable[[], BaseException]): Builds the failure.

    Returns:
        Callable[[JobState, PushEvent], dict[str, Any]]: The stand-in.
    """

    def runner(state: JobState, push: PushEvent) -> dict[str, Any]:
        raise exc_factory()

    return runner


@contextlib.contextmanager
def _client_running(runner: Any) -> Iterator[tuple[TestClient, list[str]]]:
    """Run ``with`` a client whose job manager uses ``runner``, plus captured logs.

    Args:
        runner (Any): Pipeline stand-in invoked by the job worker.

    Yields:
        tuple[TestClient, list[str]]: The HTTP client and log records.
    """
    yield from _app_client_with_runner(runner)


def _failed_count(code: str) -> float:
    """Read ``nextext_jobs_failed_total{code=...}`` from the default registry.

    Args:
        code (str): The ``code`` label value.

    Returns:
        float: The counter's value, ``0.0`` while the series does not exist.
    """
    return REGISTRY.get_sample_value("nextext_jobs_failed_total", {"code": code}) or 0.0


_FAILURE_CASES = [
    pytest.param(_connection_refused_runner, "inference_unavailable", id="connection-error"),
    pytest.param(
        _raising(lambda: openai.APITimeoutError(request=_INFERENCE_REQUEST)), "inference_unavailable", id="timeout"
    ),
    pytest.param(_raising(lambda: _status_error(503)), "inference_unavailable", id="http-503"),
    pytest.param(_raising(lambda: _status_error(429)), "inference_unavailable", id="http-429"),
    pytest.param(_wrapped_503_runner, "inference_unavailable", id="wrapped-503-cause"),
    pytest.param(_handled_503_runner, "inference_unavailable", id="wrapped-503-context"),
    pytest.param(
        _raising(lambda: ConnectionError("The configured inference provider is not reachable.")),
        "inference_unavailable",
        id="unreachable-provider",
    ),
    pytest.param(_raising(lambda: _status_error(400)), "internal", id="http-400"),
    pytest.param(_raising(lambda: _status_error(401)), "internal", id="http-401"),
    pytest.param(_raising(lambda: RuntimeError(f"bug {ECHOED}")), "internal", id="runtime-error"),
    pytest.param(_unrelated_runner, "internal", id="suppressed-context"),
    pytest.param(_decode_failure_runner, "undecodable_media", id="undecodable"),
]


@pytest.mark.parametrize(("runner", "expected_code"), _FAILURE_CASES)
def test_a_failed_job_carries_the_typed_cause_and_counts_it(runner: Any, expected_code: str) -> None:
    """Inference down is ``inference_unavailable``, a decode failure ``undecodable_media``, the rest ``internal``.

    Args:
        runner (Any): The failing pipeline stand-in.
        expected_code (str): The code the job should fail with.
    """
    before = _failed_count(expected_code)
    with _client_running(runner) as (client, _records):
        job_id = _post_job(client)
        snapshot = _wait_for_status(client, job_id, "failed")
        item = next(job for job in client.get("/api/v1/jobs").json()["jobs"] if job["job_id"] == job_id)
        body = client.get("/metrics").text

    assert snapshot["error"] == "Job failed."
    assert snapshot["error_code"] == expected_code
    assert item["error_code"] == expected_code
    assert _failed_count(expected_code) == before + 1
    assert f'nextext_jobs_failed_total{{code="{expected_code}"}}' in body
    assert ECHOED not in json.dumps(snapshot)


def test_an_inference_outage_is_logged_as_a_warning_by_class_and_status_without_frames() -> None:
    """The outage's warning names the wrapped error's class and status, not its message or a traceback."""
    with _client_running(_wrapped_503_runner) as (client, records):
        job_id = _post_job(client)
        _wait_for_status(client, job_id, "failed")

    logged = "\n".join(records)
    assert "failed: inference was not available (APIStatusError (HTTP 503))" in logged
    assert ECHOED not in logged
    assert "Traceback" not in logged and "_wrapped_503_runner" not in logged


def test_an_internal_failure_still_logs_its_frames_but_not_its_message() -> None:
    """A failure that is not an outage keeps the error-level log with frames only."""
    with _client_running(_raising(lambda: RuntimeError(f"bug {ECHOED}"))) as (client, records):
        job_id = _post_job(client)
        _wait_for_status(client, job_id, "failed")

    logged = "\n".join(records)
    assert "failed: RuntimeError" in logged
    assert ECHOED not in logged
    assert "inference was not available" not in logged


def test_the_failed_event_carries_the_inference_code() -> None:
    """The ``job_failed`` frame names ``inference_unavailable``."""
    with _client_running(_connection_refused_runner) as (client, _records):
        job_id = _post_job(client)
        _wait_for_status(client, job_id, "failed")
        with client.stream("GET", f"/api/v1/jobs/{job_id}/events") as events:
            body = ""
            for chunk in events.iter_text():
                body += chunk
                if "job_failed" in body:
                    break

    assert '"error_code": "inference_unavailable"' in body


# ---------------------------------------------------------------------------
# A video without an audio track is skipped with its frames, not failed
# ---------------------------------------------------------------------------


def _post_file_job(client: TestClient, path: Path, *, keyframes: bool) -> str:
    """Create a job over a real media file and return its id.

    Args:
        client (TestClient): Client carrying the owner header.
        path (Path): The upload.
        keyframes (bool): Whether the job asks for keyframes.

    Returns:
        str: The new job's identifier.
    """
    options = {**_options_payload(), "keyframes": keyframes}
    response = client.post(
        "/api/v1/jobs",
        files={"file": (path.name, io.BytesIO(path.read_bytes()), "video/mp4")},
        data={"options": json.dumps(options)},
    )
    assert response.status_code == 201
    return str(response.json()["job_id"])


@pytest.fixture
def real_pipeline_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Client whose jobs run the real pipeline, with no inference service behind it.

    The ``/vad`` guard fails the test if it is asked, and captioning is off, so
    nothing a job does can reach a service.

    Args:
        monkeypatch (pytest.MonkeyPatch): Sets the provider and the guard.

    Yields:
        TestClient: The HTTP client.
    """

    def _vad_must_not_be_asked(_path: Path) -> bool:
        raise AssertionError("the /vad service must not be asked about a file without audio")

    monkeypatch.setenv("INFERENCE_PROVIDER", "vllm")
    monkeypatch.setenv("OPENAI_API_BASE", "http://router.invalid/v1")
    monkeypatch.setenv("NEXTEXT_VISUAL_SUMMARY", "off")
    monkeypatch.setattr(transcription_module, "has_speech", _vad_must_not_be_asked)
    monkeypatch.setattr(main_module, "setup_logging", lambda: None)
    app = create_app(audit_emitter_factory=AuditCapture().factory)
    client = TestClient(app, raise_server_exceptions=False)
    client.headers[OWNER_HEADER] = OWNER_ID
    with client:
        yield client


def test_video_without_audio_completes_skipped_with_its_keyframes(
    real_pipeline_client: TestClient, tmp_path: Path
) -> None:
    """A screen recording is skipped as ``no_audio_stream`` and its keyframes are served.

    Args:
        real_pipeline_client (TestClient): Client running the real pipeline.
        tmp_path (Path): Where the clip is written.
    """
    client = real_pipeline_client
    clip = make_clip(tmp_path / "screen.mp4", seconds=4, fps=8, gop_size=8)
    job_id = _post_file_job(client, clip, keyframes=True)

    snapshot = _wait_for_status(client, job_id, "completed", timeout=30.0)

    assert snapshot["skipped"] is True
    assert snapshot["skip_reason_code"] == "no_audio_stream"
    assert snapshot["error_code"] is None
    item = next(job for job in client.get("/api/v1/jobs").json()["jobs"] if job["job_id"] == job_id)
    assert item["skipped"] is True
    assert item["skip_reason_code"] == "no_audio_stream"

    served = client.get(f"/api/v1/jobs/{job_id}/artifacts/keyframes.zip")
    assert served.status_code == 200
    with zipfile.ZipFile(io.BytesIO(served.content)) as archive:
        frames = [name for name in archive.namelist() if name.endswith(".jpg")]
    assert frames
    assert client.get(f"/api/v1/jobs/{job_id}/artifacts/docint.jsonl").status_code == 404
    metrics = client.get("/metrics").text
    assert 'nextext_jobs_skipped_total{reason="no_audio_stream"}' in metrics


def test_cover_art_only_file_still_fails_as_undecodable(
    real_pipeline_client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A file whose only picture is cover art is no video: it fails ``undecodable_media``, as before.

    Args:
        real_pipeline_client (TestClient): Client running the real pipeline.
        monkeypatch (pytest.MonkeyPatch): Lets the ``/vad`` guard pass, as it fails open.
        tmp_path (Path): Where the file is written.
    """
    client = real_pipeline_client
    monkeypatch.setattr(transcription_module, "has_speech", lambda _path: True)
    cover = make_with_cover_art(tmp_path / "cover.mp4")
    job_id = _post_file_job(client, cover, keyframes=True)

    snapshot = _wait_for_status(client, job_id, "failed", timeout=30.0)

    assert snapshot["error_code"] == "undecodable_media"
    assert snapshot["skipped"] is False
