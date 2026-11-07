"""Tests for the face stage in the job pipeline and the API that exposes it.

The stage runs after the keyframes and before the no-speech short-circuit, is
gated by the per-job ``faces`` option and the operator switch
``NEXTEXT_FACES``, never fails the job, and surfaces thumbnails only in the
finished snapshot. ``faces_pipeline`` is stubbed: no frame is decoded and no
face is involved.
"""

from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from nextext.api import main as main_module
from nextext.api.jobs import (
    PIPELINE_STAGE_LABELS,
    STAGE_FACES,
    JobState,
    PushEvent,
    _run_pipeline_blocking,
)
from nextext.api.main import create_app
from nextext.api.schemas import JobOptions, JobStatus
from nextext.core.face_extraction import DetectionRow, FaceCrop, FaceEntry, FacesResult
from nextext.core.inference_trace import record_inference
from nextext.pipeline import TranscriptionOutcome
from nextext.utils.env_cfg import FacesMode
from tests.audit_helpers import AuditCapture, assert_no_content, capturing_job_audit, job_audit

THUMB = b"\xff\xd8synthetic-thumbnail"
BEST = b"\xff\xd8synthetic-best"
TRANSCRIPT = pd.DataFrame({"start": ["0:00:00"], "end": ["0:00:02"], "text": ["An invented line."]})


def _faces_result(*, entries: int = 1, frames: int = 8, tracks: int = 2) -> FacesResult:
    """Build a synthetic face result.

    Args:
        entries: How many entries it holds.
        frames: Frames it reports as judged.
        tracks: Tracks it reports as sent for grouping.

    Returns:
        FacesResult: The result.
    """
    built: list[FaceEntry] = []
    rows: list[DetectionRow] = []
    for number in range(1, entries + 1):
        crop = FaceCrop(time_sec=float(number), box=(10, 10, 80, 80), score=0.95, quality=0.8, jpeg=BEST)
        built.append(
            FaceEntry(
                number=number,
                assigned=True,
                appearances=((1.0, 3.0), (6.0, 8.0)),
                detections=3,
                best=crop,
                crops=(),
                thumbnail=THUMB,
            )
        )
        rows.extend(DetectionRow(time_sec=t, entry=number, box=(10, 10, 80, 80), score=0.95) for t in (1.0, 2.0, 7.0))
    return FacesResult(
        outcome="ok",
        grouped=True,
        truncated=False,
        entries=tuple(built),
        rows=tuple(rows),
        frames_sent=frames,
        tracks_sent=tracks,
    )


def _job(tmp_path: Path, *, faces_mode: FacesMode = "group", **options: Any) -> JobState:
    """Build a queued job over a synthetic upload.

    Args:
        tmp_path: Temporary directory.
        faces_mode: The operator switch the job runs under.
        **options: Job options.

    Returns:
        JobState: The job.
    """
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"video")
    return JobState(
        job_id="f" * 32,
        owner_id="owner-example",
        file_name="clip.mp4",
        file_path=media,
        source_file_hash="sha256:x",
        options=JobOptions.model_validate({"task": "transcribe", **options}),
        status=JobStatus.QUEUED,
        faces_mode=faces_mode,
        audit=job_audit(),
    )


@pytest.fixture
def transcribed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub transcription with a one-line transcript.

    Args:
        monkeypatch: Pytest patcher.
    """
    monkeypatch.setattr(
        "nextext.pipeline.transcription_pipeline",
        lambda **kwargs: TranscriptionOutcome(transcript=TRANSCRIPT.copy(), src_lang="en"),
    )


def _events(state: JobState) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
    """Run the pipeline and collect the events it pushes.

    Args:
        state: The job.

    Returns:
        tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]: The result and the events.
    """
    events: list[tuple[str, dict[str, Any]]] = []
    result = _run_pipeline_blocking(state, lambda name, payload: events.append((name, payload)))
    return result, events


def _face_completed(events: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """Find the face stage's completion event.

    Args:
        events: The pushed events.

    Returns:
        dict[str, Any]: Its payload.
    """
    (payload,) = [p for n, p in events if n == "stage_completed" and p["stage"] == "Detecting faces"]
    return payload


def test_faces_stage_runs_at_index_2_and_lands_in_result(
    transcribed: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The stage runs third, in the switch's mode, and its entries reach the snapshot as data URIs."""
    seen: dict[str, Any] = {}

    def faces(path: Path, *, mode: str) -> FacesResult:
        seen.update(path=path, mode=mode)
        return _faces_result()

    monkeypatch.setattr("nextext.pipeline.faces_pipeline", faces)
    state = _job(tmp_path, faces=True)

    result, events = _events(state)

    assert STAGE_FACES == 2
    assert PIPELINE_STAGE_LABELS[STAGE_FACES] == "Detecting faces"
    assert PIPELINE_STAGE_LABELS[STAGE_FACES - 1] == "Describing keyframes"
    started = [p for n, p in events if n == "stage_started" and p["stage"] == "Detecting faces"]
    assert [p["stage_index"] for p in started] == [2]
    assert seen == {"path": state.file_path, "mode": "group"}
    assert result["faces"] == _faces_result()

    state.result = result
    state.status = JobStatus.COMPLETED
    snapshot = state.snapshot()
    assert snapshot.result is not None and snapshot.result.faces is not None
    faces_out = snapshot.result.faces
    assert (faces_out.outcome, faces_out.grouped, faces_out.truncated) == ("ok", True, False)
    entry = faces_out.entries[0]
    assert entry.thumbnail == "data:image/jpeg;base64," + base64.b64encode(THUMB).decode("ascii")
    assert entry.best_artifact == "face_001.jpg"
    assert (entry.number, entry.assigned, entry.detections) == (1, True, 3)
    assert (entry.first_seen_seconds, entry.last_seen_seconds) == (1.0, 7.0)
    assert [(a.start_seconds, a.end_seconds) for a in entry.appearances] == [(1.0, 3.0), (6.0, 8.0)]


def test_face_stage_writes_its_ai_calls(transcribed: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Detection and grouping are recorded with frames and tracks, and nothing of what was found."""

    def faces(path: Path, *, mode: str) -> FacesResult:
        record_inference("face_detect", "faces", ok=True)
        record_inference("face_group", "faces", ok=True)
        return _faces_result(frames=8, tracks=2)

    monkeypatch.setattr("nextext.pipeline.faces_pipeline", faces)
    audit, capture = capturing_job_audit("f" * 32)
    audit.collect({"files": 1})
    state = _job(tmp_path, faces=True)
    state.audit = audit

    _run_pipeline_blocking(state, lambda *a, **k: None)

    detect = capture.one("ai_call", "ai.detect_faces")
    group = capture.one("ai_call", "ai.group_faces")
    assert detect["criteria"] == {"requests": 1, "frames": 8}
    assert group["criteria"] == {"requests": 1, "tracks": 2}
    for event in (detect, group):
        assert_no_content(event, "thumbnail", "entries", base64.b64encode(THUMB).decode("ascii"))
    capture.assert_chain_ok()


def test_silent_video_keeps_its_faces(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A clip without speech is skipped for the text stages but keeps its faces."""
    empty = pd.DataFrame({"start": [], "end": [], "text": []})
    monkeypatch.setattr(
        "nextext.pipeline.transcription_pipeline",
        lambda **kwargs: TranscriptionOutcome(transcript=empty, src_lang="en", skip_reason="vad_no_speech"),
    )
    monkeypatch.setattr("nextext.pipeline.faces_pipeline", lambda path, *, mode: _faces_result())

    result, _events_seen = _events(_job(tmp_path, faces=True))

    assert result["skipped"] is True
    assert result["faces"] == _faces_result()


def test_faces_failure_does_not_fail_the_job(
    transcribed: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, log_lines: list[str]
) -> None:
    """An unexpected error in the stage leaves the job running, without faces, logged by class."""
    detail = " ".join(("invented", "failure", "detail", "0427"))

    def faces(path: Path, *, mode: str) -> FacesResult:
        raise RuntimeError(detail)

    monkeypatch.setattr("nextext.pipeline.faces_pipeline", faces)

    result, events = _events(_job(tmp_path, faces=True))

    assert result["faces"] is None
    assert result["transcript"] is not None
    assert _face_completed(events)["result_delta"] == {"skipped": True}
    logged = "\n".join(log_lines)
    assert "RuntimeError" in logged
    assert "Traceback (most recent call last):" in logged
    assert ", in faces" in logged
    assert "invented failure detail" not in logged


def test_sse_delta_carries_counts_only(transcribed: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The stage's event carries counts, never a thumbnail or image bytes."""
    monkeypatch.setattr("nextext.pipeline.faces_pipeline", lambda path, *, mode: _faces_result(frames=8))

    _result, events = _events(_job(tmp_path, faces=True))

    assert _face_completed(events)["result_delta"] == {"face_frames": 8, "faces": 1}
    rendered = json.dumps(events, default=str)
    assert "thumbnail" not in rendered
    assert "data:image" not in rendered
    assert base64.b64encode(THUMB).decode("ascii") not in rendered


def test_audio_only_has_no_faces_tab(transcribed: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A file without frames gives no face result, and the stage reports itself skipped."""
    monkeypatch.setattr("nextext.pipeline.faces_pipeline", lambda path, *, mode: None)
    state = _job(tmp_path, faces=True)

    result, events = _events(state)

    assert result["faces"] is None
    assert _face_completed(events)["result_delta"] == {"skipped": True}
    state.result = result
    state.status = JobStatus.COMPLETED
    snapshot = state.snapshot()
    assert snapshot.result is not None and snapshot.result.faces is None


@pytest.mark.parametrize(("faces_mode", "requested"), [("off", True), ("group", False), ("detect", False)])
def test_stage_is_skipped_unless_asked_and_switched_on(
    transcribed: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, faces_mode: FacesMode, requested: bool
) -> None:
    """The stage runs only when the job asks for faces and the switch is on.

    Args:
        transcribed: Stubbed transcription.
        monkeypatch: Pytest patcher.
        tmp_path: Temporary directory.
        faces_mode: The operator switch.
        requested: The job's ``faces`` option.
    """

    def faces(path: Path, *, mode: str) -> FacesResult:
        raise AssertionError("the face stage must not run")

    monkeypatch.setattr("nextext.pipeline.faces_pipeline", faces)

    result, events = _events(_job(tmp_path, faces_mode=faces_mode, faces=requested))

    assert result["faces"] is None
    assert _face_completed(events)["result_delta"] == {"skipped": True}


# --- API -----------------------------------------------------------------------------


def _faces_runner(state: JobState, push: PushEvent) -> dict[str, Any]:
    """Stand-in pipeline returning a transcript and a face result.

    Args:
        state: The job.
        push: Event sink.

    Returns:
        dict[str, Any]: The result.
    """
    return {"transcript": TRANSCRIPT.copy(), "task": "transcribe", "faces": _faces_result()}


@pytest.fixture
def faces_app(monkeypatch: pytest.MonkeyPatch, audit_capture: AuditCapture) -> Any:
    """Build an app factory running the face-result stub under a given switch.

    Args:
        monkeypatch: Pytest patcher.
        audit_capture: Where the app's audit events go.

    Returns:
        Any: ``make(mode) -> TestClient`` (a context manager).
    """
    monkeypatch.setattr(main_module, "setup_logging", lambda: None)

    def make(mode: str | None) -> TestClient:
        if mode is None:
            monkeypatch.delenv("NEXTEXT_FACES", raising=False)
        else:
            monkeypatch.setenv("NEXTEXT_FACES", mode)
        client = TestClient(create_app(audit_emitter_factory=audit_capture.factory, pipeline_runner=_faces_runner))
        client.headers["X-Auth-User"] = "a" * 32
        return client

    return make


def _post(client: TestClient, **options: Any) -> Any:
    """Submit a job with a synthetic upload.

    Args:
        client: The client.
        **options: Job options.

    Returns:
        Any: The response.
    """
    files = {"file": ("clip.mp4", io.BytesIO(b"video-bytes"), "video/mp4")}
    return client.post("/api/v1/jobs", files=files, data={"options": json.dumps({"task": "transcribe", **options})})


def _wait_completed(client: TestClient, job_id: str) -> dict[str, Any]:
    """Poll a job until it completes.

    Args:
        client: The client.
        job_id: The job.

    Returns:
        dict[str, Any]: The completed snapshot.
    """
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        body = client.get(f"/api/v1/jobs/{job_id}").json()
        if body["status"] == "completed":
            return body
        time.sleep(0.02)
    raise AssertionError("job never completed")


def test_faces_requested_while_off_is_422_and_not_collected(faces_app: Any, audit_capture: AuditCapture) -> None:
    """Asking for faces while the switch is off is refused before anything is stored or recorded."""
    with faces_app(None) as client:
        manager = client.app.state.job_manager  # type: ignore[attr-defined]
        before = sorted(manager.tmp_root.iterdir())
        response = _post(client, faces=True)
        after = sorted(manager.tmp_root.iterdir())

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid request."}
    assert audit_capture.of("collect") == []
    assert after == before


def test_faces_requested_while_on_is_collected(faces_app: Any, audit_capture: AuditCapture) -> None:
    """With the switch on, a faces job is accepted and its collect event names the analysis."""
    with faces_app("detect") as client:
        response = _post(client, faces=True)
        assert response.status_code == 201
        _wait_completed(client, response.json()["job_id"])

    (collect,) = audit_capture.of("collect")
    assert "faces" in collect["criteria"]["analyses"]
    assert collect["criteria"]["options"]["faces"] is True


@pytest.mark.parametrize(("mode", "expected"), [(None, "off"), ("detect", "detect"), ("group", "group")])
def test_config_reports_faces_mode(faces_app: Any, mode: str | None, expected: str) -> None:
    """GET /config tells the SPA whether to offer faces, and in which mode.

    Args:
        faces_app: App factory.
        mode: ``NEXTEXT_FACES``.
        expected: What /config reports.
    """
    with faces_app(mode) as client:
        response = client.get("/api/v1/config")

    assert response.status_code == 200
    assert response.json()["faces"] == expected


def test_config_event_records_faces_mode(faces_app: Any, audit_capture: AuditCapture) -> None:
    """The startup config event records the operator switch."""
    with faces_app("group"):
        pass

    (config,) = audit_capture.of("admin", "config.audit")
    assert config["criteria"]["faces"] == "group"


def test_list_items_carry_no_thumbnails(faces_app: Any) -> None:
    """The job list never carries thumbnails; only the finished snapshot does."""
    with faces_app("group") as client:
        response = _post(client, faces=True)
        job_id = response.json()["job_id"]
        snapshot = _wait_completed(client, job_id)
        listing = client.get("/api/v1/jobs")

    assert snapshot["result"]["faces"]["entries"][0]["thumbnail"].startswith("data:image/jpeg;base64,")
    assert listing.status_code == 200
    assert "data:image" not in listing.text
    assert "thumbnail" not in listing.text


def test_faces_option_is_strict_and_defaults_off() -> None:
    """``faces`` defaults off and is a plain boolean option; unknown fields still fail."""
    assert JobOptions.model_validate({}).faces is False
    assert JobOptions.model_validate({"faces": True}).faces is True
    with pytest.raises(ValueError, match="extra"):
        JobOptions.model_validate({"faces_mode": "group"})


def test_manager_hands_its_switch_to_each_job(faces_app: Any) -> None:
    """The switch read at startup is what every job of the manager runs under."""
    with faces_app("detect") as client:
        manager = client.app.state.job_manager  # type: ignore[attr-defined]
        job_id = _post(client, faces=True).json()["job_id"]
        _wait_completed(client, job_id)
        assert manager.faces_mode == "detect"
        assert manager._jobs[job_id].faces_mode == "detect"


def _endpoint_warnings(log_lines: list[str]) -> list[str]:
    """Pick the startup warnings about a missing faces endpoint.

    Args:
        log_lines: Captured messages.

    Returns:
        list[str]: The matching messages.
    """
    return [line for line in log_lines if "NEXTEXT_FACES" in line and "FACES_API_BASE" in line]


def test_faces_on_without_an_endpoint_warns_at_startup(
    faces_app: Any, monkeypatch: pytest.MonkeyPatch, log_lines: list[str]
) -> None:
    """Switching faces on with no endpoint to send them to is named once at startup, without values."""
    monkeypatch.setenv("FACES_API_BASE", "off")
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    with faces_app("group"):
        pass

    (warning,) = _endpoint_warnings(log_lines)
    assert "group" not in warning
    assert "off" not in warning.split("(")[0]


@pytest.mark.parametrize(("mode", "base"), [("detect", "http://faces.invalid"), (None, "off")])
def test_no_endpoint_warning_when_it_resolves_or_faces_are_off(
    faces_app: Any, monkeypatch: pytest.MonkeyPatch, log_lines: list[str], mode: str | None, base: str
) -> None:
    """No warning when the endpoint resolves, or when faces are off anyway.

    Args:
        faces_app: App factory.
        monkeypatch: Pytest patcher.
        log_lines: Captured messages.
        mode: ``NEXTEXT_FACES``.
        base: ``FACES_API_BASE``.
    """
    monkeypatch.setenv("FACES_API_BASE", base)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    with faces_app(mode):
        pass

    assert _endpoint_warnings(log_lines) == []
