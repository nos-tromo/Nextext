"""Tests for the face artifacts: ``faces.zip``, ``faces.csv``, ``face_NNN.jpg`` and the archives.

Face results are synthetic: placeholder JPEG bytes and invented boxes, never a
real face.
"""

from __future__ import annotations

import csv
import io
import json
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from nextext.api import artifacts
from nextext.api import main as main_module
from nextext.api.jobs import JobState, PushEvent
from nextext.api.main import create_app
from nextext.api.schemas import JobOptions, JobStatus
from nextext.core.face_extraction import DetectionRow, FaceCrop, FaceEntry, FacesResult
from tests.audit_helpers import AuditCapture, job_audit

TRANSCRIPT = pd.DataFrame({"start": ["0:00:00"], "end": ["0:00:02"], "text": ["An invented line."]})


def _crop(time_sec: float, tag: bytes, quality: float) -> FaceCrop:
    """Build a synthetic crop.

    Args:
        time_sec: When it was taken.
        tag: Distinguishing bytes appended to a JPEG marker.
        quality: Its quality.

    Returns:
        FaceCrop: The crop.
    """
    return FaceCrop(time_sec=time_sec, box=(12, 34, 80, 90), score=0.9512345, quality=quality, jpeg=b"\xff\xd8" + tag)


def _faces() -> FacesResult:
    """A grouped person with two extra crops and an ungrouped face.

    Returns:
        FacesResult: The result.
    """
    person = FaceEntry(
        number=1,
        assigned=True,
        appearances=((0.0, 2.0), (6.0, 8.0)),
        detections=3,
        best=_crop(1.0, b"best-1", 0.9),
        crops=(_crop(0.0, b"crop-1a", 0.8), _crop(7.0, b"crop-1b", 0.7)),
        thumbnail=b"\xff\xd8thumb-1",
    )
    single = FaceEntry(
        number=2,
        assigned=False,
        appearances=((3.0, 4.0),),
        detections=1,
        best=_crop(3.0, b"best-2", 0.6),
        crops=(),
        thumbnail=b"\xff\xd8thumb-2",
    )
    rows = (
        DetectionRow(time_sec=0.0, entry=1, box=(12, 34, 80, 90), score=0.95),
        DetectionRow(time_sec=1.0, entry=1, box=(14, 34, 80, 90), score=0.96),
        DetectionRow(time_sec=3.0, entry=2, box=(300, 40, 60, 60), score=0.88),
        DetectionRow(time_sec=7.0, entry=1, box=(16, 34, 80, 90), score=0.91),
    )
    return FacesResult(
        outcome="partial",
        grouped=True,
        truncated=False,
        entries=(person, single),
        rows=rows,
        frames_sent=8,
        tracks_sent=3,
    )


def _state(faces: FacesResult | None, *, transcript: pd.DataFrame | None = None, name: str = "clip.mp4") -> JobState:
    """Build a completed job carrying a face result.

    Args:
        faces: The face result.
        transcript: The transcript, if any.
        name: The upload's file name.

    Returns:
        JobState: The job.
    """
    return JobState(
        job_id="j1",
        owner_id="o",
        file_name=name,
        file_path=Path(name),
        source_file_hash="sha256:x",
        options=JobOptions.model_validate({"faces": True}),
        status=JobStatus.COMPLETED,
        result={"transcript": TRANSCRIPT.copy() if transcript is None else transcript, "faces": faces},
        audit=job_audit(),
    )


def _zip(payload: bytes) -> dict[str, bytes]:
    """Read every member of a ZIP.

    Args:
        payload: The ZIP bytes.

    Returns:
        dict[str, bytes]: Name to bytes.
    """
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def test_faces_zip_layout_and_manifest() -> None:
    """faces.zip holds a folder per entry with its best shot and crops, plus a manifest."""
    rendered = artifacts.render_artifact(_state(_faces()), "faces.zip")
    assert rendered is not None
    payload, content_type = rendered
    assert content_type == "application/zip"
    members = _zip(payload)
    assert sorted(members) == [
        "face_002/best.jpg",
        "manifest.json",
        "person_001/best.jpg",
        "person_001/crop_1.jpg",
        "person_001/crop_2.jpg",
    ]
    assert members["person_001/best.jpg"] == b"\xff\xd8best-1"
    assert members["person_001/crop_2.jpg"] == b"\xff\xd8crop-1b"
    manifest = json.loads(members["manifest.json"])
    assert (manifest["outcome"], manifest["grouped"], manifest["truncated"]) == ("partial", True, False)
    first = manifest["entries"][0]
    assert {key: first[key] for key in ("number", "assigned", "folder", "appearances")} == {
        "number": 1,
        "assigned": True,
        "folder": "person_001",
        "appearances": [[0.0, 2.0], [6.0, 8.0]],
    }
    assert first["files"][0] == {
        "file": "person_001/best.jpg",
        "time_sec": 1.0,
        "box": [12, 34, 80, 90],
        "score": 0.9512,
        "quality": 0.9,
    }
    assert [f["file"] for f in first["files"]] == [
        "person_001/best.jpg",
        "person_001/crop_1.jpg",
        "person_001/crop_2.jpg",
    ]
    assert manifest["entries"][1]["folder"] == "face_002"
    assert b"clip" not in members["manifest.json"]


def test_faces_csv_one_row_per_detection() -> None:
    """faces.csv lists every detection, without images."""
    rendered = artifacts.render_artifact(_state(_faces()), "faces.csv")
    assert rendered is not None
    payload, content_type = rendered
    assert content_type.startswith("text/csv")
    rows = list(csv.reader(io.StringIO(payload.decode("utf-8"))))
    assert rows[0] == ["time_sec", "entry", "x", "y", "w", "h", "score"]
    assert rows[1:] == [
        ["0.0", "1", "12", "34", "80", "90", "0.95"],
        ["1.0", "1", "14", "34", "80", "90", "0.96"],
        ["3.0", "2", "300", "40", "60", "60", "0.88"],
        ["7.0", "1", "16", "34", "80", "90", "0.91"],
    ]


def test_face_best_image_by_number_renders() -> None:
    """face_NNN.jpg is that entry's best shot; an unknown number is not produced."""
    state = _state(_faces())

    assert artifacts.render_artifact(state, "face_001.jpg") == (b"\xff\xd8best-1", "image/jpeg")
    assert artifacts.render_artifact(state, "face_002.jpg") == (b"\xff\xd8best-2", "image/jpeg")
    assert artifacts.render_artifact(state, "face_999.jpg") is None
    assert artifacts.FACE_IMAGE_ARTIFACT.fullmatch("face_001.jpg")
    assert not artifacts.FACE_IMAGE_ARTIFACT.fullmatch("face_1.jpg")
    assert {"faces.zip", "faces.csv"} <= artifacts.SUPPORTED_ARTIFACTS


def test_archive_holds_faces_folder_and_csv() -> None:
    """archive.zip carries the faces under faces/ beside the stem-named face table."""
    rendered = artifacts.render_artifact(_state(_faces()), "archive.zip")
    assert rendered is not None
    members = _zip(rendered[0])
    assert "clip/faces/person_001/best.jpg" in members
    assert "clip/faces/person_001/crop_1.jpg" in members
    assert "clip/faces/face_002/best.jpg" in members
    assert "clip/faces/manifest.json" in members
    assert members["clip/clip_faces.csv"].startswith(b"time_sec,entry,x,y,w,h,score")


def test_archive_of_a_silent_video_holds_its_faces() -> None:
    """A clip without a transcript still has an archive when it has faces."""
    state = _state(_faces(), transcript=pd.DataFrame({"start": [], "end": [], "text": []}))

    rendered = artifacts.render_artifact(state, "archive.zip")

    assert rendered is not None
    assert "clip/faces/manifest.json" in _zip(rendered[0])


def test_batch_archive_holds_faces() -> None:
    """The batch archive nests each job's faces in its own folder."""
    payload = artifacts.build_batch_archive([_state(_faces(), name="first.mp4"), _state(None, name="second.mp4")])

    members = _zip(payload)
    assert "first/faces/person_001/best.jpg" in members
    assert "first/first_faces.csv" in members
    assert not any(name.startswith("second/faces/") for name in members)


def test_docint_jsonl_has_no_faces() -> None:
    """The docint export never carries faces."""
    rendered = artifacts.render_artifact(_state(_faces()), "docint.jsonl")
    assert rendered is not None
    text = rendered[0].decode("utf-8")
    assert "face" not in text.lower()
    assert "best-1" not in text


@pytest.mark.parametrize("name", ["faces.zip", "faces.csv", "face_001.jpg"])
def test_no_faces_result_404s_face_artifacts(name: str) -> None:
    """Without a face result, or with no entries, no face artifact is produced.

    Args:
        name: The artifact.
    """
    empty = FacesResult(
        outcome="unavailable", grouped=False, truncated=False, entries=(), rows=(), frames_sent=4, tracks_sent=0
    )
    assert artifacts.render_artifact(_state(None), name) is None
    assert artifacts.render_artifact(_state(empty), name) is None
    rendered = artifacts.render_artifact(_state(empty), "archive.zip")
    assert rendered is not None
    assert not any("faces" in member for member in _zip(rendered[0]))


# --- HTTP ----------------------------------------------------------------------------


def _runner(state: JobState, push: PushEvent) -> dict[str, Any]:
    """Stand-in pipeline returning a transcript and a face result.

    Args:
        state: The job.
        push: Event sink.

    Returns:
        dict[str, Any]: The result.
    """
    return {"transcript": TRANSCRIPT.copy(), "task": "transcribe", "faces": _faces()}


@pytest.fixture
def faces_client(monkeypatch: pytest.MonkeyPatch, audit_capture: AuditCapture) -> Any:
    """A client over an app with faces switched on and one completed faces job.

    Args:
        monkeypatch: Pytest patcher.
        audit_capture: Where the app's audit events go.

    Yields:
        tuple[TestClient, str]: The client and the job id.
    """
    monkeypatch.setattr(main_module, "setup_logging", lambda: None)
    monkeypatch.setenv("NEXTEXT_FACES", "group")
    with TestClient(create_app(audit_emitter_factory=audit_capture.factory, pipeline_runner=_runner)) as client:
        client.headers["X-Auth-User"] = "a" * 32
        files = {"file": ("clip.mp4", io.BytesIO(b"video-bytes"), "video/mp4")}
        response = client.post("/api/v1/jobs", files=files, data={"options": json.dumps({"faces": True})})
        job_id = response.json()["job_id"]
        deadline = time.monotonic() + 5.0
        while client.get(f"/api/v1/jobs/{job_id}").json()["status"] != "completed":
            assert time.monotonic() < deadline, "job never completed"
            time.sleep(0.02)
        yield client, job_id


def test_face_best_image_by_number(faces_client: tuple[TestClient, str]) -> None:
    """GET face_001.jpg serves the best shot; face_999.jpg is not produced; face_1.jpg is not an artifact."""
    client, job_id = faces_client

    best = client.get(f"/api/v1/jobs/{job_id}/artifacts/face_001.jpg")
    missing = client.get(f"/api/v1/jobs/{job_id}/artifacts/face_999.jpg")
    unsupported = client.get(f"/api/v1/jobs/{job_id}/artifacts/face_1.jpg")

    assert best.status_code == 200
    assert best.content == b"\xff\xd8best-1"
    assert best.headers["content-type"] == "image/jpeg"
    assert best.headers["content-disposition"].startswith('attachment; filename="clip_face_001.jpg"')
    assert missing.status_code == 404
    assert "was not produced" in missing.json()["detail"]
    assert unsupported.status_code == 404
    assert "Unsupported artifact" in unsupported.json()["detail"]


def test_face_download_is_disclosed_as_jpg(faces_client: tuple[TestClient, str], audit_capture: AuditCapture) -> None:
    """Downloading a best shot is its own disclose event to download:jpg."""
    client, job_id = faces_client

    assert client.get(f"/api/v1/jobs/{job_id}/artifacts/face_001.jpg").status_code == 200
    assert client.get(f"/api/v1/jobs/{job_id}/artifacts/faces.zip").status_code == 200

    downloads = audit_capture.of("disclose", "artifact.download")
    assert [(e["recipient"], e["criteria"]) for e in downloads] == [
        ("download:jpg", {"artifact": "face_001.jpg"}),
        ("download:zip", {"artifact": "faces.zip"}),
    ]
    assert all(e["target"]["ids"] == [job_id] for e in downloads)
    audit_capture.assert_chain_ok()
