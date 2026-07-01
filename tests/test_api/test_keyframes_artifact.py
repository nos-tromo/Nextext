"""Tests for the ``keyframes.zip`` artifact (nextext.api.artifacts)."""

import io
import zipfile
from pathlib import Path

from nextext.api import artifacts
from nextext.api.jobs import JobState
from nextext.api.schemas import JobOptions, JobStatus


def _job_with_keyframes(frames: list[bytes]) -> JobState:
    return JobState(
        job_id="j1",
        owner_id="o",
        file_name="clip.mp4",
        file_path=Path("clip.mp4"),
        source_file_hash="sha256:x",
        options=JobOptions.model_validate({}),
        status=JobStatus.COMPLETED,
        result={"keyframes": frames},
    )


def test_keyframes_zip_contains_each_frame() -> None:
    """Each keyframe payload becomes its own ``.jpg`` member in the zip."""
    rendered = artifacts.render_artifact(_job_with_keyframes([b"\xff\xd8\xff0", b"\xff\xd8\xff1"]), "keyframes.zip")
    assert rendered is not None
    payload, content_type = rendered
    assert content_type == "application/zip"
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = sorted(zf.namelist())
    assert len(names) == 2
    assert names[0].endswith(".jpg")


def test_keyframes_zip_absent_returns_none() -> None:
    """No keyframes in the result means no artifact to render."""
    assert artifacts.render_artifact(_job_with_keyframes([]), "keyframes.zip") is None


def test_keyframes_zip_is_supported() -> None:
    """The artifact name is advertised in the supported-artifacts set."""
    assert "keyframes.zip" in artifacts.SUPPORTED_ARTIFACTS
