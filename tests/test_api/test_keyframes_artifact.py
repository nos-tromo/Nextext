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
    """Each keyframe payload becomes its own ``.jpg`` member, byte-for-byte.

    Beyond counting members, this asserts every member name ends in ``.jpg``
    and round-trips one member back to its exact input bytes — guarding against
    payload corruption or a frame/name swap in the zip builder.
    """
    frames = [b"\xff\xd8\xff0", b"\xff\xd8\xff1"]
    rendered = artifacts.render_artifact(_job_with_keyframes(frames), "keyframes.zip")
    assert rendered is not None
    payload, content_type = rendered
    assert content_type == "application/zip"
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = sorted(zf.namelist())
        assert len(names) == 2
        assert all(name.endswith(".jpg") for name in names)
        # Sorted, ``frame_000.jpg`` is the first enumerated frame; round-trip
        # it back to its exact source bytes.
        assert zf.read(names[0]) == frames[0]


def test_keyframes_zip_absent_returns_none() -> None:
    """No keyframes in the result means no artifact to render."""
    assert artifacts.render_artifact(_job_with_keyframes([]), "keyframes.zip") is None


def test_keyframes_zip_is_supported() -> None:
    """The artifact name is advertised in the supported-artifacts set."""
    assert "keyframes.zip" in artifacts.SUPPORTED_ARTIFACTS
