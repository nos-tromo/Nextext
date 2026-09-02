"""Tests for keyframes surfaced through artifacts (nextext.api.artifacts).

Covers both the standalone ``keyframes.zip`` artifact and the ``keyframes/``
subfolder that ``archive.zip`` nests frames under.
"""

import io
import json
import zipfile
from pathlib import Path

import pandas as pd

from nextext.api import artifacts
from nextext.api.jobs import JobState
from nextext.api.schemas import JobOptions, JobStatus
from nextext.core.visual_context import FrameCaption


def _job_with_keyframes(frames: list[bytes], times: list[float] | None = None) -> JobState:
    result: dict[str, object] = {"keyframes": frames}
    if times is not None:
        result["keyframe_times"] = times
    return JobState(
        job_id="j1",
        owner_id="o",
        file_name="clip.mp4",
        file_path=Path("clip.mp4"),
        source_file_hash="sha256:x",
        options=JobOptions.model_validate({}),
        status=JobStatus.COMPLETED,
        result=result,
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


def test_archive_zip_nests_keyframes_under_subfolder() -> None:
    """``archive.zip`` nests keyframes under a dedicated ``keyframes/`` subfolder.

    Unlike the standalone ``keyframes.zip`` artifact (flat ``frame_000.jpg``
    names), the combined archive groups frames under ``keyframes/`` so they
    never collide with the flat ``{stem}_transcript.csv``-style member names
    produced by the other archive members.
    """
    frames = [b"\xff\xd8\xff0", b"\xff\xd8\xff1"]
    rendered = artifacts.render_artifact(_job_with_keyframes(frames), "archive.zip")
    assert rendered is not None
    payload, content_type = rendered
    assert content_type == "application/zip"
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = set(zf.namelist())
        assert "clip/keyframes/frame_000.jpg" in names
        assert "clip/keyframes/frame_001.jpg" in names
        # Round-trip one member back to its exact source bytes.
        assert zf.read("clip/keyframes/frame_000.jpg") == frames[0]


def test_archive_zip_without_keyframes_has_no_keyframes_folder() -> None:
    """No keyframes in the result means no ``keyframes/`` entries in the archive.

    The job still carries a transcript, so the archive itself exists — a
    result with no members at all is covered by the test below.
    """
    state = _job_with_keyframes([])
    state.result["transcript"] = pd.DataFrame({"start": ["0:00:00"], "end": ["0:00:01"], "text": ["hi."]})
    rendered = artifacts.render_artifact(state, "archive.zip")
    assert rendered is not None
    payload, _content_type = rendered
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        assert not any("keyframes/" in name for name in zf.namelist())


def test_archive_zip_absent_when_job_produced_nothing() -> None:
    """A job with no outputs has no archive, matching every sibling artifact.

    A skipped (speech-free, frame-less) job used to be handed a 200 with an
    empty ZIP while its transcript/summary artifacts 404'd — an empty archive
    reads as a real but empty result.
    """
    assert artifacts.render_artifact(_job_with_keyframes([]), "archive.zip") is None


# ---------------------------------------------------------------------------
# manifest.json — the time each frame was sampled from
# ---------------------------------------------------------------------------


def test_keyframes_zip_carries_a_manifest_of_frame_times() -> None:
    """The zip names each frame's sampling time so a reader can place it in the clip."""
    frames = [b"\xff\xd8\xff0", b"\xff\xd8\xff1"]
    rendered = artifacts.render_artifact(_job_with_keyframes(frames, [0.0, 12.5]), "keyframes.zip")
    assert rendered is not None
    payload, _content_type = rendered
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    assert manifest["frames"] == [
        {"file": "frame_000.jpg", "index": 0, "time_sec": 0.0},
        {"file": "frame_001.jpg", "index": 1, "time_sec": 12.5},
    ]


def test_keyframes_zip_omits_the_manifest_without_times() -> None:
    """No times means no manifest, rather than one claiming times it does not have."""
    rendered = artifacts.render_artifact(_job_with_keyframes([b"\xff\xd8\xff0"]), "keyframes.zip")
    assert rendered is not None
    payload, _content_type = rendered
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        assert zf.namelist() == ["frame_000.jpg"]


def test_keyframes_zip_omits_the_manifest_on_a_length_mismatch() -> None:
    """A times list that does not pair with the frames is dropped, never guessed."""
    frames = [b"\xff\xd8\xff0", b"\xff\xd8\xff1"]
    rendered = artifacts.render_artifact(_job_with_keyframes(frames, [3.0]), "keyframes.zip")
    assert rendered is not None
    payload, _content_type = rendered
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        assert "manifest.json" not in zf.namelist()


def test_archive_zip_nests_the_manifest_with_the_frames() -> None:
    """The combined archive carries the manifest inside its ``keyframes/`` folder."""
    frames = [b"\xff\xd8\xff0"]
    rendered = artifacts.render_artifact(_job_with_keyframes(frames, [7.25]), "archive.zip")
    assert rendered is not None
    payload, _content_type = rendered
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        manifest = json.loads(zf.read("clip/keyframes/manifest.json"))
    assert manifest["frames"] == [{"file": "frame_000.jpg", "index": 0, "time_sec": 7.25}]


# ---------------------------------------------------------------------------
# visual_context.txt — the captions behind a video summary
# ---------------------------------------------------------------------------


def _job_with_captions(captions: list[FrameCaption]) -> JobState:
    """Build a completed job carrying frame captions.

    Args:
        captions (list[FrameCaption]): Captions to place on the job result.

    Returns:
        JobState: A completed job state ready for artifact rendering.
    """
    return JobState(
        job_id="j9",
        owner_id="o",
        file_name="clip.mp4",
        file_path=Path("clip.mp4"),
        source_file_hash="sha256:x",
        options=JobOptions.model_validate({}),
        status=JobStatus.COMPLETED,
        result={"frame_captions": captions, "summary": "a summary"},
    )


def test_visual_context_txt_renders_the_timestamped_block() -> None:
    """The artifact is the same block the summarizer was given."""
    rendered = artifacts.render_artifact(
        _job_with_captions(
            [FrameCaption(time_sec=0.0, caption="a hallway"), FrameCaption(time_sec=61.0, caption="a sign")]
        ),
        "visual_context.txt",
    )
    assert rendered is not None
    payload, media_type = rendered
    assert payload.decode("utf-8") == "[00:00] a hallway\n[01:01] a sign"
    assert media_type.startswith("text/plain")


def test_visual_context_txt_absent_without_captions() -> None:
    """An audio-only job 404s rather than serving an empty file."""
    assert artifacts.render_artifact(_job_with_captions([]), "visual_context.txt") is None


def test_visual_context_txt_is_a_supported_artifact() -> None:
    """The route's allowlist must know the name or it 404s before rendering."""
    assert "visual_context.txt" in artifacts.SUPPORTED_ARTIFACTS


def test_archive_includes_the_visual_context_file() -> None:
    """The combined archive carries the captions next to the summary."""
    state = _job_with_captions([FrameCaption(time_sec=5.0, caption="a whiteboard")])
    members = artifacts._render_archive_members(state)
    assert members["clip_visual_context.txt"].decode("utf-8") == "[00:05] a whiteboard"


def test_archive_omits_visual_context_when_absent() -> None:
    """No captions means no member, not an empty one."""
    members = artifacts._render_archive_members(_job_with_captions([]))
    assert "clip_visual_context.txt" not in members
