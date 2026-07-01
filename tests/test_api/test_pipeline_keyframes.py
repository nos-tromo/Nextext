"""Tests for keyframe wiring in ``_run_pipeline_blocking`` (nextext.api.jobs)."""

from pathlib import Path

import pandas as pd
import pytest

from nextext.api import jobs as jobs_module
from nextext.api.jobs import JobState, _run_pipeline_blocking
from nextext.api.schemas import JobOptions, JobStatus


def test_pipeline_populates_keyframes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The job result carries keyframes produced by ``extract_keyframes``."""
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"video")

    # Stub the heavy stages so only keyframe wiring is exercised.
    df = pd.DataFrame({"start": [0.0], "end": [1.0], "speaker": [""], "text": ["hi"]})
    monkeypatch.setattr("nextext.pipeline.transcription_pipeline", lambda **kwargs: (df, "en"))
    monkeypatch.setattr(jobs_module, "extract_keyframes", lambda path, **kw: [b"\xff\xd8\xff0", b"\xff\xd8\xff1"])

    state = JobState(
        job_id="j1",
        owner_id="o",
        file_name="clip.mp4",
        file_path=media,
        source_file_hash="sha256:x",
        options=JobOptions.model_validate({"task": "transcribe"}),
        status=JobStatus.QUEUED,
    )
    result = _run_pipeline_blocking(state, lambda *a, **k: None)
    assert result["keyframes"] == [b"\xff\xd8\xff0", b"\xff\xd8\xff1"]


def test_pipeline_empty_transcript_still_sets_keyframes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The empty-transcript early return still carries keyframes in the result.

    An empty transcript DataFrame drives ``_run_pipeline_blocking`` down its
    ``df.empty or not transcript_text`` short-circuit, which returns before any
    inference stage. That early payload must still include the keyframes, so a
    speechless-but-visual clip keeps its frames.
    """
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"video")

    # Empty transcript (has the ``text`` column but no rows) trips the
    # skip-branch; keyframe extraction is stubbed to a known payload.
    empty = pd.DataFrame({"start": [], "end": [], "speaker": [], "text": []})
    monkeypatch.setattr("nextext.pipeline.transcription_pipeline", lambda **kwargs: (empty, "en"))
    monkeypatch.setattr(jobs_module, "extract_keyframes", lambda path, **kw: [b"\xff\xd8\xff0", b"\xff\xd8\xff1"])

    state = JobState(
        job_id="j2",
        owner_id="o",
        file_name="clip.mp4",
        file_path=media,
        source_file_hash="sha256:x",
        options=JobOptions.model_validate({"task": "transcribe"}),
        status=JobStatus.QUEUED,
    )
    result = _run_pipeline_blocking(state, lambda *a, **k: None)
    assert result["skipped"] is True
    assert result["keyframes"] == [b"\xff\xd8\xff0", b"\xff\xd8\xff1"]
