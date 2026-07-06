"""Tests for keyframe wiring in ``_run_pipeline_blocking`` (nextext.api.jobs).

Also covers surfacing the stashed URL as ``JobResult.keyframes_url`` via
``_serialize_result``.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nextext.api import jobs as jobs_module
from nextext.api.jobs import JobState, _run_pipeline_blocking, _serialize_result
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
    assert result["_keyframes_url"] == f"/api/v1/jobs/{state.job_id}/artifacts/keyframes.zip"


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
    assert result["_keyframes_url"] == f"/api/v1/jobs/{state.job_id}/artifacts/keyframes.zip"


def test_serialize_result_surfaces_keyframes_url() -> None:
    """A non-empty ``keyframes`` list surfaces the pre-baked artifact URL.

    Mirrors how ``wordcloud_url`` is only forwarded when a real wordcloud
    ``Figure`` is present: the URL is pre-baked by the pipeline (which has the
    job id in scope) and ``_serialize_result`` merely forwards it when the
    corresponding output actually exists.
    """
    result: dict[str, Any] = {
        "keyframes": [b"\xff\xd8\xff0", b"\xff\xd8\xff1"],
        "_keyframes_url": "/api/v1/jobs/j1/artifacts/keyframes.zip",
    }
    serialized = _serialize_result(result)
    assert serialized.keyframes_url == "/api/v1/jobs/j1/artifacts/keyframes.zip"


def test_serialize_result_omits_keyframes_url_when_keyframes_empty() -> None:
    """An empty ``keyframes`` list keeps ``keyframes_url`` unset, even if stashed.

    The guard checks the actual ``keyframes`` output, not merely whether
    ``_keyframes_url`` happens to be present, so a job that produced no frames
    never advertises a URL that would 404.
    """
    result: dict[str, Any] = {
        "keyframes": [],
        "_keyframes_url": "/api/v1/jobs/j3/artifacts/keyframes.zip",
    }
    serialized = _serialize_result(result)
    assert serialized.keyframes_url is None


def test_serialize_result_omits_keyframes_url_when_keyframes_absent() -> None:
    """A result dict with no ``keyframes`` key at all yields ``keyframes_url is None``."""
    serialized = _serialize_result({})
    assert serialized.keyframes_url is None
