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
