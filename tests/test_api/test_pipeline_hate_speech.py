"""Tests for the hate-speech stage wiring in ``_run_pipeline_blocking`` (nextext.api.jobs)."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nextext.api.jobs import JobState, _run_pipeline_blocking
from nextext.api.schemas import JobOptions, JobStatus
from nextext.pipeline import TranscriptionOutcome


def test_worker_passes_the_resolved_language_and_stores_findings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The stage hands the transcription's resolved language to the classifier and keeps its findings.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching module attributes.
        tmp_path (Path): Temporary directory fixture.
    """
    media = tmp_path / "talk.wav"
    media.write_bytes(b"audio")
    df = pd.DataFrame({"start": ["0:00:00"], "end": ["0:00:02"], "text": ["Hallo."]})
    monkeypatch.setattr(
        "nextext.pipeline.transcription_pipeline", lambda **kwargs: TranscriptionOutcome(transcript=df, src_lang="de")
    )

    class _StubInference:
        """Stand-in for ``InferencePipeline`` that is always healthy."""

        def get_health(self) -> bool:
            """Report the provider as reachable.

            Returns:
                bool: Always ``True``.
            """
            return True

    monkeypatch.setattr("nextext.core.openai_cfg.InferencePipeline", _StubInference)
    findings = [{"hate_speech": True, "category": "other", "confidence": "low", "reason": "r", "text": "Hallo."}]
    seen: dict[str, Any] = {}

    def _fake_hate_speech(**kwargs: Any) -> list[dict[str, Any]]:
        seen.update(kwargs)
        return findings

    monkeypatch.setattr("nextext.pipeline.hate_speech_pipeline", _fake_hate_speech)
    state = JobState(
        job_id="hs1",
        owner_id="o",
        file_name="talk.wav",
        file_path=media,
        source_file_hash="sha256:x",
        options=JobOptions.model_validate({"task": "transcribe", "hate_speech": True}),
        status=JobStatus.QUEUED,
    )

    result = _run_pipeline_blocking(state, lambda *a, **k: None)

    assert seen["src_lang"] == "de"
    assert result["hate_speech_findings"] == findings
