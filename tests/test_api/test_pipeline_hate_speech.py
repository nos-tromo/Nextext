"""Tests for the hate-speech stage wiring in ``_run_pipeline_blocking`` (nextext.api.jobs)."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nextext.api.artifacts import render_artifact
from nextext.api.jobs import JobState, _run_pipeline_blocking, _serialize_result
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


_DIARIZED_FINDING: dict[str, Any] = {
    "hate_speech": True,
    "category": "ethnicity",
    "confidence": "high",
    "reason": "r",
    "text": "Genau, raus mit denen.",
    "start": "0:00:04",
    "speaker": "Speaker 2",
    "translation": "Exactly, get them out.",
}


def test_serialized_findings_keep_speaker_and_translation() -> None:
    """The API result exposes who said it and the translation aid, not just the text."""
    serialized = _serialize_result({"hate_speech_findings": [dict(_DIARIZED_FINDING)]})

    assert serialized.hate_speech_findings is not None
    finding = serialized.hate_speech_findings[0]
    assert finding.speaker == "Speaker 2"
    assert finding.translation == "Exactly, get them out."


def test_serialized_findings_default_speaker_and_translation_to_none() -> None:
    """Findings from an undiarized, untranslated job still validate."""
    plain = {k: v for k, v in _DIARIZED_FINDING.items() if k not in {"speaker", "translation"}}

    serialized = _serialize_result({"hate_speech_findings": [plain]})

    assert serialized.hate_speech_findings is not None
    assert serialized.hate_speech_findings[0].speaker is None
    assert serialized.hate_speech_findings[0].translation is None


def test_hate_speech_csv_carries_speaker_and_translation_columns() -> None:
    """The CSV download keeps the existing column order and appends speaker and translation."""
    state = JobState(
        job_id="hs2",
        owner_id="o",
        file_name="talk.wav",
        file_path=Path("talk.wav"),
        source_file_hash="sha256:x",
        options=JobOptions.model_validate({}),
        status=JobStatus.COMPLETED,
        result={"hate_speech_findings": [dict(_DIARIZED_FINDING)]},
    )

    rendered = render_artifact(state, "hate_speech.csv")

    assert rendered is not None
    header = rendered[0].decode("utf-8").splitlines()[0]
    assert header == "hate_speech,category,confidence,reason,text,start,speaker,translation"
