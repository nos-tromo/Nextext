"""Tests for the hate-speech stage wiring in ``_run_pipeline_blocking`` (nextext.api.jobs)."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nextext.api import jobs as jobs_module
from nextext.api.artifacts import render_artifact
from nextext.api.jobs import JobState, _run_pipeline_blocking, _serialize_result
from nextext.api.schemas import JobOptions, JobStatus
from nextext.core.keyframes import Keyframe
from nextext.core.visual_context import FrameCaption
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


_FRAME_FINDING: dict[str, Any] = {
    "hate_speech": True,
    "category": "extremism",
    "confidence": "high",
    "reason": "r",
    "text": "A flag bearing a hate symbol hangs on a wall.",
    "start": "0:00:03",
    "source": "frame",
}


class _HealthyInference:
    """Stand-in for ``InferencePipeline`` that is always healthy."""

    def get_health(self) -> bool:
        """Report the provider as reachable.

        Returns:
            bool: Always ``True``.
        """
        return True


def _captioned_video(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, transcript: pd.DataFrame) -> JobState:
    """Stub a video job whose one keyframe is captioned, with hate speech and keyframes on.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching module attributes.
        tmp_path (Path): Temporary directory fixture.
        transcript (pd.DataFrame): What transcription returns; empty for a silent clip.

    Returns:
        JobState: The queued job.
    """
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"video")
    skip = None if len(transcript) else "asr_empty_transcript"
    monkeypatch.delenv("NEXTEXT_VISUAL_SUMMARY", raising=False)
    monkeypatch.setattr(
        "nextext.pipeline.transcription_pipeline",
        lambda **kwargs: TranscriptionOutcome(transcript=transcript, src_lang="en", skip_reason=skip),
    )
    monkeypatch.setattr(jobs_module, "extract_keyframe_samples", lambda path, **kw: [Keyframe(3.0, b"\xff\xd8a")])
    monkeypatch.setattr("nextext.core.openai_cfg.InferencePipeline", _HealthyInference)
    monkeypatch.setattr(
        "nextext.core.visual_context.describe_keyframes",
        lambda samples, pipeline, **kw: [FrameCaption(time_sec=3.0, caption=_FRAME_FINDING["text"])],
    )
    return JobState(
        job_id="hs-frames",
        owner_id="o",
        file_name="clip.mp4",
        file_path=media,
        source_file_hash="sha256:x",
        options=JobOptions.model_validate({"task": "transcribe", "hate_speech": True, "keyframes": True}),
        status=JobStatus.QUEUED,
    )


def test_a_silent_clip_is_judged_from_its_captions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No speech is no reason to skip what the video shows; the stage is announced on this path too.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching module attributes.
        tmp_path (Path): Temporary directory fixture.
    """
    empty = pd.DataFrame({"start": [], "end": [], "speaker": [], "text": []})
    state = _captioned_video(monkeypatch, tmp_path, empty)
    judged: list[list[FrameCaption]] = []

    def _fake_frames(captions: list[FrameCaption], inference_pipeline: Any) -> list[dict[str, Any]]:
        judged.append(list(captions))
        return [dict(_FRAME_FINDING)]

    monkeypatch.setattr("nextext.pipeline.frame_hate_speech_pipeline", _fake_frames)
    events: list[tuple[str, dict[str, Any]]] = []

    result = _run_pipeline_blocking(state, lambda name, payload: events.append((name, payload)))

    assert result["skipped"] is True
    assert [[c.caption for c in captions] for captions in judged] == [[_FRAME_FINDING["text"]]]
    assert result["hate_speech_findings"] == [_FRAME_FINDING]
    assert [p["stage"] for name, p in events if name == "stage_started"][-1] == "Detecting hate speech"


def test_a_spoken_clip_reports_transcript_then_frame_findings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The captions are judged beside the transcript, and their findings follow its findings.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching module attributes.
        tmp_path (Path): Temporary directory fixture.
    """
    spoken = pd.DataFrame({"start": ["0:00:00"], "end": ["0:00:02"], "speaker": [""], "text": ["Hallo."]})
    state = _captioned_video(monkeypatch, tmp_path, spoken)
    row_finding = {**_DIARIZED_FINDING, "source": "transcript"}
    monkeypatch.setattr("nextext.pipeline.hate_speech_pipeline", lambda **kwargs: [dict(row_finding)])
    monkeypatch.setattr(
        "nextext.pipeline.frame_hate_speech_pipeline", lambda captions, inference_pipeline: [dict(_FRAME_FINDING)]
    )

    result = _run_pipeline_blocking(state, lambda *a, **k: None)

    assert result["hate_speech_findings"] == [row_finding, _FRAME_FINDING]


def test_serialized_findings_say_where_they_came_from() -> None:
    """A frame finding keeps its source; one stored without a source reads as a transcript finding."""
    serialized = _serialize_result({"hate_speech_findings": [dict(_FRAME_FINDING), dict(_DIARIZED_FINDING)]})

    assert serialized.hate_speech_findings is not None
    assert [finding.source for finding in serialized.hate_speech_findings] == ["frame", "transcript"]


def test_hate_speech_csv_appends_the_source_column() -> None:
    """The CSV download names each finding's source in its last column."""
    state = JobState(
        job_id="hs3",
        owner_id="o",
        file_name="clip.mp4",
        file_path=Path("clip.mp4"),
        source_file_hash="sha256:x",
        options=JobOptions.model_validate({}),
        status=JobStatus.COMPLETED,
        result={"hate_speech_findings": [{**_DIARIZED_FINDING, "source": "transcript"}, dict(_FRAME_FINDING)]},
    )

    rendered = render_artifact(state, "hate_speech.csv")

    assert rendered is not None
    lines = rendered[0].decode("utf-8").splitlines()
    assert lines[0] == "hate_speech,category,confidence,reason,text,start,speaker,translation,source"
    assert lines[2].endswith(",frame")
