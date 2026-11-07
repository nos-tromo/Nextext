"""Tests for the order of the pipeline's stage events (``_run_pipeline_blocking``).

The SPA renders progress from ``stage_index`` against ``PIPELINE_STAGE_LABELS``,
so every stage must announce itself at its own label's index, in label order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nextext.api import jobs as jobs_module
from nextext.api.jobs import (
    PIPELINE_STAGE_LABELS,
    STAGE_HATE_SPEECH,
    STAGE_KEYFRAMES,
    STAGE_SUMMARY,
    STAGE_TRANSCRIBE,
    STAGE_TRANSLATE,
    STAGE_WORDS,
    JobState,
    _run_pipeline_blocking,
)
from nextext.api.schemas import JobOptions, JobStatus
from nextext.pipeline import TranscriptionOutcome
from tests.audit_helpers import job_audit

TRANSCRIPT = pd.DataFrame(
    {
        "start": ["0:00:00", "0:00:03"],
        "end": ["0:00:03", "0:00:06"],
        "speaker": ["Speaker 1", "Speaker 2"],
        "text": ["An invented first line.", "An invented reply."],
    }
)


class _HealthyInference:
    """Stand-in for ``InferencePipeline`` that is always reachable."""

    def get_health(self) -> bool:
        """Report the provider as reachable.

        Returns:
            bool: Always ``True``.
        """
        return True


@pytest.fixture
def stubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace every model-backed stage with a deterministic fake.

    Args:
        monkeypatch: Pytest patcher.
    """

    def translation(df: pd.DataFrame, trg_lang: str, **kwargs: Any) -> pd.DataFrame:
        translated = df.copy()
        translated["translation"] = ["Eine erfundene erste Zeile.", "Eine erfundene Antwort."]
        return translated

    monkeypatch.setattr(
        "nextext.pipeline.transcription_pipeline",
        lambda **kwargs: TranscriptionOutcome(transcript=TRANSCRIPT.copy(), src_lang="en"),
    )
    monkeypatch.setattr("nextext.pipeline.translation_pipeline", translation)
    monkeypatch.setattr(
        "nextext.pipeline.wordlevel_pipeline",
        lambda df, language: (pd.DataFrame({"word": ["invented"], "count": [2]}), None, None),
    )
    monkeypatch.setattr("nextext.pipeline.summarization_pipeline", lambda text, **kwargs: "An invented summary.")
    monkeypatch.setattr("nextext.pipeline.hate_speech_pipeline", lambda **kwargs: [])
    monkeypatch.setattr("nextext.pipeline.frame_hate_speech_pipeline", lambda captions, **kwargs: [])
    monkeypatch.setattr("nextext.core.openai_cfg.InferencePipeline", _HealthyInference)
    monkeypatch.setattr(jobs_module, "extract_keyframe_samples", lambda path, **kwargs: [])


def _job(tmp_path: Path, **options: Any) -> JobState:
    """Build a queued job over a synthetic upload.

    Args:
        tmp_path: Temporary directory.
        **options: Job options.

    Returns:
        JobState: The job.
    """
    media = tmp_path / "upload.wav"
    media.write_bytes(b"audio")
    audit = job_audit()
    return JobState(
        job_id=audit.job_id,
        owner_id="owner-example",
        file_name="upload.wav",
        file_path=media,
        source_file_hash="sha256:x",
        options=JobOptions.model_validate({"task": "transcribe", **options}),
        status=JobStatus.QUEUED,
        audit=audit,
    )


def test_stage_started_sequence_matches_labels(stubbed: None, tmp_path: Path) -> None:
    """A job with every option on announces each stage once, in label order, at its own index."""
    events: list[tuple[str, dict[str, Any]]] = []
    state = _job(
        tmp_path,
        task="translate",
        trg_lang="de",
        words=True,
        summarization=True,
        hate_speech=True,
        keyframes=True,
    )

    _run_pipeline_blocking(state, lambda name, payload: events.append((name, payload)))

    started = [payload["stage"] for name, payload in events if name == "stage_started"]
    assert started == list(PIPELINE_STAGE_LABELS)
    assert [p["stage_index"] for n, p in events if n == "stage_started"] == list(range(len(PIPELINE_STAGE_LABELS)))
    completed = [payload["stage"] for name, payload in events if name == "stage_completed"]
    assert completed == list(PIPELINE_STAGE_LABELS)


def test_stage_constants_name_their_labels() -> None:
    """Each named stage index points at its own label."""
    assert PIPELINE_STAGE_LABELS[STAGE_TRANSCRIBE] == "Transcribing"
    assert PIPELINE_STAGE_LABELS[STAGE_KEYFRAMES] == "Describing keyframes"
    assert PIPELINE_STAGE_LABELS[STAGE_TRANSLATE] == "Translating"
    assert PIPELINE_STAGE_LABELS[STAGE_WORDS] == "Running word-level analysis"
    assert PIPELINE_STAGE_LABELS[STAGE_SUMMARY] == "Summarizing"
    assert PIPELINE_STAGE_LABELS[STAGE_HATE_SPEECH] == "Detecting hate speech"
