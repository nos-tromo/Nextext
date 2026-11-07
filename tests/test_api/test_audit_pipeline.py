"""Tests for the ``ai_call`` events a job's pipeline writes (``_run_pipeline_blocking``).

Every stage that sends job content to a model writes one event per inference
task when it ends — model alias, request and token counts, the output's hash
where there is a text output, and the events whose output it was given as
sources. Hate-speech detection is recorded by what was judged, never by what
was found.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nextext.api import jobs as jobs_module
from nextext.api.jobs import JobState, _joined_analyses, _run_pipeline_blocking
from nextext.api.schemas import JobOptions, JobStatus
from nextext.audit import JobAudit
from nextext.core.inference_trace import record_inference
from nextext.core.keyframes import Keyframe
from nextext.core.visual_context import FrameCaption
from nextext.pipeline import TranscriptionOutcome
from tests.audit_helpers import AuditCapture, assert_no_content, capturing_job_audit

TRANSCRIPT = pd.DataFrame(
    {
        "start": ["0:00:00", "0:00:03"],
        "end": ["0:00:03", "0:00:06"],
        "speaker": ["Speaker 1", "Speaker 2"],
        "text": ["Alice Example spoke first.", "Bob Example answered."],
    }
)
TRANSLATIONS = ["Alice Example sprach zuerst.", "Bob Example antwortete."]
SUMMARY = "Two invented speakers talk."
FINDING = {
    "hate_speech": True,
    "category": "ethnicity",
    "confidence": "high",
    "reason": "invented reason",
    "text": "Bob Example answered.",
    "start": "0:00:03",
    "speaker": "Speaker 2",
    "source": "transcript",
}


def _sha(text: str) -> str:
    """Hash a text as ``ai.output_sha256`` does.

    Args:
        text: The text.

    Returns:
        str: Its hex SHA-256.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _HealthyInference:
    """Stand-in for ``InferencePipeline`` that is always reachable."""

    def get_health(self) -> bool:
        """Report the provider as reachable.

        Returns:
            bool: Always ``True``.
        """
        return True


@pytest.fixture
def stubbed(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace every model-backed stage with a fake that reports its requests.

    Args:
        monkeypatch: Pytest patcher.

    Returns:
        dict[str, Any]: What the fakes were called with.
    """
    seen: dict[str, Any] = {}

    def transcription(**kwargs: Any) -> TranscriptionOutcome:
        seen["transcription"] = kwargs
        record_inference("vad", "vad", ok=True)
        record_inference("asr", "whisper-test", ok=True)
        record_inference("diarize", "diarize", ok=True)
        record_inference("chat", "chat-model", ok=True, input_tokens=30, output_tokens=6)
        return TranscriptionOutcome(transcript=TRANSCRIPT.copy(), src_lang="en")

    def translation(df: pd.DataFrame, trg_lang: str, **kwargs: Any) -> pd.DataFrame:
        for _ in range(len(df)):
            record_inference("chat", "chat-model", ok=True, input_tokens=20, output_tokens=8)
        translated = df.copy()
        translated["translation"] = TRANSLATIONS
        return translated

    def wordlevel(df: pd.DataFrame, language: str) -> tuple[pd.DataFrame, pd.DataFrame, None]:
        record_inference("ner", "gliner", ok=True)
        entities = pd.DataFrame({"Category": ["PERSON"], "Entity": ["Alice Example"], "Frequency": [1]})
        return pd.DataFrame({"word": ["example"], "count": [2]}), entities, None

    def summarization(text: str, **kwargs: Any) -> str:
        record_inference("chat", "chat-model", ok=True, input_tokens=50, output_tokens=10)
        return SUMMARY

    def hate_speech(**kwargs: Any) -> list[dict[str, Any]]:
        record_inference("chat", "chat-model", ok=True, input_tokens=90, output_tokens=12)
        return [dict(FINDING)]

    monkeypatch.setattr("nextext.pipeline.transcription_pipeline", transcription)
    monkeypatch.setattr("nextext.pipeline.translation_pipeline", translation)
    monkeypatch.setattr("nextext.pipeline.wordlevel_pipeline", wordlevel)
    monkeypatch.setattr("nextext.pipeline.summarization_pipeline", summarization)
    monkeypatch.setattr("nextext.pipeline.hate_speech_pipeline", hate_speech)
    monkeypatch.setattr("nextext.pipeline.frame_hate_speech_pipeline", lambda captions, **kw: [])
    monkeypatch.setattr("nextext.core.openai_cfg.InferencePipeline", _HealthyInference)
    return seen


def _job(tmp_path: Path, audit: JobAudit, **options: Any) -> JobState:
    """Build a queued job over a synthetic upload in its own directory.

    Args:
        tmp_path: Temporary directory.
        audit: The job's audit trail.
        **options: Job options.

    Returns:
        JobState: The job.
    """
    job_dir = tmp_path / "job-1"
    job_dir.mkdir()
    media = job_dir / "upload.wav"
    media.write_bytes(b"audio")
    return JobState(
        job_id=audit.job_id,
        owner_id="alice.example",
        file_name="upload.wav",
        file_path=media,
        source_file_hash="sha256:x",
        options=JobOptions.model_validate({"task": "transcribe", **options}),
        status=JobStatus.QUEUED,
        audit=audit,
    )


def _ai_calls(capture: AuditCapture) -> dict[str, dict[str, Any]]:
    """Index the captured ``ai_call`` events by operation, checking each appears once.

    Args:
        capture: The capture.

    Returns:
        dict[str, dict[str, Any]]: Operation -> event.
    """
    events = capture.of("ai_call")
    operations = [event["operation"] for event in events]
    assert len(operations) == len(set(operations)), operations
    return {event["operation"]: event for event in events}


def test_every_stage_writes_one_ai_call_per_task(stubbed: dict[str, Any], tmp_path: Path) -> None:
    """A full job records each transfer once, in order, with recipient, model, counts and hashes."""
    audit, capture = capturing_job_audit("a" * 32, "alice.example")
    collect = audit.collect({"files": 1})
    state = _job(tmp_path, audit, trg_lang="de", task="translate", words=True, summarization=True, hate_speech=True)

    _run_pipeline_blocking(state, lambda *a, **k: None)

    calls = _ai_calls(capture)
    assert list(calls) == [
        "ai.vad",
        "ai.transcribe",
        "ai.diarize",
        "ai.sentence_restore",
        "ai.translate",
        "ai.ner",
        "ai.summarize",
        "ai.hate_speech",
    ]
    recipients = {op: event["recipient"] for op, event in calls.items()}
    assert recipients == {
        "ai.vad": "inference:vad",
        "ai.transcribe": "inference:asr",
        "ai.diarize": "inference:diarize",
        "ai.sentence_restore": "inference:chat",
        "ai.translate": "inference:translate",
        "ai.ner": "inference:ner",
        "ai.summarize": "inference:chat",
        "ai.hate_speech": "inference:chat",
    }
    for event in calls.values():
        assert event["outcome"] == "ok"
        assert event["actor"] == {"user": "alice.example", "via": "edge"}
        assert event["target"] == {"kind": "job", "ids": ["a" * 32], "count": 1}
        assert event["parent_event_id"] == collect["event_id"]
        assert_no_content(event, "Alice Example spoke", "invented reason", "ethnicity", SUMMARY)

    transcribe = calls["ai.transcribe"]
    assert transcribe["ai"]["model_alias"] == "whisper-test"
    assert transcribe["ai"]["output_sha256"] == _sha("Alice Example spoke first.\nBob Example answered.")
    assert transcribe["ai"]["sources"] == [collect["event_id"]]
    assert transcribe["criteria"] == {"requests": 1, "segments": 2}

    restore = calls["ai.sentence_restore"]
    assert restore["ai"]["sources"] == [transcribe["event_id"], calls["ai.diarize"]["event_id"]]
    assert (restore["ai"]["input_tokens"], restore["ai"]["output_tokens"]) == (30, 6)

    translate = calls["ai.translate"]
    assert translate["ai"]["output_sha256"] == _sha("\n".join(TRANSLATIONS))
    assert translate["criteria"] == {"requests": 2, "src_lang": "en", "trg_lang": "de"}
    assert (translate["ai"]["input_tokens"], translate["ai"]["output_tokens"]) == (40, 16)

    assert calls["ai.ner"]["ai"]["model_alias"] == "gliner"
    assert "output_sha256" not in calls["ai.ner"]["ai"]

    summarize = calls["ai.summarize"]
    assert summarize["ai"]["output_sha256"] == _sha(SUMMARY)
    assert summarize["criteria"]["inputs"] == ["translation"]
    assert translate["event_id"] in summarize["ai"]["sources"]

    hate = calls["ai.hate_speech"]
    assert hate["criteria"] == {"requests": 1, "segments": 2, "frames": 0}
    assert "output_sha256" not in hate["ai"]

    assert stubbed["transcription"]["scratch_dir"] == state.file_path.parent
    capture.assert_chain_ok()


def test_a_failing_stage_records_its_transfer_as_an_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When the ASR request fails the job fails, and the attempted transfer is still recorded."""

    def transcription(**kwargs: Any) -> TranscriptionOutcome:
        record_inference("vad", "vad", ok=True)
        record_inference("asr", "whisper-test", ok=False)
        raise RuntimeError("upstream rejected the audio")

    monkeypatch.setattr("nextext.pipeline.transcription_pipeline", transcription)
    audit, capture = capturing_job_audit()
    state = _job(tmp_path, audit)

    with pytest.raises(RuntimeError):
        _run_pipeline_blocking(state, lambda *a, **k: None)

    calls = _ai_calls(capture)
    assert (calls["ai.vad"]["outcome"], calls["ai.transcribe"]["outcome"]) == ("ok", "error")
    assert calls["ai.transcribe"]["criteria"] == {"requests": 1, "failed_requests": 1}
    assert "output_sha256" not in calls["ai.transcribe"]["ai"]
    assert "upstream rejected" not in capture.stream.getvalue()
    capture.assert_chain_ok()


def test_a_fail_soft_task_whose_requests_all_failed_is_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Diarization that degrades to no speakers is recorded as an error transfer; the job goes on."""

    def transcription(**kwargs: Any) -> TranscriptionOutcome:
        record_inference("asr", "whisper-test", ok=True)
        record_inference("diarize", "diarize", ok=False)
        return TranscriptionOutcome(transcript=TRANSCRIPT.drop(columns=["speaker"]), src_lang="en")

    monkeypatch.setattr("nextext.pipeline.transcription_pipeline", transcription)
    audit, capture = capturing_job_audit()
    result = _run_pipeline_blocking(_job(tmp_path, audit), lambda *a, **k: None)

    calls = _ai_calls(capture)
    assert (calls["ai.transcribe"]["outcome"], calls["ai.diarize"]["outcome"]) == ("ok", "error")
    assert not result.get("skipped")
    capture.assert_chain_ok()


def test_a_silent_clip_cites_only_its_captions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """With no speech, the caption, summary and frame-judging events cite only what the captions gave them."""
    empty = pd.DataFrame({"start": [], "end": [], "text": []})

    def transcription(**kwargs: Any) -> TranscriptionOutcome:
        record_inference("vad", "vad", ok=True)
        return TranscriptionOutcome(transcript=empty, src_lang="en", skip_reason="vad_no_speech")

    def describe(samples: Any, pipeline: Any, **kwargs: Any) -> list[FrameCaption]:
        record_inference("chat", "chat-model", ok=True)
        return [FrameCaption(time_sec=1.0, caption="An invented slide.")]

    def frame_hate(captions: Any, **kwargs: Any) -> list[dict[str, Any]]:
        record_inference("chat", "chat-model", ok=True)
        return []

    def summarize(text: str, **kwargs: Any) -> str:
        record_inference("chat", "chat-model", ok=True)
        return "A summary of an invented slide."

    monkeypatch.setattr("nextext.pipeline.transcription_pipeline", transcription)
    monkeypatch.setattr(jobs_module, "extract_keyframe_samples", lambda path, **kw: [Keyframe(jpeg=b"j", time_sec=1.0)])
    monkeypatch.setattr("nextext.core.visual_context.describe_keyframes", describe)
    monkeypatch.setattr("nextext.pipeline.frame_hate_speech_pipeline", frame_hate)
    monkeypatch.setattr("nextext.pipeline.summarization_pipeline", summarize)
    monkeypatch.setattr("nextext.core.openai_cfg.InferencePipeline", _HealthyInference)
    audit, capture = capturing_job_audit()
    collect = audit.collect({"files": 1})
    state = _job(tmp_path, audit, keyframes=True, summarization=True, hate_speech=True)

    result = _run_pipeline_blocking(state, lambda *a, **k: None)

    assert result["skipped"] is True
    calls = _ai_calls(capture)
    assert list(calls) == ["ai.vad", "ai.caption_frames", "ai.summarize", "ai.hate_speech"]
    captions = calls["ai.caption_frames"]
    assert captions["ai"]["sources"] == [collect["event_id"]]
    assert captions["ai"]["output_sha256"] == _sha("[00:01] An invented slide.")
    assert calls["ai.summarize"]["ai"]["sources"] == [captions["event_id"]]
    assert calls["ai.summarize"]["criteria"]["inputs"] == ["visual_context"]
    assert calls["ai.hate_speech"]["ai"]["sources"] == [captions["event_id"]]
    assert calls["ai.hate_speech"]["criteria"] == {"requests": 1, "frames": 1}
    capture.assert_chain_ok()


def test_combination_names_hate_speech_whenever_it_ran() -> None:
    """The combine event lists hate-speech detection by whether it ran, never by whether it found anything."""
    options = JobOptions.model_validate({"task": "transcribe", "hate_speech": True})
    found: dict[str, Any] = {"transcript": TRANSCRIPT, "hate_speech_findings": [FINDING]}
    nothing: dict[str, Any] = {"transcript": TRANSCRIPT, "hate_speech_findings": []}
    assert _joined_analyses(found, options) == _joined_analyses(nothing, options)
    analyses, speaker_attributed = _joined_analyses(found, options)
    assert analyses == ["transcript", "diarization", "hate_speech"]
    assert speaker_attributed is True
    plain = JobOptions.model_validate({"task": "transcribe"})
    assert _joined_analyses({"transcript": TRANSCRIPT.drop(columns=["speaker"])}, plain) == (["transcript"], False)


def test_combination_names_faces_whenever_the_stage_ran() -> None:
    """The combine event lists the face stage by whether it ran, never by what it found."""
    from nextext.core.face_extraction import FacesResult

    options = JobOptions.model_validate({"task": "transcribe"})
    empty = FacesResult(
        outcome="unavailable", grouped=False, truncated=False, entries=(), rows=(), frames_sent=4, tracks_sent=0
    )
    analyses, _speaker = _joined_analyses({"transcript": TRANSCRIPT, "faces": empty}, options)
    assert analyses == ["transcript", "diarization", "faces"]
    assert "faces" not in _joined_analyses({"transcript": TRANSCRIPT, "faces": None}, options)[0]
