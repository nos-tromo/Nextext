"""Tests for keyframe wiring in ``_run_pipeline_blocking`` (nextext.api.jobs).

The keyframe step is its own opt-in stage: ``JobOptions.keyframes`` gates both
sampling and captioning, independently of ``summarization``. Also covers
surfacing the stashed URL as ``JobResult.keyframes_url`` via
``_serialize_result``.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nextext.api import jobs as jobs_module
from nextext.api.jobs import JobState, _run_pipeline_blocking, _serialize_result
from nextext.api.schemas import JobOptions, JobStatus
from nextext.core.keyframes import Keyframe
from nextext.core.visual_context import FrameCaption
from nextext.pipeline import TranscriptionOutcome


def test_pipeline_populates_keyframes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The job result carries keyframes produced by ``extract_keyframes``."""
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"video")
    # Sampling only: the captioning half of the stage has its own tests below
    # and would otherwise reach for the inference provider.
    monkeypatch.setenv("NEXTEXT_VISUAL_SUMMARY", "off")

    # Stub the heavy stages so only keyframe wiring is exercised.
    df = pd.DataFrame({"start": [0.0], "end": [1.0], "speaker": [""], "text": ["hi"]})
    monkeypatch.setattr(
        "nextext.pipeline.transcription_pipeline", lambda **kwargs: TranscriptionOutcome(transcript=df, src_lang="en")
    )
    monkeypatch.setattr(
        jobs_module,
        "extract_keyframe_samples",
        lambda path, **kw: [
            Keyframe(time_sec=0.0, jpeg=b"\xff\xd8\xff0"),
            Keyframe(time_sec=5.0, jpeg=b"\xff\xd8\xff1"),
        ],
    )

    state = JobState(
        job_id="j1",
        owner_id="o",
        file_name="clip.mp4",
        file_path=media,
        source_file_hash="sha256:x",
        options=JobOptions.model_validate({"task": "transcribe", "keyframes": True}),
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
    monkeypatch.setenv("NEXTEXT_VISUAL_SUMMARY", "off")

    # Empty transcript (has the ``text`` column but no rows) trips the
    # skip-branch; keyframe extraction is stubbed to a known payload.
    empty = pd.DataFrame({"start": [], "end": [], "speaker": [], "text": []})
    monkeypatch.setattr(
        "nextext.pipeline.transcription_pipeline",
        lambda **kwargs: TranscriptionOutcome(transcript=empty, src_lang="en", skip_reason="vad_no_speech"),
    )
    monkeypatch.setattr(
        jobs_module,
        "extract_keyframe_samples",
        lambda path, **kw: [
            Keyframe(time_sec=0.0, jpeg=b"\xff\xd8\xff0"),
            Keyframe(time_sec=5.0, jpeg=b"\xff\xd8\xff1"),
        ],
    )

    state = JobState(
        job_id="j2",
        owner_id="o",
        file_name="clip.mp4",
        file_path=media,
        source_file_hash="sha256:x",
        options=JobOptions.model_validate({"task": "transcribe", "keyframes": True}),
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


# ---------------------------------------------------------------------------
# The keyframe stage (sampling + captioning), and what the summary makes of it
# ---------------------------------------------------------------------------


def _video_state(job_id: str, media: Path, **option_overrides: Any) -> JobState:
    """Build a job state for a video file.

    Args:
        job_id (str): Job identifier.
        media (Path): Path to the (stubbed) media file.
        **option_overrides (Any): Extra ``JobOptions`` fields.

    Returns:
        JobState: A queued job ready for ``_run_pipeline_blocking``.
    """
    options = {"task": "transcribe", **option_overrides}
    return JobState(
        job_id=job_id,
        owner_id="o",
        file_name="clip.mp4",
        file_path=media,
        source_file_hash="sha256:x",
        options=JobOptions.model_validate(options),
        status=JobStatus.QUEUED,
    )


@pytest.fixture
def _summarizable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Stub transcription, keyframes and the inference client for summary jobs.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching module attributes.
        tmp_path (Path): Temporary directory fixture.

    Returns:
        Path: The stub media file to hand to ``JobState``.
    """
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"video")
    df = pd.DataFrame({"start": [0.0], "end": [1.0], "speaker": [""], "text": ["hello"]})
    monkeypatch.setattr(
        "nextext.pipeline.transcription_pipeline", lambda **kwargs: TranscriptionOutcome(transcript=df, src_lang="en")
    )
    monkeypatch.setattr(
        jobs_module,
        "extract_keyframe_samples",
        lambda path, **kw: [Keyframe(time_sec=0.0, jpeg=b"\xff\xd8a"), Keyframe(time_sec=9.0, jpeg=b"\xff\xd8b")],
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
    return media


def test_pipeline_captions_keyframes_for_summary(monkeypatch: pytest.MonkeyPatch, _summarizable: Path) -> None:
    """A video summary job captions its frames and feeds them to the summarizer.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching module attributes.
        _summarizable (Path): Stub media path with the heavy stages patched out.
    """
    monkeypatch.delenv("NEXTEXT_VISUAL_SUMMARY", raising=False)
    monkeypatch.setattr(
        "nextext.core.visual_context.describe_keyframes",
        lambda samples, pipeline, **kw: [
            FrameCaption(time_sec=s.time_sec, caption=f"frame at {s.time_sec}") for s in samples
        ],
    )
    seen: dict[str, Any] = {}

    def _fake_summary(text: str, inference_pipeline: Any, visual_context: str | None = None) -> str:
        seen["text"] = text
        seen["visual_context"] = visual_context
        return "the summary"

    monkeypatch.setattr("nextext.pipeline.summarization_pipeline", _fake_summary)

    result = _run_pipeline_blocking(
        _video_state("v1", _summarizable, summarization=True, keyframes=True), lambda *a, **k: None
    )

    assert result["summary"] == "the summary"
    assert seen["visual_context"] == "[00:00] frame at 0.0\n[00:09] frame at 9.0"
    assert [c.caption for c in result["frame_captions"]] == ["frame at 0.0", "frame at 9.0"]


def test_pipeline_captions_keyframes_without_a_summary(monkeypatch: pytest.MonkeyPatch, _summarizable: Path) -> None:
    """Keyframes stand on their own: captions are produced with no summary asked.

    This is the whole point of the option — visual context is a result in its
    own right, not merely an input the summarizer happens to consume.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching module attributes.
        _summarizable (Path): Stub media path with the heavy stages patched out.
    """
    monkeypatch.delenv("NEXTEXT_VISUAL_SUMMARY", raising=False)
    monkeypatch.setattr(
        "nextext.core.visual_context.describe_keyframes",
        lambda samples, pipeline, **kw: [FrameCaption(time_sec=s.time_sec, caption="a room") for s in samples],
    )

    result = _run_pipeline_blocking(
        _video_state("v2", _summarizable, summarization=False, keyframes=True), lambda *a, **k: None
    )

    assert [c.caption for c in result["frame_captions"]] == ["a room", "a room"]
    assert result["keyframes"] == [b"\xff\xd8a", b"\xff\xd8b"]
    assert result["summary"] is None


def test_pipeline_samples_nothing_when_keyframes_not_requested(
    monkeypatch: pytest.MonkeyPatch, _summarizable: Path
) -> None:
    """A summary alone samples no frames — the option gates decoding, not just captions.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching module attributes.
        _summarizable (Path): Stub media path with the heavy stages patched out.
    """
    monkeypatch.delenv("NEXTEXT_VISUAL_SUMMARY", raising=False)
    sampled: list[Any] = []
    monkeypatch.setattr(
        jobs_module,
        "extract_keyframe_samples",
        lambda path, **kw: sampled.append(path) or [],
    )
    captioned: list[Any] = []
    monkeypatch.setattr(
        "nextext.core.visual_context.describe_keyframes",
        lambda samples, pipeline, **kw: captioned.append(samples) or [],
    )
    seen: dict[str, Any] = {}

    def _fake_summary(text: str, inference_pipeline: Any, visual_context: str | None = None) -> str:
        seen["visual_context"] = visual_context
        return "audio only"

    monkeypatch.setattr("nextext.pipeline.summarization_pipeline", _fake_summary)

    result = _run_pipeline_blocking(
        _video_state("v2b", _summarizable, summarization=True, keyframes=False), lambda *a, **k: None
    )

    assert sampled == []
    assert captioned == []
    assert seen["visual_context"] is None
    assert result["keyframes"] == []
    assert result["frame_captions"] is None


def test_pipeline_visual_context_disabled_by_env(monkeypatch: pytest.MonkeyPatch, _summarizable: Path) -> None:
    """The operator kill-switch suppresses captioning but keeps frames and summary.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching module attributes.
        _summarizable (Path): Stub media path with the heavy stages patched out.
    """
    monkeypatch.setenv("NEXTEXT_VISUAL_SUMMARY", "off")
    calls: list[Any] = []
    monkeypatch.setattr(
        "nextext.core.visual_context.describe_keyframes",
        lambda samples, pipeline, **kw: calls.append(samples) or [],
    )
    seen: dict[str, Any] = {}

    def _fake_summary(text: str, inference_pipeline: Any, visual_context: str | None = None) -> str:
        seen["visual_context"] = visual_context
        return "audio only"

    monkeypatch.setattr("nextext.pipeline.summarization_pipeline", _fake_summary)

    result = _run_pipeline_blocking(
        _video_state("v3", _summarizable, summarization=True, keyframes=True), lambda *a, **k: None
    )

    assert calls == []
    assert result["summary"] == "audio only"
    assert seen["visual_context"] is None
    assert result["frame_captions"] is None
    # The kill-switch stops captioning, not sampling: the frames are still
    # downloadable as ``keyframes.zip``.
    assert result["keyframes"] == [b"\xff\xd8a", b"\xff\xd8b"]


def test_pipeline_audio_only_job_has_no_visual_context(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An audio file yields no frames, so nothing is captioned and none is sent.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching module attributes.
        tmp_path (Path): Temporary directory fixture.
    """
    monkeypatch.delenv("NEXTEXT_VISUAL_SUMMARY", raising=False)
    media = tmp_path / "clip.mp3"
    media.write_bytes(b"audio")
    df = pd.DataFrame({"start": [0.0], "end": [1.0], "speaker": [""], "text": ["hello"]})
    monkeypatch.setattr(
        "nextext.pipeline.transcription_pipeline", lambda **kwargs: TranscriptionOutcome(transcript=df, src_lang="en")
    )
    monkeypatch.setattr(jobs_module, "extract_keyframe_samples", lambda path, **kw: [])

    class _StubInference:
        """Stand-in for ``InferencePipeline`` that is always healthy."""

        def get_health(self) -> bool:
            """Report the provider as reachable.

            Returns:
                bool: Always ``True``.
            """
            return True

    monkeypatch.setattr("nextext.core.openai_cfg.InferencePipeline", _StubInference)
    calls: list[Any] = []
    monkeypatch.setattr(
        "nextext.core.visual_context.describe_keyframes",
        lambda samples, pipeline, **kw: calls.append(samples) or [],
    )
    seen: dict[str, Any] = {}

    def _fake_summary(text: str, inference_pipeline: Any, visual_context: str | None = None) -> str:
        seen["visual_context"] = visual_context
        return "audio summary"

    monkeypatch.setattr("nextext.pipeline.summarization_pipeline", _fake_summary)

    result = _run_pipeline_blocking(_video_state("v4", media, summarization=True, keyframes=True), lambda *a, **k: None)

    assert calls == []
    assert seen["visual_context"] is None
    assert result["keyframes"] == []
    assert result["frame_captions"] is None


def test_pipeline_reports_caption_count_on_the_stage_event(
    monkeypatch: pytest.MonkeyPatch, _summarizable: Path
) -> None:
    """The keyframe stage's completion delta reports frames sampled and described.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching module attributes.
        _summarizable (Path): Stub media path with the heavy stages patched out.
    """
    monkeypatch.delenv("NEXTEXT_VISUAL_SUMMARY", raising=False)
    monkeypatch.setattr(
        "nextext.core.visual_context.describe_keyframes",
        lambda samples, pipeline, **kw: [FrameCaption(time_sec=0.0, caption="a room")],
    )
    monkeypatch.setattr("nextext.pipeline.summarization_pipeline", lambda *a, **kw: "s")
    events: list[tuple[str, dict[str, Any]]] = []

    _run_pipeline_blocking(
        _video_state("v5", _summarizable, summarization=True, keyframes=True),
        lambda name, payload: events.append((name, payload)),
    )

    completed = [p for name, p in events if name == "stage_completed" and p.get("stage") == "Describing keyframes"]
    assert completed[0]["result_delta"] == {"keyframes": 2, "frame_captions": 1}


def test_pipeline_keyframe_stage_reports_skipped_when_not_requested(
    monkeypatch: pytest.MonkeyPatch, _summarizable: Path
) -> None:
    """The stage still announces itself when the option is off, marked skipped.

    Every stage emits a started/completed pair regardless of its option, so the
    SSE progress fractions stay monotonic.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching module attributes.
        _summarizable (Path): Stub media path with the heavy stages patched out.
    """
    events: list[tuple[str, dict[str, Any]]] = []

    _run_pipeline_blocking(
        _video_state("v5b", _summarizable, keyframes=False),
        lambda name, payload: events.append((name, payload)),
    )

    stages = [p["stage"] for name, p in events if name == "stage_started"]
    assert stages[1] == "Describing keyframes"
    completed = [p for name, p in events if name == "stage_completed" and p.get("stage") == "Describing keyframes"]
    assert completed[0]["result_delta"] == {"skipped": True}


def test_pipeline_captioning_failure_does_not_fail_the_job(
    monkeypatch: pytest.MonkeyPatch, _summarizable: Path
) -> None:
    """An unexpected captioning error degrades to an audio-only summary.

    ``describe_keyframes`` is fail-soft by contract, but the worker must not
    depend on that being airtight — a summary is worth more than a failed job.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching module attributes.
        _summarizable (Path): Stub media path with the heavy stages patched out.
    """
    monkeypatch.delenv("NEXTEXT_VISUAL_SUMMARY", raising=False)

    def _boom(samples: Any, pipeline: Any, **kw: Any) -> list[Any]:
        raise RuntimeError("captioner exploded")

    monkeypatch.setattr("nextext.core.visual_context.describe_keyframes", _boom)
    monkeypatch.setattr("nextext.pipeline.summarization_pipeline", lambda *a, **kw: "still summarized")

    result = _run_pipeline_blocking(
        _video_state("v6", _summarizable, summarization=True, keyframes=True), lambda *a, **k: None
    )

    assert result["summary"] == "still summarized"
    assert result["frame_captions"] is None
    assert result["keyframes"] == [b"\xff\xd8a", b"\xff\xd8b"]


def test_serialize_result_surfaces_frame_captions() -> None:
    """Captions reach the API snapshot so the SPA can show the visual context."""
    serialized = _serialize_result({"frame_captions": [FrameCaption(time_sec=12.0, caption="a hallway")]})
    assert serialized.frame_captions is not None
    assert serialized.frame_captions[0].time_sec == 12.0
    assert serialized.frame_captions[0].caption == "a hallway"


def test_serialize_result_omits_frame_captions_when_absent() -> None:
    """A job with no captions advertises none rather than an empty list."""
    assert _serialize_result({}).frame_captions is None
    assert _serialize_result({"frame_captions": []}).frame_captions is None


def test_pipeline_captions_a_silent_video(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A video with no speech still gets its frames sampled and described.

    The job is still ``skipped`` — that flag means "no transcript" — but the
    visual half of the run is unaffected by the missing audio.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching module attributes.
        tmp_path (Path): Temporary directory fixture.
    """
    monkeypatch.delenv("NEXTEXT_VISUAL_SUMMARY", raising=False)
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"video")
    empty = pd.DataFrame({"start": [], "end": [], "speaker": [], "text": []})
    monkeypatch.setattr(
        "nextext.pipeline.transcription_pipeline",
        lambda **kwargs: TranscriptionOutcome(transcript=empty, src_lang="en", skip_reason="vad_no_speech"),
    )
    monkeypatch.setattr(
        jobs_module,
        "extract_keyframe_samples",
        lambda path, **kw: [Keyframe(time_sec=0.0, jpeg=b"\xff\xd8a")],
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
    monkeypatch.setattr(
        "nextext.core.visual_context.describe_keyframes",
        lambda samples, pipeline, **kw: [FrameCaption(time_sec=0.0, caption="an empty street")],
    )

    result = _run_pipeline_blocking(_video_state("s1", media, keyframes=True), lambda *a, **k: None)

    assert result["skipped"] is True
    assert result["skip_reason_code"] == "vad_no_speech"
    assert [c.caption for c in result["frame_captions"]] == ["an empty street"]
    assert result["summary"] is None


def test_pipeline_summarizes_a_silent_video_from_its_captions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No speech plus a requested summary yields one written from the visuals alone.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching module attributes.
        tmp_path (Path): Temporary directory fixture.
    """
    monkeypatch.delenv("NEXTEXT_VISUAL_SUMMARY", raising=False)
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"video")
    empty = pd.DataFrame({"start": [], "end": [], "speaker": [], "text": []})
    monkeypatch.setattr(
        "nextext.pipeline.transcription_pipeline",
        lambda **kwargs: TranscriptionOutcome(transcript=empty, src_lang="en", skip_reason="asr_empty_transcript"),
    )
    monkeypatch.setattr(
        jobs_module,
        "extract_keyframe_samples",
        lambda path, **kw: [Keyframe(time_sec=3.0, jpeg=b"\xff\xd8a")],
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
    monkeypatch.setattr(
        "nextext.core.visual_context.describe_keyframes",
        lambda samples, pipeline, **kw: [FrameCaption(time_sec=3.0, caption="a lit stage")],
    )
    seen: dict[str, Any] = {}

    def _fake_summary(text: str, inference_pipeline: Any, visual_context: str | None = None) -> str:
        seen["text"] = text
        seen["visual_context"] = visual_context
        return "a visual summary"

    monkeypatch.setattr("nextext.pipeline.summarization_pipeline", _fake_summary)
    events: list[tuple[str, dict[str, Any]]] = []

    result = _run_pipeline_blocking(
        _video_state("s2", media, keyframes=True, summarization=True),
        lambda name, payload: events.append((name, payload)),
    )

    assert result["skipped"] is True
    assert result["summary"] == "a visual summary"
    assert seen["text"] == ""
    assert seen["visual_context"] == "[00:03] a lit stage"
    # The summarize stage is announced on this path too, so the progress bar
    # does not stall silently at the keyframe stage.
    assert [p["stage"] for name, p in events if name == "stage_started"][-1] == "Summarizing"


def test_pipeline_silent_video_without_captions_has_no_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No transcript and no captions means no summary request is made at all.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching module attributes.
        tmp_path (Path): Temporary directory fixture.
    """
    media = tmp_path / "clip.mp3"
    media.write_bytes(b"audio")
    empty = pd.DataFrame({"start": [], "end": [], "speaker": [], "text": []})
    monkeypatch.setattr(
        "nextext.pipeline.transcription_pipeline",
        lambda **kwargs: TranscriptionOutcome(transcript=empty, src_lang="en", skip_reason="vad_no_speech"),
    )
    monkeypatch.setattr(jobs_module, "extract_keyframe_samples", lambda path, **kw: [])
    calls: list[Any] = []
    monkeypatch.setattr(
        "nextext.pipeline.summarization_pipeline",
        lambda *a, **kw: calls.append(a) or "unreachable",
    )

    result = _run_pipeline_blocking(_video_state("s3", media, keyframes=True, summarization=True), lambda *a, **k: None)

    assert calls == []
    assert result["summary"] is None
    assert result["frame_captions"] is None


def test_visual_context_off_samples_frames_without_describing_them(
    monkeypatch: pytest.MonkeyPatch, _summarizable: Path
) -> None:
    """``visual_context=False`` keeps the frames and skips the vision calls.

    This is the split the option exists for: a client can want the JPEGs
    without the words. docint samples keyframes to run its own image pipeline
    over them and never reads ``frame_captions``, so describing them was a
    vision request per frame spent on an answer nobody collected.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching module attributes.
        _summarizable (Path): Stub media path with the heavy stages patched out.
    """
    monkeypatch.delenv("NEXTEXT_VISUAL_SUMMARY", raising=False)

    def _must_not_caption(*args: Any, **kwargs: Any) -> list[FrameCaption]:
        raise AssertionError("describe_keyframes must not run when visual_context is off")

    monkeypatch.setattr("nextext.core.visual_context.describe_keyframes", _must_not_caption)

    result = _run_pipeline_blocking(
        _video_state("v1", _summarizable, keyframes=True, visual_context=False), lambda *a, **k: None
    )

    assert result["keyframes"] == [b"\xff\xd8a", b"\xff\xd8b"]
    assert result["_keyframes_url"] == "/api/v1/jobs/v1/artifacts/keyframes.zip"
    assert result["frame_captions"] is None


def test_visual_context_off_keeps_the_transcript_when_the_provider_is_down(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A frames-only job survives an unreachable chat provider.

    ``_ensure_inference`` sits outside the captioning fail-soft guard on
    purpose, and the keyframe stage runs first — so before this option existed,
    a caller that asked for keyframes and no model-backed stage still had its
    job failed, and its transcript discarded, by an unhealthy chat router it
    never meant to use.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for patching module attributes.
        tmp_path (Path): Temporary directory fixture.
    """
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"video")
    monkeypatch.delenv("NEXTEXT_VISUAL_SUMMARY", raising=False)
    df = pd.DataFrame({"start": [0.0], "end": [1.0], "speaker": [""], "text": ["hello"]})
    monkeypatch.setattr(
        "nextext.pipeline.transcription_pipeline", lambda **kwargs: TranscriptionOutcome(transcript=df, src_lang="en")
    )
    monkeypatch.setattr(
        jobs_module,
        "extract_keyframe_samples",
        lambda path, **kw: [Keyframe(time_sec=0.0, jpeg=b"\xff\xd8a")],
    )

    class _DeadInference:
        """Stand-in provider that is reachable-but-unhealthy."""

        def get_health(self) -> bool:
            """Report the provider as unreachable.

            Returns:
                bool: Always ``False``.
            """
            return False

    monkeypatch.setattr("nextext.core.openai_cfg.InferencePipeline", _DeadInference)

    result = _run_pipeline_blocking(
        _video_state("v2", media, keyframes=True, visual_context=False), lambda *a, **k: None
    )

    assert result["keyframes"] == [b"\xff\xd8a"]
    assert result["frame_captions"] is None
    assert not result["transcript"].empty
