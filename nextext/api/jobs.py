"""Job state, manager, and worker for the Nextext FastAPI backend.

The :class:`JobManager` owns an in-memory dictionary of :class:`JobState`
instances. Jobs are processed by async workers constrained by an
``asyncio.Semaphore`` sized from ``NEXTEXT_JOB_CONCURRENCY`` (default ``1``,
i.e. serial execution): pipeline stages are CPU-bound (decoding, DataFrame
shaping) around blocking calls to the shared inference tier, so the worker
count bounds memory use and how fairly the remote endpoints are shared.

The blocking pipeline runs on a worker thread via
``anyio.to_thread.run_sync``. Stage transitions are published to a per-job
list of subscriber queues so any number of SSE clients can attach to the
same job and receive the same events.

The manager also writes the job's data-protection audit events
(:mod:`nextext.audit`): the upload's ``collect``, one ``ai_call`` per
inference task of every stage, the result's ``combine``, and every
``delete`` — by the owner, at shutdown, and of the uploads a previous process
left behind, which it attributes through the owner record kept next to each
upload (:mod:`nextext.api.job_record`).
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import os
import secrets
import shutil
import tempfile
import uuid
from collections.abc import AsyncIterator, Callable, Hashable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from matplotlib.figure import Figure

from nextext.api.job_record import JobRecord, read_job_record, write_job_record
from nextext.api.metrics import record_completed, record_failed, record_skipped
from nextext.api.schemas import (
    FrameCaptionOut,
    HateSpeechFinding,
    JobOptions,
    JobResult,
    JobSnapshot,
    JobStatus,
    NamedEntity,
    TranscriptSegment,
    WordCount,
)
from nextext.audit import (
    CAPTION_STAGE,
    FRAME_HATE_SPEECH_STAGE,
    HATE_SPEECH_STAGE,
    SUMMARY_STAGE,
    TRANSCRIPTION_STAGE,
    TRANSLATION_STAGE,
    VISUAL_SUMMARY_STAGE,
    WORDLEVEL_STAGE,
    AuditSink,
    JobAudit,
    Origin,
    emit_with_ids,
    joined_analyses,
)
from nextext.core.audio import AudioDecodeError
from nextext.core.docint_transcript import parse_hhmmss_to_seconds
from nextext.core.inference_trace import correlate
from nextext.core.keyframes import extract_keyframe_samples
from nextext.core.outcomes import FailureCode, SkipReason, skip_reason_text
from nextext.utils.audit_event import Actor, AuditEmitter, Target
from nextext.utils.env_cfg import load_job_concurrency, load_visual_summary_env
from nextext.utils.log_cfg import error_label, exception_frames

_TERMINAL_EVENT_NAMES: frozenset[str] = frozenset({"job_completed", "job_failed", "job_cancelled"})


PIPELINE_STAGE_LABELS: tuple[str, ...] = (
    "Transcribing",
    "Describing keyframes",
    "Translating",
    "Running word-level analysis",
    "Summarizing",
    "Detecting hate speech",
)


def _utcnow() -> datetime:
    """Return a timezone-aware UTC ``datetime`` used for all timestamps.

    Returns:
        datetime: Current time with explicit UTC offset.
    """
    return datetime.now(tz=UTC)


def _format_sse(event_name: str, payload: dict[str, Any]) -> bytes:
    """Render a payload as an SSE event frame.

    Args:
        event_name: SSE event name (e.g. ``stage_started``).
        payload: JSON-serializable payload.

    Returns:
        bytes: UTF-8 encoded frame ready to send over ``text/event-stream``.
    """
    body = json.dumps(payload, default=str)
    return f"event: {event_name}\ndata: {body}\n\n".encode()


@dataclass
class JobState:
    """Mutable state for one in-flight or completed job.

    ``owner_id`` is the principal resolved per request by
    :func:`nextext.api.identity.resolve_principal`; the routes consult it
    to enforce per-owner access. Jobs live only in memory — they survive a
    browser reload (the frontend re-discovers them by owner) but never
    touch durable storage.
    """

    job_id: str
    owner_id: str
    file_name: str
    file_path: Path
    source_file_hash: str
    options: JobOptions
    # Capability granting read access to the upload. A ``<video src>`` cannot
    # send the identity header, so playback authorizes with this instead of
    # with the principal. Minted per job and never listed.
    media_token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    status: JobStatus = JobStatus.QUEUED
    stage: str | None = None
    stage_index: int = 0
    progress: float = 0.0
    error: str | None = None
    # Typed cause of a failure. ``error`` stays the static, non-leaking
    # message; this is what lets a client explain the failure.
    error_code: FailureCode | None = None
    created_at: datetime = field(default_factory=_utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: dict[str, Any] = field(default_factory=dict)
    event_history: list[tuple[str, bytes]] = field(default_factory=list)
    subscribers: list[asyncio.Queue[bytes | None]] = field(default_factory=list)
    # Rendered archive members (archive-relative name -> decompressed bytes),
    # cached so the per-job and batch ZIP builders reuse the expensive render
    # (XLSX, word cloud, docint) instead of recomputing it; only the cheap ZIP
    # compression is redone per download.
    archive_members_cache: dict[str, bytes] | None = None
    # Worker task processing this job, so a delete can cancel a queued worker
    # before it runs the pipeline against a removed upload.
    task: asyncio.Task[None] | None = None
    # Set by delete() while the pipeline thread is still running: the worker
    # owns the tempdir cleanup (in its finally block) and suppresses further
    # event dispatch for the no-longer-listed job.
    deleted: bool = False
    # The job's audit trail: every event of the job carries the actor,
    # purpose and request id of the request that created it.
    audit: JobAudit = field(kw_only=True, repr=False)

    @property
    def media_url(self) -> str | None:
        """Capability URL for playing this job's upload, if the bytes remain.

        Resolved per call rather than pre-baked: the upload outlives the
        pipeline but not a delete, so a job whose file is gone must stop
        advertising a URL that would only 404.

        Returns:
            str | None: The tokenized media URL, or ``None`` when the upload
                is no longer on disk.
        """
        if not self.file_path.exists():
            return None
        return f"/api/v1/jobs/{self.job_id}/media?token={self.media_token}"

    @property
    def is_skipped(self) -> bool:
        """Whether the job completed without producing a transcript.

        Returns:
            bool: ``True`` for a completed job whose result is a skip payload.
        """
        return bool(self.result.get("skipped", False))

    @property
    def skip_reason_code(self) -> SkipReason | None:
        """Typed reason the job produced no transcript.

        Returns:
            SkipReason | None: The recorded code, or ``None`` when the job
            was not skipped.
        """
        return self.result.get("skip_reason_code")

    def snapshot(self) -> JobSnapshot:
        """Return a Pydantic snapshot suitable for JSON serialization.

        Returns:
            JobSnapshot: Point-in-time view of the job.
        """
        return JobSnapshot(
            job_id=self.job_id,
            status=self.status,
            file_name=self.file_name,
            source_file_hash=self.source_file_hash or None,
            options=self.options,
            stage=self.stage,
            stage_index=self.stage_index,
            progress=self.progress,
            error=self.error,
            error_code=self.error_code,
            skipped=self.is_skipped,
            skip_reason_code=self.skip_reason_code,
            created_at=self.created_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
            result=(
                _serialize_result(self.result, media_url=self.media_url) if self.status == JobStatus.COMPLETED else None
            ),
        )


def _normalize_transcript_row(row: dict[Hashable, Any]) -> TranscriptSegment:
    """Convert a transcript DataFrame row to ``TranscriptSegment``.

    ``translation`` is only present in the row when translation has run (see
    :func:`nextext.pipeline.translation_pipeline`); it is ``None`` for
    transcribe-only jobs so the transcript and translation can be
    cross-referenced side by side in the UI table and CSV/XLSX exports.

    Args:
        row: One row from the transcript DataFrame.

    Returns:
        TranscriptSegment: JSON-friendly view of the row.
    """
    return TranscriptSegment(
        start=_optional_str(row.get("start")),
        end=_optional_str(row.get("end")),
        start_seconds=_optional_seconds(row.get("start")),
        end_seconds=_optional_seconds(row.get("end")),
        speaker=_optional_str(row.get("speaker")),
        text=str(row.get("text", "")),
        translation=_optional_str(row.get("translation")),
    )


def _optional_seconds(value: Any) -> float | None:
    """Parse a transcript timestamp to seconds, fail-soft.

    The transcript stores ``str(timedelta)`` for display; the player needs a
    number. A row whose timestamp cannot be parsed simply loses its seek
    rather than failing the whole snapshot.

    Args:
        value: A timestamp cell from the transcript DataFrame.

    Returns:
        float | None: Offset in seconds, or ``None`` when absent/unparseable.
    """
    if _optional_str(value) is None:
        return None
    try:
        return parse_hhmmss_to_seconds(value)
    except (ValueError, IndexError):
        return None


def _optional_str(value: Any) -> str | None:
    """Coerce a value to ``str`` unless it is ``None`` / ``NaN``.

    Args:
        value: Value to coerce.

    Returns:
        str | None: ``None`` for ``None``/``NaN``, otherwise ``str(value)``.
    """
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return str(value)


def _normalize_word_counts(df: pd.DataFrame | None) -> list[WordCount] | None:
    """Convert a word-counts DataFrame to a list of ``WordCount``.

    The DataFrame's first two columns hold the word and its frequency; the
    column names depend on the language of the input. This helper normalizes
    them to the schema-mandated ``word``/``count`` names.

    Args:
        df: Word-counts DataFrame from ``wordlevel_pipeline``.

    Returns:
        list[WordCount] | None: Normalized rows, or ``None`` when missing.
    """
    if df is None or df.empty:
        return None
    word_col = df.columns[0]
    count_col = df.columns[1]
    return [WordCount(word=str(row[word_col]), count=int(row[count_col])) for _, row in df.iterrows()]


def _normalize_named_entities(df: pd.DataFrame | None) -> list[NamedEntity] | None:
    """Convert a named-entities DataFrame to a list of ``NamedEntity``.

    Args:
        df: Named-entities DataFrame with ``Entity``/``Category``/``Frequency``
            columns.

    Returns:
        list[NamedEntity] | None: Normalized rows, or ``None`` when missing.
    """
    if df is None or df.empty:
        return None
    return [
        NamedEntity(
            entity=str(row["Entity"]),
            category=str(row["Category"]),
            frequency=int(row["Frequency"]),
        )
        for _, row in df.iterrows()
    ]


def _normalize_hate_speech(items: list[dict[str, Any]] | None) -> list[HateSpeechFinding] | None:
    """Coerce hate-speech findings into ``HateSpeechFinding`` instances.

    Args:
        items: Raw list of detection dicts from ``hate_speech_pipeline``.

    Returns:
        list[HateSpeechFinding] | None: Validated findings or ``None``.
    """
    if not items:
        return None
    return [HateSpeechFinding(**item) for item in items]


def _serialize_result(result: dict[str, Any], media_url: str | None = None) -> JobResult:
    """Convert an in-memory result dict into the JSON-friendly schema.

    The wordcloud Figure is intentionally not serialized into the JSON
    response — clients download it on demand via the artifacts endpoint.

    Args:
        result: In-memory result payload populated by the worker.
        media_url: Capability URL for the original upload, resolved by the
            caller (which holds the ``JobState``); ``None`` when the bytes are
            gone or the caller has no state to ask.

    Returns:
        JobResult: Pydantic representation suitable for HTTP responses.
    """
    transcript_df = result.get("transcript")
    transcript: list[TranscriptSegment] = []
    if isinstance(transcript_df, pd.DataFrame) and not transcript_df.empty:
        transcript = [_normalize_transcript_row(row) for row in transcript_df.to_dict(orient="records")]

    wordcloud_url: str | None = None
    if isinstance(result.get("wordcloud"), Figure):
        wordcloud_url = result.get("_wordcloud_url")

    keyframes_url: str | None = None
    if result.get("keyframes"):
        keyframes_url = result.get("_keyframes_url")

    frame_captions: list[FrameCaptionOut] | None = None
    if result.get("frame_captions"):
        frame_captions = [
            FrameCaptionOut(time_sec=caption.time_sec, caption=caption.caption) for caption in result["frame_captions"]
        ]

    return JobResult(
        transcript=transcript,
        transcript_language=result.get("transcript_language"),
        resolved_src_lang=result.get("resolved_src_lang"),
        summary=result.get("summary"),
        word_counts=_normalize_word_counts(result.get("word_counts")),
        named_entities=_normalize_named_entities(result.get("named_entities")),
        wordcloud_url=wordcloud_url,
        keyframes_url=keyframes_url,
        media_url=media_url,
        frame_captions=frame_captions,
        hate_speech_findings=_normalize_hate_speech(result.get("hate_speech_findings")),
        skipped=bool(result.get("skipped", False)),
        skip_reason=result.get("skip_reason"),
        skip_reason_code=result.get("skip_reason_code"),
        task=result.get("task", "transcribe"),
    )


def _requested_analyses(options: JobOptions) -> list[str]:
    """Name the analyses a job asked for, for its ``collect`` event.

    Args:
        options: The job's options.

    Returns:
        list[str]: Transcription always, then each opt-in analysis requested.
    """
    requested = ["transcription"]
    for name, wanted in (
        ("diarization", options.diarize),
        ("translation", options.task == "translate"),
        ("word_level", options.words),
        ("summary", options.summarization),
        ("hate_speech", options.hate_speech),
        ("keyframes", options.keyframes),
    ):
        if wanted:
            requested.append(name)
    return requested


def _collect_criteria(file_path: Path, source_file_hash: str, options: JobOptions) -> dict[str, object]:
    """Describe an upload for its ``collect`` event: size, hash, what was asked.

    Never the file name, which can name the people in the recording.

    Args:
        file_path: The stored upload.
        source_file_hash: Its ``sha256:...`` digest.
        options: The job's options.

    Returns:
        dict[str, object]: The event's criteria.
    """
    return {
        "files": 1,
        "bytes": file_path.stat().st_size,
        "source_sha256": source_file_hash.removeprefix("sha256:"),
        "analyses": _requested_analyses(options),
        "options": options.model_dump(mode="json"),
    }


def _is_filled_frame(value: Any) -> bool:
    """Report whether a result entry is a non-empty DataFrame.

    Args:
        value: A result entry.

    Returns:
        bool: ``True`` for a DataFrame with rows.
    """
    return isinstance(value, pd.DataFrame) and not value.empty


def _joined_analyses(result: dict[str, Any], options: JobOptions) -> tuple[list[str], bool]:
    """Name the analyses a finished job's result joins, for its ``combine`` event.

    Each analysis is listed when the result carries its output — except
    hate-speech detection, listed whenever it ran (see
    :func:`nextext.audit.joined_analyses`).

    Args:
        result: The finished job's result payload.
        options: The job's options.

    Returns:
        tuple[list[str], bool]: The analyses, and whether the results carry
            speaker labels.
    """
    transcript = result.get("transcript")
    columns = list(transcript.columns) if _is_filled_frame(transcript) and isinstance(transcript, pd.DataFrame) else []
    has_captions = bool(result.get("frame_captions"))
    return joined_analyses(
        transcript_columns=columns,
        word_counts=_is_filled_frame(result.get("word_counts")),
        named_entities=_is_filled_frame(result.get("named_entities")),
        summary=bool(result.get("summary")),
        frame_captions=has_captions,
        hate_speech=options.hate_speech and (bool(columns) or has_captions),
    )


PushEvent = Callable[[str, dict[str, Any]], None]

# How the summarizer's transcript input is named in its audit event.
_SUMMARY_INPUT: dict[str, str] = {"text": "transcript", "translation": "translation"}


def _run_pipeline_blocking(state: JobState, push_event: PushEvent) -> dict[str, Any]:
    """Run the full Nextext pipeline for one job, emitting SSE events.

    Stage transitions are published through ``push_event`` so SSE subscribers
    see the same progression as the job advances through the pipeline.

    Args:
        state: The job to process.
        push_event: Thread-safe callable for emitting SSE events.

    Returns:
        dict[str, Any]: In-memory result payload for the completed job.

    Raises:
        ConnectionError: If the configured inference provider is unreachable.
    """
    # Local imports keep test stubs and import-time module reloads predictable.
    from nextext.core.openai_cfg import InferencePipeline
    from nextext.core.visual_context import FrameCaption, describe_keyframes, format_visual_context
    from nextext.pipeline import (
        column_text,
        effective_text_column,
        frame_hate_speech_pipeline,
        hate_speech_pipeline,
        normalize_language_code,
        should_translate,
        summarization_pipeline,
        transcription_pipeline,
        translation_pipeline,
        wordlevel_pipeline,
    )

    opts = state.options
    file_opts: dict[str, Any] = {
        "src_lang": opts.src_lang,
        "trg_lang": opts.trg_lang,
        "task": opts.task,
        "diarize": opts.diarize,
        "words": opts.words,
        "summarization": opts.summarization,
        "hate_speech": opts.hate_speech,
        "keyframes": opts.keyframes,
        "visual_context": opts.visual_context,
    }
    total_stages = len(PIPELINE_STAGE_LABELS)

    def _notify(stage_index: int) -> None:
        """Emit a ``stage_started`` event for the given stage.

        Args:
            stage_index: Zero-based pipeline stage index.
        """
        push_event(
            "stage_started",
            {
                "stage": PIPELINE_STAGE_LABELS[stage_index],
                "stage_index": stage_index,
                "progress": stage_index / total_stages,
                "timestamp": _utcnow().isoformat(),
            },
        )

    def _complete(stage_index: int, delta: dict[str, Any] | None = None) -> None:
        """Emit a ``stage_completed`` event with optional partial result.

        Args:
            stage_index: Zero-based pipeline stage index that just completed.
            delta: Optional JSON-friendly fragment of the result to surface
                to subscribers (e.g. ``{"transcript": [...]}``).
        """
        push_event(
            "stage_completed",
            {
                "stage": PIPELINE_STAGE_LABELS[stage_index],
                "stage_index": stage_index,
                "progress": (stage_index + 1) / total_stages,
                "timestamp": _utcnow().isoformat(),
                "result_delta": delta,
            },
        )

    inference_pipeline: InferencePipeline | None = None

    def _ensure_inference() -> InferencePipeline:
        """Return the job's inference pipeline, creating it on first use.

        The provider is health-checked once, when the pipeline is built: a
        stage that needs inference on an unreachable provider fails the job
        rather than silently producing nothing.

        Returns:
            InferencePipeline: The shared pipeline for this job.

        Raises:
            ConnectionError: If the configured provider is unreachable.
        """
        nonlocal inference_pipeline
        if inference_pipeline is None:
            candidate = InferencePipeline()
            if not candidate.get_health():
                raise ConnectionError(
                    "The configured inference provider is not reachable. Please ensure it is running and accessible."
                )
            inference_pipeline = candidate
        return inference_pipeline

    # Every inference request a stage makes is tallied per task and recorded
    # as one ai_call event per task when the stage ends (nextext.audit). A
    # failed audit write raises out of the stage and fails the job.
    audit = state.audit

    # Transcription -----------------------------------------------------------
    _notify(0)
    with audit.stage(TRANSCRIPTION_STAGE) as notes:
        outcome = transcription_pipeline(
            file_path=state.file_path,
            src_lang=file_opts["src_lang"] or "",
            diarize=file_opts["diarize"],
            # The FLAC re-encode sits next to the upload, so a crash leaves no
            # copy of the audio outside the job directory.
            scratch_dir=state.file_path.parent,
        )
        notes.result("asr", text=column_text(outcome.transcript, "text"), segments=len(outcome.transcript))
    df = outcome.transcript
    file_opts["src_lang"] = outcome.src_lang
    # A file with no processable speech is a completed job, not a failure: the
    # typed code is what the SPA localizes and what the operator alerts on;
    # ``skip_reason`` is English fallback prose. The text stages below are
    # skipped, but the keyframe stage is not — a silent video still has
    # something to show.
    reason_code = outcome.skip_reason
    if outcome.is_empty:
        logger.warning(
            "Job {} skipped: {} (text stages after transcription not run).",
            state.job_id,
            reason_code or "unknown",
        )
        _complete(0, {"transcript_segments": 0, "skipped": True, "skip_reason_code": reason_code})
    else:
        _complete(
            0,
            {
                "transcript_segments": len(df),
                "resolved_src_lang": file_opts["src_lang"],
            },
        )

    # Keyframes ---------------------------------------------------------------
    # Sampling and captioning are one opt-in step of their own: a summary no
    # longer implies visual context, and visual context no longer implies a
    # summary. Captioning is the expensive half — one vision request per frame —
    # and carries two gates of its own: the per-job ``visual_context`` (for a
    # client that wants the frames but not the words) and the operator-level
    # ``NEXTEXT_VISUAL_SUMMARY``. Either one off leaves the sampled frames
    # sampled and downloadable, and skips the provider entirely.
    _notify(1)
    keyframes: list[bytes] = []
    keyframe_times: list[float] = []
    captions: list[FrameCaption] = []
    # Pre-baked here (not in ``_serialize_result``) because ``state.job_id`` is
    # in scope; mirrors the ``_wordcloud_url`` pattern below.
    keyframes_url = f"/api/v1/jobs/{state.job_id}/artifacts/keyframes.zip"
    if file_opts["keyframes"]:
        keyframe_samples = extract_keyframe_samples(
            state.file_path,
            per_minute=opts.keyframes_per_minute,
            max_frames=opts.keyframes_max,
        )
        keyframes = [sample.jpeg for sample in keyframe_samples]
        keyframe_times = [sample.time_sec for sample in keyframe_samples]
        visual_cfg = load_visual_summary_env()
        if visual_cfg.enabled and file_opts["visual_context"] and keyframe_samples:
            # Resolved outside the fail-soft guard: an unreachable provider is
            # a job failure like it is for every other stage, not a silently
            # uncaptioned run. Which is also why the per-job gate above matters
            # beyond cost: this stage runs first, so a caller that opts out of
            # captioning — and asks for no other model-backed stage — never
            # touches the chat provider and cannot lose its transcript to one.
            caption_pipeline = _ensure_inference()
            with audit.stage(CAPTION_STAGE) as notes:
                try:
                    captions = describe_keyframes(
                        keyframe_samples,
                        caption_pipeline,
                        max_frames=visual_cfg.max_frames,
                        max_side=visual_cfg.max_side,
                    )
                except Exception as exc:
                    logger.warning("Job {} visual context unavailable: {}", state.job_id, error_label(exc))
                    captions = []
                notes.result("chat", text=format_visual_context(captions) if captions else None)
        _complete(1, {"keyframes": len(keyframes), "frame_captions": len(captions)})
    else:
        logger.info("Job {} keyframes not requested.", state.job_id)
        _complete(1, {"skipped": True})

    visual_context = format_visual_context(captions) if captions else None

    if outcome.is_empty:
        transcript_language = file_opts["trg_lang" if file_opts["task"] == "translate" else "src_lang"]
        # A summary was still requested, and the captions are a summarizable
        # source in their own right — so a silent video yields one.
        visual_summary: str | None = None
        if file_opts["summarization"] and visual_context:
            _notify(4)
            with audit.stage(VISUAL_SUMMARY_STAGE) as notes:
                visual_summary = summarization_pipeline(
                    "",
                    inference_pipeline=_ensure_inference(),
                    visual_context=visual_context,
                )
                notes.result("chat", text=visual_summary, inputs=["visual_context"])
            _complete(4, {"summary": bool(visual_summary)})
        # Likewise the captions can show hate a silent clip never speaks.
        frame_findings: list[dict[str, Any]] | None = None
        if file_opts["hate_speech"] and captions:
            _notify(5)
            with audit.stage(FRAME_HATE_SPEECH_STAGE) as notes:
                frame_findings = frame_hate_speech_pipeline(captions, inference_pipeline=_ensure_inference())
                notes.result("chat", frames=len(captions))
            _complete(5, {"flagged": len(frame_findings)})
        return {
            "transcript": df,
            "summary": visual_summary,
            "word_counts": None,
            "named_entities": None,
            "wordcloud": None,
            "hate_speech_findings": frame_findings or None,
            "resolved_src_lang": file_opts["src_lang"],
            "transcript_language": transcript_language,
            "skipped": True,
            "skip_reason": skip_reason_text(reason_code),
            "skip_reason_code": reason_code,
            "task": file_opts["task"],
            "keyframes": keyframes,
            "keyframe_times": keyframe_times,
            "_keyframes_url": keyframes_url,
            "frame_captions": captions or None,
        }

    # Translation -------------------------------------------------------------
    # Whisper only transcribes; the LLM performs every translation, directly
    # from the source language to the target. Engage it whenever a translate
    # task was requested and the resolved source differs from the target (so an
    # English target is translated too, and a same-language request is a no-op).
    _notify(2)
    needs_translation = should_translate(file_opts["task"], file_opts["src_lang"], file_opts["trg_lang"])
    if needs_translation:
        with audit.stage(TRANSLATION_STAGE) as notes:
            df = translation_pipeline(
                df,
                file_opts["trg_lang"],
                src_lang=file_opts["src_lang"],
                inference_pipeline=_ensure_inference(),
            )
            notes.result(
                "chat",
                text=column_text(df, "translation"),
                src_lang=file_opts["src_lang"],
                trg_lang=file_opts["trg_lang"],
            )
        _complete(2, {"translated": True})
    else:
        _complete(2, {"translated": False})

    transcript_language = file_opts["trg_lang" if file_opts["task"] == "translate" else "src_lang"]
    result: dict[str, Any] = {
        "transcript": df,
        "summary": None,
        "word_counts": None,
        "named_entities": None,
        "wordcloud": None,
        "hate_speech_findings": None,
        "resolved_src_lang": file_opts["src_lang"],
        "transcript_language": transcript_language,
        "task": file_opts["task"],
        "keyframes": keyframes,
        "keyframe_times": keyframe_times,
        "_keyframes_url": keyframes_url,
        "frame_captions": captions or None,
    }

    # Word-level analysis -----------------------------------------------------
    _notify(3)
    if file_opts["words"]:
        with audit.stage(WORDLEVEL_STAGE):
            wc, ner, cloud = wordlevel_pipeline(
                df,
                normalize_language_code(transcript_language) or "en",
            )
        result["word_counts"] = wc
        result["named_entities"] = ner
        result["wordcloud"] = cloud
        result["_wordcloud_url"] = f"/api/v1/jobs/{state.job_id}/artifacts/wordcloud.png"
        _complete(
            3,
            {
                "word_counts": len(wc) if wc is not None else 0,
                "named_entities": len(ner) if ner is not None else 0,
                "wordcloud": cloud is not None,
            },
        )
    else:
        _complete(3, {"skipped": True})

    # Summarization -----------------------------------------------------------
    # The summarizer takes whatever the job produced: the transcript (its
    # translation when there is one) and the keyframe captions when the
    # keyframe step ran, either alone or together.
    _notify(4)
    if file_opts["summarization"]:
        summary_text_column = effective_text_column(df)
        with audit.stage(SUMMARY_STAGE) as notes:
            result["summary"] = summarization_pipeline(
                " ".join(df[summary_text_column].astype(str).tolist()),
                inference_pipeline=_ensure_inference(),
                visual_context=visual_context,
            )
            notes.result(
                "chat",
                text=result["summary"],
                inputs=[_SUMMARY_INPUT[summary_text_column], *(["visual_context"] if visual_context else [])],
            )
        _complete(4, {"summary": bool(result["summary"])})
    else:
        _complete(4, {"skipped": True})

    # Hate-speech detection ---------------------------------------------------
    _notify(5)
    if file_opts["hate_speech"]:
        # Recorded by what was judged, never by what was found: the findings
        # can amount to special categories of personal data.
        with audit.stage(HATE_SPEECH_STAGE) as notes:
            findings = hate_speech_pipeline(
                df=df, inference_pipeline=_ensure_inference(), src_lang=file_opts["src_lang"] or None
            )
            findings += frame_hate_speech_pipeline(captions, inference_pipeline=_ensure_inference())
            notes.result("chat", segments=len(df), frames=len(captions))
        result["hate_speech_findings"] = findings
        _complete(5, {"flagged": len(findings)})
    else:
        _complete(5, {"skipped": True})

    return result


class JobManager:
    """In-memory job store and worker dispatcher.

    Jobs live only in memory. Async workers, bounded by an
    ``asyncio.Semaphore`` sized from ``NEXTEXT_JOB_CONCURRENCY`` (default
    ``1``, i.e. serial execution), process them; SSE subscribers attach to a
    per-job event history that is replayed on connect, so a browser that
    reloads mid-run re-attaches and resumes the live view. There is no
    durable storage and no TTL: a job is retained until its owner deletes it
    or the process exits.

    Each upload lives in a job directory under ``tmp_root``, next to a small
    owner record (:mod:`nextext.api.job_record`). :meth:`start` removes the
    directories a previous process left behind and records their deletion,
    attributed through those records; :meth:`stop` removes every job's
    directory and records that too.
    """

    def __init__(
        self,
        tmp_root: Path | None = None,
        pipeline_runner: Callable[[JobState, PushEvent], dict[str, Any]] | None = None,
        concurrency: int | None = None,
        *,
        emitter: AuditEmitter,
    ) -> None:
        """Initialize the manager.

        Args:
            tmp_root: Directory where per-job upload tempfiles are stored.
                Defaults to ``<system tmp>/nextext-jobs``.
            pipeline_runner: Optional override for the blocking pipeline call,
                used by tests to substitute a deterministic stub.
            concurrency: Optional override for the worker semaphore size.
                Defaults to :func:`nextext.utils.env_cfg.load_job_concurrency`
                (``NEXTEXT_JOB_CONCURRENCY``, itself defaulting to ``1``).
            emitter: The process's started audit emitter, which every job
                event is written to.
        """
        resolved = concurrency if concurrency is not None else load_job_concurrency()
        self._jobs: dict[str, JobState] = {}
        # Owner-level SSE subscribers, keyed by owner_id. A single browser
        # opens one multiplexed stream (``subscribe_owner``) that receives
        # events for every job it owns, so a large batch uses one connection
        # instead of one per job.
        self._owner_subscribers: dict[str, list[asyncio.Queue[bytes | None]]] = {}
        self._lock = asyncio.Lock()
        self._workers_semaphore = asyncio.Semaphore(resolved)
        self._concurrency = resolved
        self._background_tasks: set[asyncio.Task[None]] = set()
        self.tmp_root = tmp_root or Path(tempfile.gettempdir()) / "nextext-jobs"
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self._pipeline_runner = pipeline_runner or _run_pipeline_blocking
        self._sink = AuditSink(emitter)
        # Held while the manager runs, so a second manager on the same
        # directory (another process) never sweeps this one's live uploads.
        self._root_lock_fd: int | None = None

    async def start(self) -> None:
        """Take the job directory and remove what a previous process left in it.

        Uploads outlive a process that dies without :meth:`stop`; nothing in
        memory points at them any more. They are removed here, before the API
        serves a request, and their deletion is recorded (``delete`` /
        ``job.sweep_orphans``, by the system). When another live manager holds
        the directory, nothing is removed.

        Raises:
            AuditWriteError: If the deletion cannot be recorded; startup fails.
        """
        if not self._lock_root():
            logger.warning("Another Nextext process is using the job directory; skipped the leftover-upload sweep.")
            return
        swept = await asyncio.to_thread(self._remove_leftovers)
        self._record_removals("job.sweep_orphans", swept, trigger="startup")

    async def stop(self) -> None:
        """Remove every job and its upload, record it, and close owner streams.

        Job events are refused from here on (a pipeline thread still running
        fails its job instead of writing past the chain's end); the deletion
        of the remaining jobs is recorded as one ``delete`` / ``job.purge``
        per owner, by the system.
        """
        self._sink.seal()
        states = list(self._jobs.values())
        self._jobs.clear()
        for state in states:
            await self._delete_state(state)
        removals: list[tuple[JobRecord | None, bool]] = [
            (
                JobRecord(job_id=state.job_id, owner_id=state.owner_id, created_at=state.created_at.isoformat()),
                not state.file_path.parent.exists(),
            )
            for state in states
        ]
        try:
            self._record_removals("job.purge", removals, trigger="shutdown")
        finally:
            async with self._lock:
                for queues in self._owner_subscribers.values():
                    for queue in queues:
                        queue.put_nowait(None)
                self._owner_subscribers.clear()
            self._release_root()

    def _lock_root(self) -> bool:
        """Take the job directory's lock for the manager's lifetime.

        Returns:
            bool: ``True`` when this manager now holds it, ``False`` when
                another live manager does.
        """
        if self._root_lock_fd is not None:
            return True
        fd = os.open(self.tmp_root / ".lock", os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        self._root_lock_fd = fd
        return True

    def _release_root(self) -> None:
        """Release the job directory's lock, if held."""
        if self._root_lock_fd is not None:
            os.close(self._root_lock_fd)
            self._root_lock_fd = None

    def _remove_leftovers(self) -> list[tuple[JobRecord | None, bool]]:
        """Remove every job directory under ``tmp_root``.

        Called only while no job is live in this process and no other manager
        holds the directory, so every job directory found is a leftover.

        Returns:
            list[tuple[JobRecord | None, bool]]: Per directory, its owner
                record (``None`` when it had no valid one) and whether it is
                gone.
        """
        removed: list[tuple[JobRecord | None, bool]] = []
        for entry in sorted(self.tmp_root.iterdir()):
            if not entry.name.startswith("job-") or entry.is_symlink() or not entry.is_dir():
                continue
            record = read_job_record(entry)
            shutil.rmtree(entry, ignore_errors=True)
            removed.append((record, not entry.exists()))
        if removed:
            logger.info("Removed {} leftover job director(ies) from a previous process.", len(removed))
        return removed

    def _record_removals(
        self,
        operation: str,
        removals: list[tuple[JobRecord | None, bool]],
        *,
        trigger: str,
    ) -> None:
        """Record the system's removal of job directories, one event per owner.

        Jobs are named by id, their owner as the target's container
        (``owner:<id>``); directories without an owner record are counted on
        an event of their own (``kind="upload"``). A directory that could not
        be removed is recorded with ``outcome="error"``.

        Args:
            operation: ``job.sweep_orphans`` or ``job.purge``.
            removals: Per directory, its owner record and whether it is gone.
            trigger: ``startup`` or ``shutdown``.
        """
        system = Actor(None, "system")
        by_owner: dict[tuple[str, bool], list[str]] = {}
        unrecorded: dict[bool, int] = {}
        for record, gone in removals:
            if record is None:
                unrecorded[gone] = unrecorded.get(gone, 0) + 1
            else:
                by_owner.setdefault((record.owner_id, gone), []).append(record.job_id)
        emitter = self._sink.emitter
        for (owner_id, gone), job_ids in by_owner.items():
            emit_with_ids(
                emitter,
                "delete",
                operation,
                system,
                kind="job",
                ids=job_ids,
                container=f"owner:{owner_id}",
                outcome="ok" if gone else "error",
                criteria={"trigger": trigger},
            )
        for gone, count in unrecorded.items():
            emitter.emit(
                "delete",
                operation,
                system,
                outcome="ok" if gone else "error",
                target=Target(kind="upload", count=count),
                criteria={"trigger": trigger, "owner_recorded": False},
            )

    async def create_job(
        self,
        *,
        owner_id: str,
        file_name: str,
        file_path: Path,
        source_file_hash: str,
        options: JobOptions,
        origin: Origin,
    ) -> JobState:
        """Register a new job, record its upload, and dispatch its async worker.

        The owner record is written next to the upload and the upload's
        ``collect`` event is recorded before the job exists; if either fails,
        the upload is removed and the error propagates, so no job runs whose
        upload is unrecorded.

        Args:
            owner_id: Principal resolved by
                :func:`nextext.api.identity.resolve_principal`.
            file_name: Original upload filename (basename, no directories).
            file_path: Path to the streamed upload on disk.
            source_file_hash: ``sha256:...`` digest of the upload.
            options: Parsed pipeline options.
            origin: The creating request's actor, purpose and request id,
                which every event of the job carries.

        Returns:
            JobState: The newly registered state.
        """
        job_id = uuid.uuid4().hex
        created_at = _utcnow()
        audit = JobAudit(self._sink, job_id, origin)
        try:
            await asyncio.to_thread(
                write_job_record,
                file_path.parent,
                JobRecord(job_id=job_id, owner_id=owner_id, created_at=created_at.isoformat()),
            )
            audit.collect(_collect_criteria(file_path, source_file_hash, options))
        except BaseException:
            await asyncio.to_thread(self._unlink_job_dir, file_path)
            raise
        state = JobState(
            job_id=job_id,
            owner_id=owner_id,
            file_name=file_name,
            file_path=file_path,
            source_file_hash=source_file_hash,
            options=options,
            created_at=created_at,
            audit=audit,
        )
        async with self._lock:
            self._jobs[job_id] = state
        task = asyncio.create_task(self._worker(state))
        state.task = task
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return state

    async def get_by_media_token(self, job_id: str, token: str) -> JobState | None:
        """Return a job only if ``token`` is its media capability.

        The media route has no principal to check — a media element sends no
        identity header — so the token *is* the authorization. Compared with
        :func:`secrets.compare_digest` so a wrong guess cannot be timed, and
        an unknown job is indistinguishable from a wrong token.

        Args:
            job_id: Job identifier from the URL path.
            token: Candidate capability token from the query string.

        Returns:
            JobState | None: The job, or ``None`` for an unknown id, a deleted
                job, or a token mismatch.
        """
        async with self._lock:
            state = self._jobs.get(job_id)
        if state is None or state.deleted:
            return None
        if not secrets.compare_digest(state.media_token, token):
            return None
        return state

    async def get(self, job_id: str) -> JobState | None:
        """Return the in-memory state for a job, or ``None`` if absent.

        Args:
            job_id: Job identifier returned by ``create_job``.

        Returns:
            JobState | None: The job, or ``None`` if unknown.
        """
        async with self._lock:
            return self._jobs.get(job_id)

    async def delete(self, job_id: str, *, origin: Origin, owner_id: str | None = None) -> bool:
        """Remove a job, its tempfile, and signal any SSE subscribers to close.

        The erasure is recorded (``delete`` / ``job.delete``) once the job is
        gone from memory and its upload removed — or, for a running job, once
        its removal is handed to the worker, which removes the upload when the
        pipeline step that holds it returns.

        Args:
            job_id: Job identifier.
            origin: The deleting request's actor, purpose and request id.
            owner_id: When provided, the deletion only succeeds if the job
                belongs to ``owner_id``. Allows the routes to enforce the
                ownership boundary without re-fetching the state.

        Returns:
            bool: ``True`` if the job existed and was removed, ``False`` otherwise.
        """
        async with self._lock:
            state = self._jobs.get(job_id)
            if state is None or (owner_id is not None and state.owner_id != owner_id):
                return False
            self._jobs.pop(job_id, None)
        status_at_delete = state.status
        state.deleted = True
        task = state.task
        if task is not None and not task.done() and state.status is JobStatus.QUEUED:
            # The worker is still parked on the semaphore (or not yet
            # scheduled): cancel it before it can run the pipeline against
            # the upload we are about to remove.
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if state.status is JobStatus.RUNNING:
            # The blocking pipeline thread may still be reading the upload.
            # Close subscribers now; the worker's finally block removes the
            # tempdir once the thread has finished.
            for queue in list(state.subscribers):
                queue.put_nowait(None)
            state.subscribers.clear()
        else:
            await self._delete_state(state)
        self._sink.emit(
            "delete",
            "job.delete",
            origin.actor,
            request_id=origin.request_id,
            purpose=origin.purpose,
            target=state.audit.target,
            criteria={"status": status_at_delete.value},
        )
        return True

    async def _delete_state(self, state: JobState) -> None:
        """Tear down a single ``JobState`` (tempdir + subscribers).

        Args:
            state: The state to dispose.
        """
        for queue in list(state.subscribers):
            queue.put_nowait(None)
        state.subscribers.clear()
        await asyncio.to_thread(self._unlink_job_dir, state.file_path)

    @staticmethod
    def _unlink_job_dir(file_path: Path) -> None:
        """Remove a per-job tempdir best-effort.

        Args:
            file_path: Any path inside the per-job tempdir.
        """
        parent = file_path.parent
        if parent.exists():
            shutil.rmtree(parent, ignore_errors=True)

    def allocate_tmpdir(self, suffix: str | None = None) -> tuple[Path, str]:
        """Create a fresh per-job tempdir and return the staged upload path.

        Args:
            suffix: Optional filename suffix (e.g. ``.mp3``) preserved on the
                staged upload path.

        Returns:
            tuple[Path, str]: ``(file_path, job_dir_name)``. The file does not
                yet exist; the caller streams the upload into it.
        """
        job_dir = Path(tempfile.mkdtemp(prefix="job-", dir=self.tmp_root))
        file_path = job_dir / f"upload{suffix or ''}"
        return file_path, job_dir.name

    async def subscribe(self, state: JobState) -> AsyncIterator[bytes]:
        """Yield SSE-formatted events for a job, replaying history first.

        Args:
            state: The job to subscribe to.

        Yields:
            bytes: SSE-formatted event frames.
        """
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        async with self._lock:
            for _, frame in state.event_history:
                await queue.put(frame)
            state.subscribers.append(queue)
            already_done = state.status in (JobStatus.COMPLETED, JobStatus.FAILED)
        try:
            while True:
                try:
                    raw: bytes | None = await asyncio.wait_for(
                        queue.get(),
                        timeout=15.0,
                    )
                except TimeoutError:
                    yield b": ping\n\n"
                    continue
                if raw is None:
                    return
                frame = raw
                yield frame
                # Detect terminal frames so the connection closes cleanly.
                if frame.startswith(b"event: job_"):
                    return
                if already_done and queue.empty():
                    return
        finally:
            async with self._lock:
                if queue in state.subscribers:
                    state.subscribers.remove(queue)

    async def subscribe_owner(self, owner_id: str) -> AsyncIterator[bytes]:
        """Yield SSE events for *all* of an owner's jobs over one connection.

        Replays every owned job's history on connect (oldest job first) so a
        reloading client rebuilds each job's progression, then live-tails every
        subsequent event dispatched for the owner — including jobs created
        after the stream opened. Unlike :meth:`subscribe`, a single job
        reaching a terminal state does not close the stream; it stays open for
        the owner's other (and future) jobs until the client disconnects.

        Args:
            owner_id: Principal resolved by
                :func:`nextext.api.identity.resolve_principal`.

        Yields:
            bytes: SSE-formatted event frames, each tagged with its ``job_id``.
        """
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        async with self._lock:
            # Snapshot + register atomically: put_nowait never awaits, so no
            # concurrently dispatched frame can slip between replay and
            # registration (dispatch runs on this same loop thread).
            owned = sorted(
                (state for state in self._jobs.values() if state.owner_id == owner_id),
                key=lambda state: state.created_at,
            )
            for state in owned:
                for _, frame in state.event_history:
                    queue.put_nowait(frame)
            self._owner_subscribers.setdefault(owner_id, []).append(queue)
        try:
            while True:
                try:
                    raw: bytes | None = await asyncio.wait_for(queue.get(), timeout=15.0)
                except TimeoutError:
                    yield b": ping\n\n"
                    continue
                if raw is None:
                    return
                yield raw
        finally:
            async with self._lock:
                queues = self._owner_subscribers.get(owner_id)
                if queues is not None and queue in queues:
                    queues.remove(queue)
                if queues is not None and not queues:
                    del self._owner_subscribers[owner_id]

    async def _worker(self, state: JobState) -> None:
        """Run the pipeline for ``state`` once a worker slot is available.

        Args:
            state: The job to process.
        """
        async with self._workers_semaphore:
            state.status = JobStatus.RUNNING
            state.started_at = _utcnow()
            loop = asyncio.get_running_loop()

            def _push(event_name: str, payload: dict[str, Any]) -> None:
                """Publish an event to history + subscribers (thread-safe).

                Tags every frame with ``job_id`` at this single choke point so
                each event is self-identifying — the multiplexed owner stream
                (:meth:`subscribe_owner`) carries events for many jobs over one
                connection and routes them by ``job_id``. Terminal events
                already carry it; re-adding is idempotent.

                Args:
                    event_name: SSE event name.
                    payload: JSON-serializable payload.
                """
                if state.deleted:
                    # The owner deleted the job mid-run; don't stream events
                    # for a job that is no longer listed.
                    return
                tagged = {"job_id": state.job_id, **payload}
                frame = _format_sse(event_name, tagged)
                loop.call_soon_threadsafe(self._dispatch_event, state, event_name, tagged, frame)

            def _push_terminal(event_name: str, payload: dict[str, Any]) -> None:
                """Publish a terminal event synchronously, on the loop thread.

                ``_push`` defers dispatch to a later loop step. For a terminal
                event that would leave a window where ``state.status`` already
                reads finished but history lacks the terminal frame, and
                :meth:`subscribe` closes a finished job's stream once history
                is drained — so a subscriber attaching in that window lost the
                frame. Dispatching in the same step as the status flip closes
                it. Order is kept: every frame the pipeline thread pushed is
                dispatched before the worker resumes from ``to_thread``.

                Args:
                    event_name: Terminal SSE event name.
                    payload: JSON-serializable payload.
                """
                if state.deleted:
                    return
                tagged = {"job_id": state.job_id, **payload}
                self._dispatch_event(state, event_name, tagged, _format_sse(event_name, tagged))

            try:
                # The creating request's id rides along on every inference
                # request the job makes (X-Request-ID), for correlation.
                with correlate(state.audit.origin.request_id):
                    result = await asyncio.to_thread(self._pipeline_runner, state, _push)
                if not state.deleted:
                    # A job deleted mid-run discards its result instead of
                    # storing the joined analyses; its transfers are recorded.
                    analyses, speaker_attributed = _joined_analyses(result, state.options)
                    state.audit.combine(analyses, speaker_attributed=speaker_attributed)
                state.result = result
                state.finished_at = _utcnow()
                state.status = JobStatus.COMPLETED
                state.progress = 1.0
                state.stage = None
                skipped = bool(result.get("skipped", False))
                skip_code = result.get("skip_reason_code")
                if skipped:
                    record_skipped(skip_code)
                else:
                    record_completed()
                _push_terminal(
                    "job_completed",
                    {
                        "job_id": state.job_id,
                        "skipped": skipped,
                        "skip_reason_code": skip_code,
                        "timestamp": _utcnow().isoformat(),
                    },
                )
            except Exception as exc:
                # An undecodable upload is the user's to fix, so it earns a
                # typed code and a warning; everything else is ours and keeps
                # its traceback frames, but not its message, which can quote
                # an upstream reply. Neither leaks detail past ``error``.
                if isinstance(exc, AudioDecodeError):
                    failure_code: FailureCode = "undecodable_media"
                    logger.warning("Job {} failed: media could not be decoded ({}).", state.job_id, exc)
                else:
                    failure_code = "internal"
                    logger.error("Job {} failed: {}\n{}", state.job_id, error_label(exc), exception_frames(exc))
                record_failed(failure_code)
                state.status = JobStatus.FAILED
                state.error = "Job failed."
                state.error_code = failure_code
                state.finished_at = _utcnow()
                _push_terminal(
                    "job_failed",
                    {
                        "job_id": state.job_id,
                        "error": "Job failed.",
                        "error_code": failure_code,
                        "timestamp": _utcnow().isoformat(),
                    },
                )
            finally:
                if state.deleted:
                    # delete() ran while the pipeline thread was still using
                    # the upload and deferred the tempdir cleanup to us.
                    await asyncio.to_thread(self._unlink_job_dir, state.file_path)

    def _dispatch_event(
        self,
        state: JobState,
        event_name: str,
        payload: dict[str, Any],
        frame: bytes,
    ) -> None:
        """Record an event in history and fan it out to subscribers.

        Runs on the event loop thread (scheduled via ``call_soon_threadsafe``).

        Args:
            state: The job emitting the event.
            event_name: SSE event name.
            payload: JSON-serializable payload (used to update job state).
            frame: Pre-rendered SSE frame.
        """
        state.event_history.append((event_name, frame))
        stage_index = payload.get("stage_index")
        if isinstance(stage_index, int):
            state.stage_index = stage_index
        stage = payload.get("stage")
        if isinstance(stage, str):
            state.stage = stage
        progress = payload.get("progress")
        if isinstance(progress, (int, float)):
            state.progress = float(progress)
        for queue in state.subscribers:
            queue.put_nowait(frame)
        # Owner-multiplexed subscribers receive the same frame (terminal frames
        # included) but never the ``None`` sentinel: one job finishing must not
        # close a stream that also serves the owner's other and future jobs.
        for queue in self._owner_subscribers.get(state.owner_id, ()):
            queue.put_nowait(frame)
        if event_name in _TERMINAL_EVENT_NAMES:
            for queue in list(state.subscribers):
                queue.put_nowait(None)

    async def list_for_owner(self, owner_id: str) -> list[JobState]:
        """Return all in-memory jobs owned by ``owner_id``, newest first.

        Powers the frontend's reload re-discovery: the browser carries a
        stable identity in its URL (``?owner=<id>``), so after a reload it
        lists its own jobs and re-attaches to any still running while
        re-rendering those already finished.

        Args:
            owner_id: Principal resolved by
                :func:`nextext.api.identity.resolve_principal`.

        Returns:
            list[JobState]: In-memory states whose owner matches, newest
                first.
        """
        async with self._lock:
            states = [state for state in self._jobs.values() if state.owner_id == owner_id]
        states.sort(key=lambda s: s.created_at, reverse=True)
        return states
