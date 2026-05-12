"""Job state, manager, and worker for the Nextext FastAPI backend.

The :class:`JobManager` owns an in-memory dictionary of :class:`JobState`
instances. Jobs are processed by a single async worker constrained to
serial execution via an ``asyncio.Semaphore(1)``: the GPU model registry
(``nextext.utils.model_registry``) is thread-safe but contention from
parallel pipeline runs in one process would oversubscribe the GPU, so
the safe choice is one job in flight per backend container.

The blocking pipeline runs on a worker thread via
``anyio.to_thread.run_sync``. Stage transitions are published to a per-job
list of subscriber queues so any number of SSE clients can attach to the
same job and receive the same events.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
from loguru import logger
from matplotlib.figure import Figure

from nextext.api.schemas import (
    HateSpeechFinding,
    JobOptions,
    JobResult,
    JobSnapshot,
    JobStatus,
    NamedEntity,
    TranscriptSegment,
    WordCount,
)


_TERMINAL_EVENT_NAMES: frozenset[str] = frozenset(
    {"job_completed", "job_failed", "job_cancelled"}
)


PIPELINE_STAGE_LABELS: tuple[str, ...] = (
    "Transcribing",
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
    return datetime.now(tz=timezone.utc)


def _format_sse(event_name: str, payload: dict[str, Any]) -> bytes:
    """Render a payload as an SSE event frame.

    Args:
        event_name: SSE event name (e.g. ``stage_started``).
        payload: JSON-serializable payload.

    Returns:
        bytes: UTF-8 encoded frame ready to send over ``text/event-stream``.
    """
    body = json.dumps(payload, default=str)
    return f"event: {event_name}\ndata: {body}\n\n".encode("utf-8")


@dataclass
class JobState:
    """Mutable state for one in-flight or completed job."""

    job_id: str
    file_name: str
    file_path: Path
    source_file_hash: str
    options: JobOptions
    status: JobStatus = JobStatus.QUEUED
    stage: str | None = None
    stage_index: int = 0
    progress: float = 0.0
    error: str | None = None
    created_at: datetime = field(default_factory=_utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: dict[str, Any] = field(default_factory=dict)
    event_history: list[tuple[str, bytes]] = field(default_factory=list)
    subscribers: list[asyncio.Queue[bytes | None]] = field(default_factory=list)
    archive_cache: bytes | None = None

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
            created_at=self.created_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
            result=_serialize_result(self.result)
            if self.status == JobStatus.COMPLETED
            else None,
        )


def _normalize_transcript_row(row: dict[str, Any]) -> TranscriptSegment:
    """Convert a transcript DataFrame row to ``TranscriptSegment``.

    Args:
        row: One row from the transcript DataFrame.

    Returns:
        TranscriptSegment: JSON-friendly view of the row.
    """
    return TranscriptSegment(
        start=_optional_str(row.get("start")),
        end=_optional_str(row.get("end")),
        speaker=_optional_str(row.get("speaker")),
        text=str(row.get("text", "")),
    )


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
    return [
        WordCount(word=str(row[word_col]), count=int(row[count_col]))
        for _, row in df.iterrows()
    ]


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


def _normalize_hate_speech(items: list[dict] | None) -> list[HateSpeechFinding] | None:
    """Coerce hate-speech findings into ``HateSpeechFinding`` instances.

    Args:
        items: Raw list of detection dicts from ``hate_speech_pipeline``.

    Returns:
        list[HateSpeechFinding] | None: Validated findings or ``None``.
    """
    if not items:
        return None
    return [HateSpeechFinding(**item) for item in items]


def _serialize_result(result: dict[str, Any]) -> JobResult:
    """Convert an in-memory result dict into the JSON-friendly schema.

    The wordcloud Figure is intentionally not serialized into the JSON
    response — clients download it on demand via the artifacts endpoint.

    Args:
        result: In-memory result payload populated by the worker.

    Returns:
        JobResult: Pydantic representation suitable for HTTP responses.
    """
    transcript_df = result.get("transcript")
    transcript: list[TranscriptSegment] = []
    if isinstance(transcript_df, pd.DataFrame) and not transcript_df.empty:
        transcript = [
            _normalize_transcript_row(row)
            for row in transcript_df.to_dict(orient="records")
        ]

    wordcloud_url: str | None = None
    if isinstance(result.get("wordcloud"), Figure):
        wordcloud_url = result.get("_wordcloud_url")

    return JobResult(
        transcript=transcript,
        transcript_language=result.get("transcript_language"),
        resolved_src_lang=result.get("resolved_src_lang"),
        summary=result.get("summary"),
        word_counts=_normalize_word_counts(result.get("word_counts")),
        named_entities=_normalize_named_entities(result.get("named_entities")),
        wordcloud_url=wordcloud_url,
        hate_speech_findings=_normalize_hate_speech(result.get("hate_speech_findings")),
        skipped=bool(result.get("skipped", False)),
        skip_reason=result.get("skip_reason"),
        task=result.get("task", "transcribe"),
    )


PushEvent = Callable[[str, dict[str, Any]], None]


def _run_pipeline_blocking(state: JobState, push_event: PushEvent) -> dict[str, Any]:
    """Run the full Nextext pipeline for one job, emitting SSE events.

    Ported from :func:`nextext.app._run_pipeline`; instead of writing into
    a Streamlit ``stage_container``, this version publishes stage transitions
    through ``push_event`` so SSE subscribers see the same progression.

    Args:
        state: The job to process.
        push_event: Thread-safe callable for emitting SSE events.

    Returns:
        dict[str, Any]: In-memory result payload mirroring the Streamlit shape.

    Raises:
        ConnectionError: If the configured inference provider is unreachable.
    """
    # Local imports keep test stubs and import-time module reloads predictable.
    from nextext.core.docint_transcript import language_name
    from nextext.core.openai_cfg import InferencePipeline
    from nextext.pipeline import (
        hate_speech_pipeline,
        normalize_language_code,
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
        "speakers": opts.speakers,
        "words": opts.words,
        "summarization": opts.summarization,
        "hate_speech": opts.hate_speech,
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

    # Transcription -----------------------------------------------------------
    _notify(0)
    df, updated_src_lang = transcription_pipeline(
        file_path=state.file_path,
        trg_lang=file_opts["trg_lang"],
        src_lang=file_opts["src_lang"] or "",
        task=file_opts["task"],
        n_speakers=file_opts["speakers"],
    )
    file_opts["src_lang"] = updated_src_lang

    transcript_text = " ".join(df["text"].astype(str).tolist()).strip()
    if df.empty or not transcript_text:
        transcript_language = file_opts[
            "trg_lang" if file_opts["task"] == "translate" else "src_lang"
        ]
        payload = {
            "transcript": df,
            "summary": None,
            "word_counts": None,
            "named_entities": None,
            "wordcloud": None,
            "hate_speech_findings": None,
            "resolved_src_lang": file_opts["src_lang"],
            "transcript_language": transcript_language,
            "skipped": True,
            "skip_reason": "No speech detected in audio file.",
            "task": file_opts["task"],
        }
        _complete(0, {"transcript_segments": 0, "skipped": True})
        return payload

    _complete(
        0,
        {
            "transcript_segments": int(len(df)),
            "resolved_src_lang": file_opts["src_lang"],
        },
    )

    # Translation -------------------------------------------------------------
    _notify(1)
    inference_pipeline: InferencePipeline | None = None
    if file_opts["task"] == "translate" and file_opts["trg_lang"] != "en":
        inference_pipeline = InferencePipeline(
            out_language=language_name(file_opts["trg_lang"])
        )
        if not inference_pipeline.get_health():
            raise ConnectionError(
                "The configured inference provider is not reachable. "
                "Please ensure it is running and accessible."
            )
        df = translation_pipeline(
            df,
            file_opts["trg_lang"],
            src_lang=file_opts["src_lang"],
            inference_pipeline=inference_pipeline,
        )
        _complete(1, {"translated": True})
    else:
        _complete(1, {"translated": False})

    transcript_language = file_opts[
        "trg_lang" if file_opts["task"] == "translate" else "src_lang"
    ]
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
    }

    # Word-level analysis -----------------------------------------------------
    _notify(2)
    if file_opts["words"]:
        wc, ner, cloud = wordlevel_pipeline(
            df,
            normalize_language_code(transcript_language) or "en",
        )
        result["word_counts"] = wc
        result["named_entities"] = ner
        result["wordcloud"] = cloud
        result["_wordcloud_url"] = (
            f"/api/v1/jobs/{state.job_id}/artifacts/wordcloud.png"
        )
        _complete(
            2,
            {
                "word_counts": int(len(wc)) if wc is not None else 0,
                "named_entities": int(len(ner)) if ner is not None else 0,
                "wordcloud": cloud is not None,
            },
        )
    else:
        _complete(2, {"skipped": True})

    # Summarization -----------------------------------------------------------
    _notify(3)
    if file_opts["summarization"]:
        if inference_pipeline is None:
            inference_pipeline = InferencePipeline(
                out_language=language_name(transcript_language)
            )
            if not inference_pipeline.get_health():
                raise ConnectionError(
                    "The configured inference provider is not reachable. "
                    "Please ensure it is running and accessible."
                )
        result["summary"] = summarization_pipeline(
            " ".join(df["text"].astype(str).tolist()),
            inference_pipeline=inference_pipeline,
        )
        _complete(3, {"summary": bool(result["summary"])})
    else:
        _complete(3, {"skipped": True})

    # Hate-speech detection ---------------------------------------------------
    _notify(4)
    if file_opts["hate_speech"]:
        if inference_pipeline is None:
            inference_pipeline = InferencePipeline(
                out_language=language_name(transcript_language)
            )
            if not inference_pipeline.get_health():
                raise ConnectionError(
                    "The configured inference provider is not reachable. "
                    "Please ensure it is running and accessible."
                )
        findings = hate_speech_pipeline(df=df, inference_pipeline=inference_pipeline)
        result["hate_speech_findings"] = findings
        _complete(4, {"flagged": len(findings)})
    else:
        _complete(4, {"skipped": True})

    return result


class JobManager:
    """In-memory job store and worker dispatcher."""

    def __init__(
        self,
        tmp_root: Path | None = None,
        ttl_seconds: int = 3600,
        pipeline_runner: Callable[[JobState, PushEvent], dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the manager.

        Args:
            tmp_root: Directory where per-job upload tempfiles are stored.
                Defaults to ``<system tmp>/nextext-jobs``.
            ttl_seconds: Lifetime in seconds for completed jobs before the
                sweeper evicts them.
            pipeline_runner: Optional override for the blocking pipeline call,
                used by tests to substitute a deterministic stub.
        """
        self._jobs: dict[str, JobState] = {}
        self._lock = asyncio.Lock()
        self._workers_semaphore = asyncio.Semaphore(1)
        self._sweeper_task: asyncio.Task[None] | None = None
        self.tmp_root = tmp_root or Path(tempfile.gettempdir()) / "nextext-jobs"
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self._pipeline_runner = pipeline_runner or _run_pipeline_blocking

    async def start(self) -> None:
        """Launch the background sweeper task on app startup."""
        if self._sweeper_task is None:
            self._sweeper_task = asyncio.create_task(self._sweeper_loop())

    async def stop(self) -> None:
        """Cancel the sweeper and clean up tempdirs on shutdown."""
        if self._sweeper_task is not None:
            self._sweeper_task.cancel()
            try:
                await self._sweeper_task
            except asyncio.CancelledError:
                pass
            self._sweeper_task = None
        for state in list(self._jobs.values()):
            await self._delete_state(state)

    async def create_job(
        self,
        file_name: str,
        file_path: Path,
        source_file_hash: str,
        options: JobOptions,
    ) -> JobState:
        """Register a new job and dispatch its async worker.

        Args:
            file_name: Original upload filename (basename, no directories).
            file_path: Path to the streamed upload on disk.
            source_file_hash: ``sha256:...`` digest of the upload.
            options: Parsed pipeline options.

        Returns:
            JobState: The newly registered state.
        """
        job_id = uuid.uuid4().hex
        state = JobState(
            job_id=job_id,
            file_name=file_name,
            file_path=file_path,
            source_file_hash=source_file_hash,
            options=options,
        )
        async with self._lock:
            self._jobs[job_id] = state
        asyncio.create_task(self._worker(state))
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

    async def delete(self, job_id: str) -> bool:
        """Remove a job, its tempfile, and signal any SSE subscribers to close.

        Args:
            job_id: Job identifier.

        Returns:
            bool: ``True`` if the job existed and was removed, ``False`` otherwise.
        """
        async with self._lock:
            state = self._jobs.pop(job_id, None)
        if state is None:
            return False
        await self._delete_state(state)
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
                        queue.get(),  # type: ignore[arg-type]
                        timeout=15.0,
                    )
                except asyncio.TimeoutError:
                    yield b": ping\n\n"
                    continue
                frame = raw
                if frame is None:
                    return
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

                Args:
                    event_name: SSE event name.
                    payload: JSON-serializable payload.
                """
                frame = _format_sse(event_name, payload)
                loop.call_soon_threadsafe(
                    self._dispatch_event, state, event_name, payload, frame
                )

            try:
                result = await asyncio.to_thread(self._pipeline_runner, state, _push)
                state.result = result
                state.status = JobStatus.COMPLETED
                state.progress = 1.0
                state.stage = None
                state.finished_at = _utcnow()
                _push(
                    "job_completed",
                    {
                        "job_id": state.job_id,
                        "skipped": bool(result.get("skipped", False)),
                        "timestamp": _utcnow().isoformat(),
                    },
                )
            except Exception as exc:  # noqa: BLE001 - we want any failure mode
                logger.exception("Job {} failed.", state.job_id)
                state.status = JobStatus.FAILED
                state.error = str(exc)
                state.finished_at = _utcnow()
                _push(
                    "job_failed",
                    {
                        "job_id": state.job_id,
                        "error": str(exc),
                        "timestamp": _utcnow().isoformat(),
                    },
                )
            finally:
                # Reclaim GPU resources between jobs (mirrors CLI's finally
                # block — see ``nextext/cli.py``).
                try:
                    from nextext.utils.model_registry import flush_gpu

                    await asyncio.to_thread(flush_gpu)
                except Exception:  # pragma: no cover - defensive
                    logger.exception("flush_gpu raised during job teardown.")

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
        if event_name in _TERMINAL_EVENT_NAMES:
            for queue in list(state.subscribers):
                queue.put_nowait(None)

    async def _sweeper_loop(self) -> None:
        """Evict completed jobs older than ``ttl_seconds`` every 5 minutes."""
        try:
            while True:
                await asyncio.sleep(300)
                cutoff_ts = _utcnow().timestamp() - self.ttl_seconds
                async with self._lock:
                    stale = [
                        state
                        for state in self._jobs.values()
                        if state.finished_at is not None
                        and state.finished_at.timestamp() < cutoff_ts
                        and not state.subscribers
                    ]
                    for state in stale:
                        self._jobs.pop(state.job_id, None)
                for state in stale:
                    await self._delete_state(state)
                if stale:
                    logger.info("Evicted {} stale job(s).", len(stale))
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            raise
