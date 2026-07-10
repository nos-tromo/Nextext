"""Pydantic schemas for the Nextext FastAPI backend."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from nextext.utils.env_cfg import load_keyframe_defaults


class JobStatus(StrEnum):
    """Lifecycle states for a backend processing job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class JobOptions(BaseModel):
    """Per-job pipeline options mirroring the Streamlit ``opts`` dict."""

    model_config = ConfigDict(extra="forbid")

    src_lang: str | None = None
    trg_lang: str = "de"
    task: Literal["transcribe", "translate"] = "transcribe"
    diarize: bool = True
    words: bool = False
    summarization: bool = False
    hate_speech: bool = False
    keyframes_per_minute: int = Field(default_factory=lambda: load_keyframe_defaults().per_minute, ge=0)
    keyframes_max: int = Field(default_factory=lambda: load_keyframe_defaults().max_frames, ge=0, le=200)


class JobCreateResponse(BaseModel):
    """Response body returned by ``POST /jobs``."""

    job_id: str
    status: JobStatus
    created_at: datetime


class TranscriptSegment(BaseModel):
    """One row of the transcript DataFrame, serialized to JSON."""

    start: str | None = None
    end: str | None = None
    speaker: str | None = None
    text: str
    translation: str | None = None


class WordCount(BaseModel):
    """One row of the word-counts DataFrame."""

    word: str
    count: int


class NamedEntity(BaseModel):
    """One row of the named-entities DataFrame."""

    entity: str
    category: str
    frequency: int


class HateSpeechFinding(BaseModel):
    """One flagged segment returned by ``hate_speech_pipeline``."""

    hate_speech: bool
    category: str
    confidence: Literal["high", "medium", "low"]
    reason: str
    text: str
    start: str | None = None


class JobResult(BaseModel):
    """Aggregate of all per-stage outputs for a single completed job."""

    transcript: list[TranscriptSegment] = Field(default_factory=list)
    transcript_language: str | None = None
    resolved_src_lang: str | None = None
    summary: str | None = None
    word_counts: list[WordCount] | None = None
    named_entities: list[NamedEntity] | None = None
    wordcloud_url: str | None = None
    keyframes_url: str | None = None
    hate_speech_findings: list[HateSpeechFinding] | None = None
    skipped: bool = False
    skip_reason: str | None = None
    task: Literal["transcribe", "translate"] = "transcribe"


class JobSnapshot(BaseModel):
    """Point-in-time view of a job, returned by ``GET /jobs/{id}``."""

    job_id: str
    status: JobStatus
    file_name: str
    source_file_hash: str | None = None
    options: JobOptions
    stage: str | None = None
    stage_index: int = 0
    progress: float = 0.0
    error: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: JobResult | None = None


class StageEvent(BaseModel):
    """SSE event payload describing a pipeline stage transition."""

    job_id: str
    stage: str
    stage_index: int
    progress: float
    timestamp: datetime
    message: str | None = None
    result_delta: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    """Response body for ``GET /health``."""

    status: Literal["ok"] = "ok"
    inference: bool
    version: str


class VersionResponse(BaseModel):
    """App release version."""

    version: str


class LanguageEntry(BaseModel):
    """One language entry returned by ``GET /languages``."""

    code: str
    name: str


class LanguagesResponse(BaseModel):
    """Response body for ``GET /languages``."""

    whisper: list[LanguageEntry]
    target: list[LanguageEntry]
    default_target: str


class JobListItem(BaseModel):
    """Compact view of an in-memory job for the listing endpoint."""

    job_id: str
    status: JobStatus
    file_name: str
    stage: str | None = None
    progress: float = 0.0
    error: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    task: Literal["transcribe", "translate"] = "transcribe"


class JobListResponse(BaseModel):
    """Response body for ``GET /jobs``."""

    jobs: list[JobListItem]
