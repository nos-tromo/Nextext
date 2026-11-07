"""Pydantic schemas for the Nextext FastAPI backend."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from nextext.core.outcomes import FailureCode, SkipReason
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
    # Sampling of video keyframes, independent of every other option: a summary
    # no longer implies visual context, and visual context no longer implies a
    # summary. The two rates below only bite when this is on.
    keyframes: bool = False
    # Captioning of those frames, which is the expensive half — one vision
    # request per frame. Split out because a client can want the frames without
    # the words: docint samples keyframes to run its own image pipeline over
    # them and never reads ``frame_captions``, so it was paying for a
    # description per frame that it discarded. Defaults true, so the SPA and
    # every existing client keep today's behaviour by omitting the field
    # (``extra="forbid"`` means they cannot send it).
    visual_context: bool = True
    keyframes_per_minute: int = Field(default_factory=lambda: load_keyframe_defaults().per_minute, ge=0)
    keyframes_max: int = Field(default_factory=lambda: load_keyframe_defaults().max_frames, ge=0, le=200)
    # The faces of the video (who is on screen when). Refused with 422 while
    # the operator switch ``NEXTEXT_FACES`` is off.
    faces: bool = False


class JobCreateResponse(BaseModel):
    """Response body returned by ``POST /jobs``."""

    job_id: str
    status: JobStatus
    created_at: datetime


class TranscriptSegment(BaseModel):
    """One row of the transcript DataFrame, serialized to JSON."""

    start: str | None = None
    end: str | None = None
    # Numeric twins of the display strings above, so a media player can seek
    # without re-implementing the timedelta parser in the browser. ``None``
    # when the stored timestamp is absent or unparseable.
    start_seconds: float | None = None
    end_seconds: float | None = None
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
    """One endorsed-hate finding, from a transcript segment or a keyframe caption.

    ``source`` says which: ``transcript`` findings come from
    ``hate_speech_pipeline``, where ``text`` is always the original transcript
    wording, ``translation`` carries the translation shown to the classifier
    as an aid when the job translated, and ``speaker`` is set for diarized
    transcripts; ``frame`` findings come from ``frame_hate_speech_pipeline``,
    where ``text`` is the caption and ``start`` the frame's time.
    """

    hate_speech: bool
    category: str
    confidence: Literal["high", "medium", "low"]
    reason: str
    text: str
    start: str | None = None
    speaker: str | None = None
    translation: str | None = None
    source: Literal["transcript", "frame"] = "transcript"


class FrameCaptionOut(BaseModel):
    """One timestamped description of a sampled video keyframe."""

    time_sec: float
    caption: str


class FaceAppearanceOut(BaseModel):
    """One span of time a face entry was on screen."""

    start_seconds: float
    end_seconds: float


class FaceEntryOut(BaseModel):
    """One person (grouped) or one face track (ungrouped) of a video."""

    number: int
    # True when the faces service grouped it as a person ("Person N");
    # otherwise a single track ("Face N").
    assigned: bool
    first_seen_seconds: float
    last_seen_seconds: float
    appearances: list[FaceAppearanceOut]
    detections: int
    # A small JPEG as a ``data:`` URI, so looking at the faces is the one
    # snapshot query rather than one disclosure per image.
    thumbnail: str
    # The artifact holding the entry's best shot, ``face_NNN.jpg``.
    best_artifact: str


class FacesOut(BaseModel):
    """The faces of a video, as the finished snapshot carries them."""

    outcome: Literal["ok", "partial", "unavailable"]
    grouped: bool
    truncated: bool
    entries: list[FaceEntryOut]


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
    # Capability URL for the original upload: a media element cannot send the
    # trusted identity header, so the token in the URL is what authorizes it.
    # ``None`` once the bytes are gone (deleted job / restarted backend).
    media_url: str | None = None
    frame_captions: list[FrameCaptionOut] | None = None
    hate_speech_findings: list[HateSpeechFinding] | None = None
    # ``None`` when the job did not ask for faces or the file had no frames.
    faces: FacesOut | None = None
    skipped: bool = False
    # Human-readable fallback prose; the SPA localizes ``skip_reason_code``
    # instead and never renders this string.
    skip_reason: str | None = None
    skip_reason_code: SkipReason | None = None
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
    # Typed cause of a failure, so clients can distinguish a user-fixable
    # upload problem from an outage without ever seeing the raw detail.
    error_code: FailureCode | None = None
    # Lifted out of ``result`` so a client can tell a job produced nothing
    # without fetching or parsing the whole result payload.
    skipped: bool = False
    skip_reason_code: SkipReason | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: JobResult | None = None


class MediaGrantResponse(BaseModel):
    """Response body of ``POST /jobs/{id}/media-grant``."""

    # Server-relative URL of the upload, carrying the grant in its query
    # string: what a media element's ``src`` is set to.
    media_url: str
    # When the grant stops working; ask for a new one after that.
    expires_at: datetime


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


class ConfigResponse(BaseModel):
    """Client bootstrap config for the SPA."""

    language: str
    # The audit actions that need a stated purpose (``AUDIT_REQUIRE_PURPOSE``),
    # sorted: the SPA's purpose bar warns while none is stated when any is
    # listed, and refuses to save a blank one.
    require_purpose: list[str] = Field(default_factory=list)
    # The operator switch ``NEXTEXT_FACES``: the SPA offers faces unless off.
    faces: Literal["off", "detect", "group"] = "off"


class WhoamiResponse(BaseModel):
    """The resolved calling identity, for the SPA's AppHeader."""

    username: str
    # Decorative only — the edge gateway's Authelia displayname, injected
    # alongside (not instead of) X-Auth-User. Never part of principal
    # resolution/identity; None when the gateway isn't in front (dev).
    display_name: str | None = None


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
    error_code: FailureCode | None = None
    # Carried here too: after a browser reload the list is the SPA's only
    # source, so without these a skipped job would read as a plain "done".
    skipped: bool = False
    skip_reason_code: SkipReason | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    task: Literal["transcribe", "translate"] = "transcribe"


class JobListResponse(BaseModel):
    """Response body for ``GET /jobs``."""

    jobs: list[JobListItem]
