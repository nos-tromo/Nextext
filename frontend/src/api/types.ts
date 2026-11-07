// Mirrors nextext/api/schemas.py. Keep field names identical to the JSON.

export type JobStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'interrupted'

export type Task = 'transcribe' | 'translate'

export interface JobOptions {
  src_lang: string | null
  trg_lang: string
  task: Task
  diarize: boolean
  words: boolean
  summarization: boolean
  hate_speech: boolean
  /** Sample video keyframes and describe them. Independent of `summarization`. */
  keyframes: boolean
  /** Find the faces in the video. Refused (422) while the operator switch is off. */
  faces: boolean
}

export interface JobCreateResponse {
  job_id: string
  status: JobStatus
  created_at: string
}

export interface TranscriptSegment {
  start: string | null
  end: string | null
  /** Numeric twin of `start`, for seeking the player. Null when unparseable. */
  start_seconds: number | null
  /** Numeric twin of `end`. */
  end_seconds: number | null
  speaker: string | null
  text: string
  translation: string | null
}

export interface WordCount {
  word: string
  count: number
}

export interface NamedEntity {
  entity: string
  category: string
  frequency: number
}

export interface HateSpeechFinding {
  hate_speech: boolean
  category: string
  confidence: 'high' | 'medium' | 'low'
  reason: string
  /** Original transcript wording (never the translation). */
  text: string
  start: string | null
  /** Diarization label; null for undiarized transcripts. */
  speaker: string | null
  /** Translation shown to the classifier as an aid; null when the job did not translate. */
  translation: string | null
  /** A transcript segment, or a keyframe caption (whose `start` is the frame's time); absent reads as transcript. */
  source?: 'transcript' | 'frame'
}

/** Why a completed job produced no transcript. Mirrors nextext/core/outcomes.py. */
export type SkipReason = 'vad_no_speech' | 'asr_empty_transcript' | 'asr_all_segments_filtered'

/** Why a job failed. `internal` is everything the user cannot act on. */
export type FailureCode = 'undecodable_media' | 'internal'

/** One model-written description of a sampled video keyframe. */
export interface FrameCaption {
  /** Seconds from the start of the clip. */
  time_sec: number
  caption: string
}

/** One span of time a face entry was on screen, in seconds. */
export interface FaceAppearance {
  start_seconds: number
  end_seconds: number
}

/** One person (grouped by the faces service) or one face track (ungrouped). */
export interface FaceEntry {
  /** Position by first appearance, from 1. */
  number: number
  /** Grouped as a person ("Person N"); otherwise a single track ("Face N"). */
  assigned: boolean
  first_seen_seconds: number
  /** Time of the last detection. */
  last_seen_seconds: number
  appearances: FaceAppearance[]
  /** How many sampled frames it was found in. */
  detections: number
  /** A small JPEG as a `data:` URI. */
  thumbnail: string
  /** The artifact holding the best shot, `face_NNN.jpg`. */
  best_artifact: string
}

export interface FacesResult {
  /** `partial`: detection failed for part of the video; `unavailable`: it never answered. */
  outcome: 'ok' | 'partial' | 'unavailable'
  grouped: boolean
  /** Some faces were dropped by a cap; the most prominent are shown. */
  truncated: boolean
  entries: FaceEntry[]
}

export interface JobResult {
  transcript: TranscriptSegment[]
  transcript_language: string | null
  resolved_src_lang: string | null
  summary: string | null
  word_counts: WordCount[] | null
  named_entities: NamedEntity[] | null
  wordcloud_url: string | null
  keyframes_url: string | null
  /** Capability URL for playing the original upload; null once it is gone. */
  media_url: string | null
  /** Descriptions of the sampled keyframes; null unless the job asked for them. */
  frame_captions: FrameCaption[] | null
  hate_speech_findings: HateSpeechFinding[] | null
  /** The faces of the video; null unless the job asked for them and it had frames. */
  faces: FacesResult | null
  skipped: boolean
  /** Backend English prose. Never rendered — the SPA localizes the code below. */
  skip_reason: string | null
  skip_reason_code: SkipReason | null
  task: Task
}

export interface JobSnapshot {
  job_id: string
  status: JobStatus
  file_name: string
  source_file_hash: string | null
  options: JobOptions
  stage: string | null
  stage_index: number
  progress: number
  error: string | null
  error_code: FailureCode | null
  skipped: boolean
  skip_reason_code: SkipReason | null
  created_at: string
  started_at: string | null
  finished_at: string | null
  result: JobResult | null
}

export interface JobListItem {
  job_id: string
  status: JobStatus
  file_name: string
  stage: string | null
  progress: number
  error: string | null
  error_code: FailureCode | null
  /** Carried on the list too: after a reload this is the SPA's only source. */
  skipped: boolean
  skip_reason_code: SkipReason | null
  created_at: string
  started_at: string | null
  finished_at: string | null
  task: Task
}

export interface JobListResponse {
  jobs: JobListItem[]
}

// SSE event payload (event name carried by the SSE `event:` line).
export interface StageEventData {
  stage: string
  stage_index: number
  progress: number
  timestamp: string
  message: string | null
  result_delta: Record<string, unknown> | null
}

export interface StageStartedEvent {
  job_id: string
  stage: string
  stage_index: number
  progress: number
  timestamp: string
}

export interface StageCompletedEvent {
  job_id: string
  stage: string
  stage_index: number
  progress: number
  timestamp: string
  result_delta: Record<string, unknown> | null
}

export interface JobCompletedEvent {
  job_id: string
  skipped: boolean
  skip_reason_code: SkipReason | null
  timestamp: string
}

export interface JobFailedEvent {
  job_id: string
  error: string
  error_code: FailureCode | null
  timestamp: string
}

export interface JobCancelledEvent {
  job_id: string
  timestamp: string
}

export type JobEvent =
  | { name: 'stage_started'; data: StageStartedEvent }
  | { name: 'stage_completed'; data: StageCompletedEvent }
  | { name: 'job_completed'; data: JobCompletedEvent }
  | { name: 'job_failed'; data: JobFailedEvent }
  | { name: 'job_cancelled'; data: JobCancelledEvent }

export type JobEventName = JobEvent['name']

export interface HealthResponse {
  status: 'ok'
  inference: boolean
  version: string
}

export interface LanguageEntry {
  code: string
  name: string
}

export interface LanguagesResponse {
  whisper: LanguageEntry[]
  target: LanguageEntry[]
  default_target: string
}

/** `POST /jobs/{id}/media-grant`: a URL to play the upload with, and when it stops working. */
export interface MediaGrantResponse {
  /** Server-relative media URL carrying the grant (`…/media?grant=…`). */
  media_url: string
  /** ISO timestamp; the next range request after it fails. */
  expires_at: string
}

export interface AppConfig {
  language: 'en' | 'de'
  /** Audit actions that need a stated purpose (`AUDIT_REQUIRE_PURPOSE`), sorted. */
  require_purpose: string[]
  /** The operator switch `NEXTEXT_FACES`: the upload form offers faces unless `off`. */
  faces: 'off' | 'detect' | 'group'
}

export interface WhoamiResponse {
  username: string
  display_name: string | null
}
