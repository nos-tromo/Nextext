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
}

export interface JobCreateResponse {
  job_id: string
  status: JobStatus
  created_at: string
}

export interface TranscriptSegment {
  start: string | null
  end: string | null
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
  text: string
  start: string | null
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
  hate_speech_findings: HateSpeechFinding[] | null
  skipped: boolean
  skip_reason: string | null
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
  timestamp: string
}

export interface JobFailedEvent {
  job_id: string
  error: string
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
