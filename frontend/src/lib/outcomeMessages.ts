import type { FailureCode, SkipReason } from '../api/types'
import type { Strings } from '../i18n'

/**
 * Map backend outcome codes to catalog keys.
 *
 * The backend's own `skip_reason` / `error` strings are English prose written
 * for logs and non-UI consumers; rendering them would leak untranslated text
 * into a German UI (and, for failures, risk leaking detail). The typed code is
 * the contract — these mappers turn it into a key the catalog translates.
 *
 * Unknown codes fall back to the generic message rather than rendering the
 * raw code, so a frontend running against a newer backend degrades quietly.
 */

const JOB_SKIP_KEYS: Record<SkipReason, keyof Strings> = {
  vad_no_speech: 'jobs.skipped_vad_no_speech',
  asr_empty_transcript: 'jobs.skipped_asr_empty',
  asr_all_segments_filtered: 'jobs.skipped_asr_filtered',
}

const RESULT_SKIP_KEYS: Record<SkipReason, keyof Strings> = {
  vad_no_speech: 'results.skipped_vad_no_speech',
  asr_empty_transcript: 'results.skipped_asr_empty',
  asr_all_segments_filtered: 'results.skipped_asr_filtered',
}

const FAILURE_KEYS: Record<FailureCode, keyof Strings> = {
  undecodable_media: 'jobs.error_undecodable',
  internal: 'jobs.unknown_error',
}

/**
 * Short job-card line for a skipped job.
 *
 * @param code - Typed reason from the backend, or `null` when unknown.
 * @returns The catalog key to render.
 */
export function jobSkipMessageKey(code: SkipReason | null): keyof Strings {
  return (code && JOB_SKIP_KEYS[code]) || 'jobs.skipped'
}

/**
 * Fuller explanation for the result panel of a skipped job.
 *
 * @param code - Typed reason from the backend, or `null` when unknown.
 * @returns The catalog key to render.
 */
export function resultSkipMessageKey(code: SkipReason | null): keyof Strings {
  return (code && RESULT_SKIP_KEYS[code]) || 'results.skipped'
}

/**
 * Job-card line for a failed job.
 *
 * @param code - Typed failure code, or `null` for an older backend.
 * @param interrupted - Whether the job was interrupted by a backend restart,
 *   which describes the job's fate rather than a cause and wins over the code.
 * @returns The catalog key to render.
 */
export function failureMessageKey(code: FailureCode | null, interrupted: boolean): keyof Strings {
  if (interrupted) return 'jobs.interrupted'
  return (code && FAILURE_KEYS[code]) || 'jobs.unknown_error'
}
