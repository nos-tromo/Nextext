import type { FailureCode, JobEvent, SkipReason } from '../api/types'

export type JobProgressStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'

export interface JobProgress {
  status: JobProgressStatus
  stageIndex: number
  stageLabel: string | null
  progress: number
  error: string | null
  skipped: boolean
  /** Why nothing was transcribed; drives the localized message. */
  skipReason: SkipReason | null
  /** Why the job failed; `error` itself is a static, non-explanatory string. */
  errorCode: FailureCode | null
  terminal: boolean
}

/** Terminal detail carried by the job list, used to seed state after a reload. */
export interface JobProgressSeed {
  skipped?: boolean
  skipReason?: SkipReason | null
  errorCode?: FailureCode | null
}

const TERMINAL: ReadonlySet<JobProgressStatus> = new Set(['completed', 'failed', 'cancelled'])

/**
 * Seed progress from a known starting status (e.g. a snapshot on reload).
 *
 * The terminal detail matters here: after a browser reload there is no live
 * store entry, so a skipped job seeded with `skipped: false` would read as an
 * ordinary "done".
 *
 * @param status - The job's current status. Defaults to `'queued'`.
 * @param error - Optional error message, carried from a `failed` snapshot.
 * @param seed - Terminal detail from the job list (skip flag and codes).
 * @returns A {@link JobProgress} seeded from the snapshot values.
 */
export function initialJobProgress(
  status: JobProgressStatus = 'queued',
  error: string | null = null,
  seed: JobProgressSeed = {},
): JobProgress {
  return {
    status,
    stageIndex: 0,
    stageLabel: null,
    progress: status === 'completed' ? 1 : 0,
    error: status === 'failed' ? error : null,
    skipped: seed.skipped ?? false,
    skipReason: seed.skipReason ?? null,
    errorCode: seed.errorCode ?? null,
    terminal: TERMINAL.has(status),
  }
}

/**
 * Fold one job event into progress. Monotonic in `progress` and idempotent
 * under replay, so re-reading the event history after a reconnect is safe.
 */
export function reduceJobEvent(state: JobProgress, event: JobEvent): JobProgress {
  switch (event.name) {
    case 'stage_started':
    case 'stage_completed':
      return {
        ...state,
        status: state.terminal ? state.status : 'running',
        stageIndex: Math.max(state.stageIndex, event.data.stage_index),
        stageLabel: event.data.stage,
        progress: Math.max(state.progress, event.data.progress),
      }
    case 'job_completed':
      return {
        ...state,
        status: 'completed',
        progress: 1,
        skipped: event.data.skipped,
        skipReason: event.data.skip_reason_code ?? null,
        stageLabel: null,
        terminal: true,
      }
    case 'job_failed':
      return {
        ...state,
        status: 'failed',
        error: event.data.error,
        errorCode: event.data.error_code ?? null,
        terminal: true,
      }
    case 'job_cancelled':
      return { ...state, status: 'cancelled', terminal: true }
  }
}
