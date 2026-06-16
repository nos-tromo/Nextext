import type { JobEvent } from '../api/types'

export type JobProgressStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'

export interface JobProgress {
  status: JobProgressStatus
  stageIndex: number
  stageLabel: string | null
  progress: number
  error: string | null
  skipped: boolean
  terminal: boolean
}

const TERMINAL: ReadonlySet<JobProgressStatus> = new Set(['completed', 'failed', 'cancelled'])

/**
 * Seed progress from a known starting status (e.g. a snapshot on reload).
 *
 * @param status - The job's current status. Defaults to `'queued'`.
 * @param error - Optional error message, carried from a `failed` snapshot.
 * @returns A {@link JobProgress} seeded from the snapshot values.
 */
export function initialJobProgress(status: JobProgressStatus = 'queued', error: string | null = null): JobProgress {
  return {
    status,
    stageIndex: 0,
    stageLabel: null,
    progress: status === 'completed' ? 1 : 0,
    error: status === 'failed' ? error : null,
    skipped: false,
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
      return { ...state, status: 'completed', progress: 1, skipped: event.data.skipped, stageLabel: null, terminal: true }
    case 'job_failed':
      return { ...state, status: 'failed', error: event.data.error, terminal: true }
    case 'job_cancelled':
      return { ...state, status: 'cancelled', terminal: true }
  }
}
