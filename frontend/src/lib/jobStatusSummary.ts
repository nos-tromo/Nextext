import type { JobListItem, JobStatus } from '../api/types'
import type { JobProgress } from './jobProgress'

/** Aggregate counts across the caller's jobs. Buckets are exhaustive, so
 * `processing + queued + finished + skipped + failed === total`. A skipped job
 * completed but produced nothing, so it is counted apart from `finished` —
 * "8 finished" should not include files that yielded no transcript. */
export interface JobCounts {
  processing: number
  queued: number
  finished: number
  skipped: number
  failed: number
  total: number
}

/** The running job's live detail for the status-bar step line. */
export interface RunningJob {
  stageLabel: string | null
  progress: number
  fileName: string
}

export interface JobStatusSummary {
  counts: JobCounts
  running: RunningJob | null
}

/**
 * Derive header counts plus the running job's live detail.
 *
 * The `['jobs']` list is the authoritative, owner-scoped job set; `live` only
 * overlays it. For each job the live store entry wins when present (it is
 * fresher — e.g. it reflects a `queued -> running` flip the list has not
 * refetched), otherwise the list snapshot is used. Live entries for jobs absent
 * from the list are ignored, which keeps a global store owner-safe.
 *
 * @param jobs - The caller's jobs from `GET /jobs` (authoritative set + scope).
 * @param live - Live reduced progress keyed by `job_id` (from the shared store).
 * @returns Counts by bucket and the first running job's live detail, or `null`.
 */
export function summarizeJobs(jobs: JobListItem[], live: Record<string, JobProgress>): JobStatusSummary {
  const counts: JobCounts = { processing: 0, queued: 0, finished: 0, skipped: 0, failed: 0, total: jobs.length }
  let running: RunningJob | null = null

  for (const job of jobs) {
    const livePart = live[job.job_id]
    const status: JobStatus = livePart?.status ?? job.status
    switch (status) {
      case 'running':
        counts.processing += 1
        running ??= livePart
          ? { stageLabel: livePart.stageLabel, progress: livePart.progress, fileName: job.file_name }
          : { stageLabel: job.stage, progress: job.progress, fileName: job.file_name }
        break
      case 'queued':
        counts.queued += 1
        break
      case 'completed':
        if (livePart?.skipped ?? job.skipped) counts.skipped += 1
        else counts.finished += 1
        break
      default: // failed | interrupted | cancelled (the worker never sets the last two; folded defensively)
        counts.failed += 1
    }
  }

  return { counts, running }
}
