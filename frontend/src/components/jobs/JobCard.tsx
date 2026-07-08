import { useState } from 'react'
import { useDeleteJob } from '../../hooks/useJobs'
import { ResultPanel } from '../results/ResultPanel'
import { useJobProgressStore } from '../../lib/jobProgressStore'
import { initialJobProgress } from '../../lib/jobProgress'
import type { JobListItem } from '../../api/types'
import type { JobProgressStatus } from '../../lib/jobProgress'

const LABEL: Record<JobProgressStatus, string> = {
  queued: 'Queued',
  running: 'Processing',
  completed: 'Complete',
  failed: 'Failed',
  cancelled: 'Cancelled',
}

/**
 * Map a backend JobStatus (which may include `interrupted`) to a
 * terminal-safe JobProgressStatus seed. `interrupted` becomes `failed`
 * so the SSE hook does not enter a reconnect loop for a dead job.
 */
function seedOf(job: JobListItem): { status: JobProgressStatus; error: string | null } {
  switch (job.status) {
    case 'running':
    case 'completed':
    case 'failed':
    case 'cancelled':
      return { status: job.status, error: job.error }
    case 'interrupted':
      return {
        status: 'failed',
        error: job.error ?? 'Job was interrupted before it could finish.',
      }
    default:
      return { status: 'queued', error: null }
  }
}

/** Live per-file progress, read from the shared owner-stream progress store. */
export function JobCard({ job }: { job: JobListItem }) {
  const seed = seedOf(job)
  // The single owner-multiplexed SSE stream (mounted in the Shell) publishes
  // every job's reduced progress into this store, keyed by job_id. Fall back to
  // the list snapshot's status for jobs the stream has not reported yet (e.g. a
  // still-queued job with no events). List refetch on completion is handled by
  // the owner stream, so this card no longer owns a stream of its own.
  const live = useJobProgressStore((state) => state.byId[job.job_id])
  const p = live ?? initialJobProgress(seed.status, seed.error)
  const pct = Math.round(p.progress * 100)
  const [showResults, setShowResults] = useState(false)
  const del = useDeleteJob()

  return (
    <div className="rounded-lg border border-border p-4">
      <div className="flex items-center justify-between">
        <span className="text-foreground">{job.file_name}</span>
        <div className="flex items-center gap-3">
          {p.status === 'completed' && (
            <button
              type="button"
              onClick={() => setShowResults((v) => !v)}
              className="text-sm text-primary hover:underline"
            >
              {showResults ? 'Hide results' : 'Show results'}
            </button>
          )}
          <span className="text-sm text-muted-foreground">{LABEL[p.status]}</span>
          <button
            type="button"
            disabled={del.isPending}
            onClick={() => del.mutate(job.job_id)}
            className="text-sm text-muted-foreground transition-colors hover:text-danger disabled:cursor-not-allowed disabled:opacity-60"
          >
            {del.isPending ? 'Removing…' : 'Remove'}
          </button>
        </div>
      </div>
      <div className="mt-2 h-2 w-full overflow-hidden rounded bg-muted">
        <div
          className={p.status === 'failed' ? 'h-full bg-danger' : 'h-full bg-primary'}
          style={{ width: `${p.status === 'failed' ? 100 : pct}%` }}
        />
      </div>
      <p className="mt-1 text-sm text-muted-foreground">
        {p.status === 'failed'
          ? (p.error ?? 'Unknown error')
          : p.skipped
            ? 'Skipped — no processable content'
            : p.stageLabel
              ? `${p.stageLabel} (${pct}%)`
              : p.status === 'completed'
                ? 'Done'
                : 'Waiting…'}
      </p>
      {del.isError && (
        <p className="mt-1 text-sm text-danger">{`Could not remove job: ${del.error?.message ?? 'unknown error'}`}</p>
      )}
      {p.status === 'completed' && showResults && (
        <div className="mt-4">
          <ResultPanel jobId={job.job_id} fileName={job.file_name} />
        </div>
      )}
    </div>
  )
}
