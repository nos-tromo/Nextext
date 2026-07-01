import { useEffect, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useJobStream } from '../../hooks/useJobStream'
import { useDeleteJob } from '../../hooks/useJobs'
import { ResultPanel } from '../results/ResultPanel'
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

/** Live per-file progress, driven by the job's SSE stream. */
export function JobCard({ job }: { job: JobListItem }) {
  const seed = seedOf(job)
  const p = useJobStream(job.job_id, seed.status, seed.error)
  const pct = Math.round(p.progress * 100)
  const [showResults, setShowResults] = useState(false)
  const del = useDeleteJob()

  // The live SSE stream updates this card's status locally, but the shared
  // ['jobs'] query (which aggregate views like the batch-download control read
  // for their completed-job count) is otherwise only fetched on mount. When
  // this job first reaches a terminal state, refresh that list so those views
  // react to the completion without a manual page reload. Jobs already terminal
  // at mount are skipped — the freshly fetched list already reflects them.
  const queryClient = useQueryClient()
  const wasTerminalOnMount = useRef(p.terminal)
  const refreshedList = useRef(false)
  useEffect(() => {
    if (p.terminal && !wasTerminalOnMount.current && !refreshedList.current) {
      refreshedList.current = true
      void queryClient.invalidateQueries({ queryKey: ['jobs'] })
    }
  }, [p.terminal, queryClient])

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
