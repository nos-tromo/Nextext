import { useState } from 'react'
import { Button, Card } from '@infra/ui'
import { useDeleteJob } from '../../hooks/useJobs'
import { ResultPanel } from '../results/ResultPanel'
import { useJobProgressStore } from '../../lib/jobProgressStore'
import { initialJobProgress } from '../../lib/jobProgress'
import { useT } from '../../i18n/LanguageContext'
import { describeError } from '../../api/errorMessage'
import type { Strings } from '../../i18n'
import type { JobListItem } from '../../api/types'
import type { JobProgressStatus } from '../../lib/jobProgress'

const LABEL_KEY: Record<JobProgressStatus, keyof Strings> = {
  queued: 'jobs.status_queued',
  running: 'jobs.status_running',
  completed: 'processing.complete',
  failed: 'processing.failed',
  cancelled: 'processing.cancelled',
}

/**
 * Map a backend JobStatus (which may include `interrupted`) to a
 * terminal-safe JobProgressStatus seed. `interrupted` becomes `failed`
 * so the SSE hook does not enter a reconnect loop for a dead job.
 */
function seedOf(
  job: JobListItem,
  t: (key: keyof Strings, vars?: Record<string, string | number>) => string,
): { status: JobProgressStatus; error: string | null } {
  switch (job.status) {
    case 'running':
    case 'completed':
    case 'failed':
    case 'cancelled':
      return { status: job.status, error: job.error }
    case 'interrupted':
      return {
        status: 'failed',
        error: job.error ?? t('jobs.interrupted'),
      }
    default:
      return { status: 'queued', error: null }
  }
}

/** Live per-file progress, read from the shared owner-stream progress store. */
export function JobCard({ job }: { job: JobListItem }) {
  const t = useT()
  const seed = seedOf(job, t)
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
    <Card>
      <div className="flex items-center justify-between">
        <span className="text-foreground">{job.file_name}</span>
        <div className="flex items-center gap-3">
          {p.status === 'completed' && (
            <Button variant="ghost" size="sm" type="button" onClick={() => setShowResults((v) => !v)}>
              {showResults ? t('jobs.hide_results') : t('jobs.show_results')}
            </Button>
          )}
          <span className="text-sm text-muted-foreground">{t(LABEL_KEY[p.status])}</span>
          <Button
            variant="ghost"
            size="sm"
            type="button"
            disabled={del.isPending}
            onClick={() => del.mutate(job.job_id)}
          >
            {del.isPending ? t('jobs.removing') : t('common.remove')}
          </Button>
        </div>
      </div>
      <div className="mt-2 h-2 w-full overflow-hidden rounded bg-background">
        <div
          className={p.status === 'failed' ? 'h-full bg-danger' : 'h-full bg-primary'}
          style={{ width: `${p.status === 'failed' ? 100 : pct}%` }}
        />
      </div>
      <p className="mt-1 text-sm text-muted-foreground">
        {p.status === 'failed'
          ? job.status === 'interrupted'
            ? t('jobs.interrupted')
            : t('jobs.unknown_error')
          : p.skipped
            ? t('jobs.skipped')
            : p.stageLabel
              ? t('jobs.stage_progress', { stage: p.stageLabel, pct })
              : p.status === 'completed'
                ? t('jobs.done')
                : t('jobs.waiting')}
      </p>
      {del.isError && (
        <p className="mt-1 text-sm text-danger">
          {t('jobs.remove_failed')} {t(describeError(del.error).key, describeError(del.error).vars)}
        </p>
      )}
      {p.status === 'completed' && showResults && (
        <div className="mt-4">
          <ResultPanel jobId={job.job_id} fileName={job.file_name} />
        </div>
      )}
    </Card>
  )
}
