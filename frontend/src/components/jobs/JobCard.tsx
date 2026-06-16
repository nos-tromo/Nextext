import { useJobStream } from '../../hooks/useJobStream'
import type { JobListItem } from '../../api/types'
import type { JobProgressStatus } from '../../lib/jobProgress'

const LABEL: Record<JobProgressStatus, string> = {
  queued: 'Queued',
  running: 'Processing',
  completed: 'Complete',
  failed: 'Failed',
  cancelled: 'Cancelled',
}

/** Live per-file progress, driven by the job's SSE stream. */
export function JobCard({ job }: { job: JobListItem }) {
  const seed: JobProgressStatus =
    job.status === 'running' || job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled'
      ? job.status
      : 'queued'
  const p = useJobStream(job.job_id, seed)
  const pct = Math.round(p.progress * 100)

  return (
    <div className="rounded-lg border border-border p-4">
      <div className="flex items-center justify-between">
        <span className="text-foreground">{job.file_name}</span>
        <span className="text-sm text-muted-foreground">{LABEL[p.status]}</span>
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
    </div>
  )
}
