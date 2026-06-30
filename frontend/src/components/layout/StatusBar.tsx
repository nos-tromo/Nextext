import { useMemo } from 'react'
import { Badge } from '@infra/ui'
import { useJobs } from '../../hooks/useJobs'
import { selectById, useJobProgressStore } from '../../lib/jobProgressStore'
import { summarizeJobs } from '../../lib/jobStatusSummary'
import type { JobListItem } from '../../api/types'

const EMPTY: JobListItem[] = [] // stable ref so the memo holds while the list is loading

/**
 * Global header status bar. Shows aggregate job counts
 * (processing / queued / finished / failed) and the running job's live step +
 * progress. The owner-scoped `['jobs']` list is the authoritative job set; the
 * live progress store overlays it, so a `queued -> running` flip the list has
 * not refetched still updates the bar in real time. Renders nothing when the
 * caller has no jobs.
 */
export function StatusBar() {
  const jobs = useJobs()
  const live = useJobProgressStore(selectById)
  const { counts, running } = useMemo(() => summarizeJobs(jobs.data?.jobs ?? EMPTY, live), [jobs.data, live])

  if (counts.total === 0) return null

  const pct = Math.round((running?.progress ?? 0) * 100)
  return (
    <div className="flex flex-col items-end gap-1 text-xs">
      <div className="flex items-center gap-2">
        {counts.processing > 0 && <Badge variant="accent">{`${counts.processing} processing`}</Badge>}
        {counts.queued > 0 && <Badge variant="neutral">{`${counts.queued} queued`}</Badge>}
        {counts.finished > 0 && <Badge variant="neutral">{`${counts.finished} finished`}</Badge>}
        {counts.failed > 0 && <Badge variant="danger">{`${counts.failed} failed`}</Badge>}
      </div>
      {running && (
        <div className="flex items-center gap-2" title={running.fileName}>
          <span className="max-w-[12rem] truncate text-muted-foreground">{running.stageLabel ?? 'Working…'}</span>
          <div className="h-1.5 w-24 overflow-hidden rounded bg-muted">
            <div className="h-full bg-primary" style={{ width: `${pct}%` }} />
          </div>
          <span className="tabular-nums text-muted-foreground">{`${pct}%`}</span>
        </div>
      )}
    </div>
  )
}
