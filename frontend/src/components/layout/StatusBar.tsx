import { useMemo } from 'react'
import { Badge } from '@infra/ui'
import { useT } from '../../i18n/LanguageContext'
import { useJobs } from '../../hooks/useJobs'
import { selectById, useJobProgressStore } from '../../lib/jobProgressStore'
import { summarizeJobs } from '../../lib/jobStatusSummary'
import type { JobListItem } from '../../api/types'

const EMPTY: JobListItem[] = [] // stable ref so the memo holds while the list is loading

/**
 * Global header status bar. Shows aggregate job counts
 * (processing / queued / finished / skipped / failed) plus an overall **batch progress**
 * bar measured in files done (terminal) over total — 1 of 10 finished reads
 * 10%. This tracks the whole upload rather than one job's intra-progress, so a
 * long batch advances steadily instead of resetting to 0% on every new file.
 * The owner-scoped `['jobs']` list is the authoritative job set; the live
 * progress store overlays it, so a job that just completed on the stream counts
 * toward progress before the list has refetched. Renders nothing with no jobs.
 */
export function StatusBar() {
  const t = useT()
  const jobs = useJobs()
  const live = useJobProgressStore(selectById)
  const { counts } = useMemo(() => summarizeJobs(jobs.data?.jobs ?? EMPTY, live), [jobs.data, live])

  if (counts.total === 0) return null

  const done = counts.finished + counts.skipped + counts.failed // terminal files
  const pct = Math.round((done / counts.total) * 100)
  return (
    <div className="flex flex-col items-end gap-1 text-xs">
      <div className="flex items-center gap-2">
        {counts.processing > 0 && (
          <Badge variant="accent">{t('common.jobs_processing', { count: counts.processing })}</Badge>
        )}
        {counts.queued > 0 && <Badge variant="neutral">{t('common.jobs_queued', { count: counts.queued })}</Badge>}
        {counts.finished > 0 && (
          <Badge variant="neutral">{t('common.jobs_finished', { count: counts.finished })}</Badge>
        )}
        {counts.skipped > 0 && (
          <Badge variant="neutral">{t('common.jobs_skipped', { count: counts.skipped })}</Badge>
        )}
        {counts.failed > 0 && <Badge variant="danger">{t('common.jobs_failed', { count: counts.failed })}</Badge>}
      </div>
      <div className="flex items-center gap-2" title={t('common.batch_progress', { pct })}>
        <div className="h-1.5 w-24 overflow-hidden rounded bg-muted">
          <div className="h-full bg-primary" style={{ width: `${pct}%` }} />
        </div>
        <span className="tabular-nums text-muted-foreground">{`${done}/${counts.total}`}</span>
      </div>
    </div>
  )
}
