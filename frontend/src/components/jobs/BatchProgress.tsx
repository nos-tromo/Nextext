import { useJobs } from '../../hooks/useJobs'
import { JobCard } from './JobCard'
import { BatchDownloadMenu } from './BatchDownloadMenu'
import { ClearJobsMenu } from './ClearJobsMenu'
import { Spinner } from '@infra/ui'
import { Banner } from '@infra/ui'
import { useT } from '../../i18n/LanguageContext'
import { describeError } from '../../api/errorMessage'

/** Renders a JobCard per discovered job (newest first), plus a batch download. */
export function BatchProgress() {
  const t = useT()
  const jobs = useJobs()
  if (jobs.isLoading) return <div className="flex items-center gap-2 text-sm text-muted-foreground"><Spinner label={t('jobs.loading')} /><span aria-hidden>{t('jobs.loading')}</span></div>
  if (jobs.error) {
    const d = describeError(jobs.error)
    return (
      <Banner variant="danger">
        {t('jobs.load_failed')} {t(d.key, d.vars)}
      </Banner>
    )
  }
  const items = jobs.data?.jobs ?? []
  if (items.length === 0) return <p className="text-sm text-muted-foreground">{t('jobs.none_yet')}</p>
  const completedCount = items.filter((job) => job.status === 'completed').length
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-end gap-2">
        <BatchDownloadMenu completedCount={completedCount} />
        <ClearJobsMenu jobs={items} />
      </div>
      {items.map((job) => (
        <JobCard key={job.job_id} job={job} />
      ))}
    </div>
  )
}
