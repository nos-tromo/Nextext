import type { ReactNode } from 'react'
import { Banner, EmptyState, SectionHeader, Spinner } from '@infra/ui'
import { useJobs } from '../../hooks/useJobs'
import { JobCard } from './JobCard'
import { BatchDownloadMenu } from './BatchDownloadMenu'
import { ClearJobsMenu } from './ClearJobsMenu'
import { StatusBar } from '../layout/StatusBar'
import { useT } from '../../i18n/LanguageContext'
import { describeError } from '../../api/errorMessage'

/**
 * The jobs list: its header (title, batch status, the batch download and clear
 * menus) above a JobCard per discovered job, newest first.
 */
export function BatchProgress() {
  const t = useT()
  const jobs = useJobs()
  const items = jobs.data?.jobs ?? []
  const completedCount = items.filter((job) => job.status === 'completed').length

  let body: ReactNode
  if (jobs.isLoading) {
    body = (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Spinner label={t('jobs.loading')} />
        <span aria-hidden>{t('jobs.loading')}</span>
      </div>
    )
  } else if (jobs.error) {
    const d = describeError(jobs.error)
    body = (
      <Banner variant="danger">
        {t('jobs.load_failed')} {t(d.key, d.vars)}
      </Banner>
    )
  } else if (items.length === 0) {
    body = <EmptyState>{t('jobs.none_yet')}</EmptyState>
  } else {
    body = items.map((job) => <JobCard key={job.job_id} job={job} />)
  }

  return (
    <div className="space-y-3">
      <SectionHeader
        title={t('home.jobs_heading')}
        caption={<StatusBar />}
        actions={
          items.length > 0 && (
            <>
              <BatchDownloadMenu completedCount={completedCount} />
              <ClearJobsMenu jobs={items} />
            </>
          )
        }
      />
      {body}
    </div>
  )
}
