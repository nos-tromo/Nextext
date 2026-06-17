import { useJobs } from '../../hooks/useJobs'
import { JobCard } from './JobCard'
import { BatchDownloadMenu } from './BatchDownloadMenu'
import { Spinner } from '../common/Spinner'
import { ErrorBanner } from '../common/ErrorBanner'

/** Renders a JobCard per discovered job (newest first), plus a batch download. */
export function BatchProgress() {
  const jobs = useJobs()
  if (jobs.isLoading) return <Spinner label="Loading jobs…" />
  if (jobs.error) return <ErrorBanner message={`Could not load jobs: ${String(jobs.error)}`} />
  const items = jobs.data?.jobs ?? []
  if (items.length === 0) return <p className="text-sm text-muted-foreground">No jobs yet.</p>
  const completedCount = items.filter((job) => job.status === 'completed').length
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-end">
        <BatchDownloadMenu completedCount={completedCount} />
      </div>
      {items.map((job) => (
        <JobCard key={job.job_id} job={job} />
      ))}
    </div>
  )
}
