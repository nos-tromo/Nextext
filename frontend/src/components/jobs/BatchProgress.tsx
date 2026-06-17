import { useJobs } from '../../hooks/useJobs'
import { JobCard } from './JobCard'
import { Spinner } from '../common/Spinner'
import { Banner } from '@infra/ui'

/** Renders a JobCard per discovered job (newest first). */
export function BatchProgress() {
  const jobs = useJobs()
  if (jobs.isLoading) return <Spinner label="Loading jobs…" />
  if (jobs.error) return <Banner variant="danger">{`Could not load jobs: ${String(jobs.error)}`}</Banner>
  const items = jobs.data?.jobs ?? []
  if (items.length === 0) return <p className="text-sm text-muted-foreground">No jobs yet.</p>
  return (
    <div className="space-y-3">
      {items.map((job) => (
        <JobCard key={job.job_id} job={job} />
      ))}
    </div>
  )
}
