import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { listJobs, submitJob } from '../api/jobs'
import type { JobListItem, JobOptions } from '../api/types'

/** The caller's jobs, re-fetched on mount (reload re-discovery). */
export function useJobs() {
  return useQuery({
    queryKey: ['jobs'],
    queryFn: ({ signal }) => listJobs(signal),
    refetchInterval: false,
  })
}

export interface SubmitBatchVars {
  files: File[]
  options: JobOptions
}

export interface SubmittedJob {
  job_id: string
  file_name: string
  error?: string
}

/**
 * Submit every file up front (so a reload can re-discover the whole batch),
 * then invalidate the jobs list so the UI re-renders with the new jobs.
 */
export function useSubmitBatch() {
  const qc = useQueryClient()
  return useMutation<SubmittedJob[], Error, SubmitBatchVars>({
    mutationFn: async ({ files, options }) => {
      const submitted: SubmittedJob[] = []
      for (const file of files) {
        try {
          const res = await submitJob(file.name, file, options)
          submitted.push({ job_id: res.job_id, file_name: file.name })
        } catch (err) {
          submitted.push({ job_id: '', file_name: file.name, error: err instanceof Error ? err.message : String(err) })
        }
      }
      return submitted
    },
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['jobs'] })
    },
  })
}

/** Status helper reused by JobCard styling. */
export function isActive(job: JobListItem): boolean {
  return job.status === 'queued' || job.status === 'running'
}
