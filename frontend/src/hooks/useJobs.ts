import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { deleteJob, listJobs, submitJob } from '../api/jobs'
import { ApiError } from '../api/client'
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

/**
 * Delete a single job, then refresh the jobs list. A 404 is treated as
 * already-gone (resolves quietly) so a double-click or a delete/refetch race
 * never surfaces a spurious error.
 */
export function useDeleteJob() {
  const qc = useQueryClient()
  return useMutation<void, Error, string>({
    mutationFn: async (jobId) => {
      try {
        await deleteJob(jobId)
      } catch (err) {
        if (err instanceof ApiError && err.status === 404) return // already gone
        throw err
      }
    },
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['jobs'] })
    },
  })
}

/** Outcome of a bulk clear: how many deletions succeeded vs. failed. */
export interface ClearJobsResult {
  cleared: number
  failed: number
}

/**
 * Delete many jobs concurrently, tolerating individual failures, then refresh
 * the jobs list once. A per-job 404 counts as cleared (already gone). The
 * mutation always resolves (never rejects) so the list refetch runs regardless
 * of partial failure; callers inspect {@link ClearJobsResult} to report it.
 *
 * @returns Counts of successfully cleared and failed deletions.
 */
export function useClearJobs() {
  const qc = useQueryClient()
  return useMutation<ClearJobsResult, Error, string[]>({
    mutationFn: async (jobIds) => {
      const settled = await Promise.allSettled(
        jobIds.map(async (jobId) => {
          try {
            await deleteJob(jobId)
          } catch (err) {
            if (err instanceof ApiError && err.status === 404) return // already gone -> cleared
            throw err
          }
        }),
      )
      const failed = settled.filter((r) => r.status === 'rejected').length
      return { cleared: settled.length - failed, failed }
    },
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['jobs'] })
    },
  })
}
