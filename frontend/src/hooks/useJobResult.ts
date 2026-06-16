import { useQuery } from '@tanstack/react-query'
import { getJob } from '../api/jobs'
import type { JobResult } from '../api/types'

/**
 * Fetch and cache a completed job's result.
 *
 * Queries the `/jobs/{id}` snapshot endpoint and returns the embedded
 * `result` object. Only enabled when the job is known to be in a terminal
 * state (caller passes `enabled`). The result is treated as immutable once
 * fetched (`staleTime: Infinity`).
 *
 * @param jobId - The job identifier.
 * @param enabled - Whether the query should run (set false for non-terminal jobs).
 * @returns TanStack Query result carrying {@link JobResult} or `null`.
 */
export function useJobResult(jobId: string, enabled: boolean) {
  return useQuery({
    queryKey: ['job', jobId, 'result'],
    queryFn: async ({ signal }): Promise<JobResult | null> => {
      const snapshot = await getJob(jobId, signal)
      return snapshot.result
    },
    enabled,
    staleTime: Infinity,
  })
}
