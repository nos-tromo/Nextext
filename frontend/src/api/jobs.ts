import { apiGet, apiSend } from './client'
import type { JobCreateResponse, JobListResponse, JobOptions, JobSnapshot } from './types'

/** SSE event-stream path for a job (relative to API_BASE). */
export function jobEventsPath(jobId: string): string {
  return `/jobs/${jobId}/events`
}

/**
 * SSE path for the owner-multiplexed stream (relative to API_BASE). One
 * connection carries events for every job the caller owns, so a batch never
 * approaches the browser's per-host connection limit.
 */
export function ownerEventsPath(): string {
  return '/jobs/events'
}

/** Queue a new job: multipart `file` + JSON `options`. */
export function submitJob(
  fileName: string,
  file: Blob,
  options: JobOptions,
): Promise<JobCreateResponse> {
  const form = new FormData()
  form.append('file', file, fileName)
  form.append('options', JSON.stringify(options))
  return apiSend<JobCreateResponse>('POST', '/jobs', { form })
}

/** List the caller's jobs (newest first). */
export function listJobs(signal?: AbortSignal): Promise<JobListResponse> {
  return apiGet<JobListResponse>('/jobs', signal)
}

/** Fetch one job snapshot. */
export function getJob(jobId: string, signal?: AbortSignal): Promise<JobSnapshot> {
  return apiGet<JobSnapshot>(`/jobs/${jobId}`, signal)
}

/** Delete a job (204 No Content). */
export function deleteJob(jobId: string): Promise<void> {
  return apiSend<void>('DELETE', `/jobs/${jobId}`)
}
