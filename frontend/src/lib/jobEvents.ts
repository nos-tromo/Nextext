import type { SseFrame } from '../api/sse'
import type { JobEvent, JobEventName } from '../api/types'

const JOB_EVENT_NAMES = new Set<JobEventName>([
  'stage_started',
  'stage_completed',
  'job_completed',
  'job_failed',
  'job_cancelled',
])

function isJobEventName(name: string): name is JobEventName {
  return JOB_EVENT_NAMES.has(name as JobEventName)
}

/**
 * Convert a raw SSE frame to a typed JobEvent, or null when the frame is not
 * a recognized job event or its data is not valid JSON.
 */
export function toJobEvent(frame: SseFrame): JobEvent | null {
  if (!isJobEventName(frame.event)) return null
  let data: unknown
  try {
    data = JSON.parse(frame.data)
  } catch {
    return null
  }
  if (typeof data !== 'object' || data === null) return null
  return { name: frame.event, data } as JobEvent
}
