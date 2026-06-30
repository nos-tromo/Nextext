import { useEffect, useState } from 'react'
import { streamSse } from '../api/sse'
import { jobEventsPath } from '../api/jobs'
import { toJobEvent } from '../lib/jobEvents'
import { initialJobProgress, reduceJobEvent, type JobProgress, type JobProgressStatus } from '../lib/jobProgress'
import { useJobProgressStore } from '../lib/jobProgressStore'

const RECONNECT_DELAY_MS = 1500
/** Maximum number of consecutive reconnect attempts before giving up on a non-terminal stream. */
export const MAX_RECONNECTS = 5

/**
 * Subscribe to a job's SSE event stream and return live JobProgress. Reconnects
 * on a non-terminal drop up to {@link MAX_RECONNECTS} consecutive times — any
 * received event resets the budget (the backend replays history on connect; the
 * reducer is idempotent, so re-reading is safe). Cleans up on unmount via
 * AbortController.
 *
 * @param jobId - The job to stream.
 * @param initialStatus - Status seed from a cached snapshot (default `'queued'`).
 * @param initialError - Error message seed from a failed snapshot (default `null`).
 */
export function useJobStream(
  jobId: string,
  initialStatus: JobProgressStatus = 'queued',
  initialError: string | null = null,
): JobProgress {
  const [progress, setProgress] = useState<JobProgress>(() => initialJobProgress(initialStatus, initialError))

  useEffect(() => {
    const controller = new AbortController()
    let cancelled = false
    let state = initialJobProgress(initialStatus, initialError)
    let reconnects = 0
    // Non-reactive access: this hook is a *producer*; it must publish to the
    // store without re-rendering on its own writes. Actions are stable.
    const { setJobProgress, removeJob } = useJobProgressStore.getState()
    setJobProgress(jobId, state) // seed on mount (terminal seeds publish here too, never connecting)

    async function run(): Promise<void> {
      while (!cancelled && !state.terminal && reconnects <= MAX_RECONNECTS) {
        try {
          for await (const frame of streamSse(jobEventsPath(jobId), controller.signal)) {
            const event = toJobEvent(frame)
            if (!event) continue
            reconnects = 0 // a real event resets the consecutive-failure budget
            state = reduceJobEvent(state, event)
            setProgress(state)
            setJobProgress(jobId, state) // mirror live progress into the shared store
            if (state.terminal) return
          }
        } catch {
          if (cancelled || controller.signal.aborted) return
        }
        if (!cancelled && !state.terminal) {
          reconnects += 1
          if (reconnects > MAX_RECONNECTS) return
          await new Promise((resolve) => setTimeout(resolve, RECONNECT_DELAY_MS))
        }
      }
    }

    void run()
    return () => {
      cancelled = true
      controller.abort()
      removeJob(jobId)
    }
  }, [jobId, initialStatus, initialError])

  return progress
}
