import { useEffect, useState } from 'react'
import { streamSse } from '../api/sse'
import { jobEventsPath } from '../api/jobs'
import { toJobEvent } from '../lib/jobEvents'
import { initialJobProgress, reduceJobEvent, type JobProgress, type JobProgressStatus } from '../lib/jobProgress'

const RECONNECT_DELAY_MS = 1500

/**
 * Subscribe to a job's SSE event stream and return live JobProgress. Reconnects
 * on a non-terminal drop (the backend replays history on connect; the reducer is
 * idempotent, so re-reading is safe). Cleans up on unmount via AbortController.
 */
export function useJobStream(jobId: string, initialStatus: JobProgressStatus = 'queued'): JobProgress {
  const [progress, setProgress] = useState<JobProgress>(() => initialJobProgress(initialStatus))

  useEffect(() => {
    const controller = new AbortController()
    let cancelled = false
    let state = initialJobProgress(initialStatus)

    async function run(): Promise<void> {
      while (!cancelled && !state.terminal) {
        try {
          for await (const frame of streamSse(jobEventsPath(jobId), controller.signal)) {
            const event = toJobEvent(frame)
            if (!event) continue
            state = reduceJobEvent(state, event)
            setProgress(state)
            if (state.terminal) return
          }
        } catch {
          if (cancelled || controller.signal.aborted) return
        }
        if (!cancelled && !state.terminal) {
          await new Promise((resolve) => setTimeout(resolve, RECONNECT_DELAY_MS))
        }
      }
    }

    void run()
    return () => {
      cancelled = true
      controller.abort()
    }
  }, [jobId, initialStatus])

  return progress
}
