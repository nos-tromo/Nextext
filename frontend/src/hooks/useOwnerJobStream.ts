import { useEffect } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { streamSse } from '../api/sse'
import { ownerEventsPath } from '../api/jobs'
import { toJobEvent } from '../lib/jobEvents'
import { initialJobProgress, reduceJobEvent, type JobProgress } from '../lib/jobProgress'
import { useJobProgressStore } from '../lib/jobProgressStore'

const RECONNECT_DELAY_MS = 1500
/** Maximum consecutive reconnect attempts before giving up on the stream. */
export const MAX_RECONNECTS = 5

/**
 * Subscribe to the owner-multiplexed SSE stream and fan its frames into the
 * shared {@link useJobProgressStore}, keyed by `job_id`.
 *
 * Mounted once for the whole session (in the app Shell), this opens exactly one
 * connection carrying events for every job the caller owns — so a batch of any
 * size uses one connection instead of one per job card, never approaching the
 * browser's per-host connection limit. Each frame is self-identifying
 * (`data.job_id`); progress is folded per job through the same idempotent
 * {@link reduceJobEvent} used by the single-job path, so the backend's
 * replay-on-connect is safe to re-read after a reconnect. When a job first
 * reaches a terminal state, the `['jobs']` query is invalidated so list-derived
 * views (completed count, batch download) refresh.
 *
 * Reconnects on a dropped stream up to {@link MAX_RECONNECTS} consecutive times;
 * any received event resets the budget. Cleans up on unmount via AbortController.
 */
export function useOwnerJobStream(): void {
  const queryClient = useQueryClient()

  useEffect(() => {
    const controller = new AbortController()
    let cancelled = false
    let reconnects = 0
    // Per-job fold state and terminal-dedupe, retained across reconnects so a
    // replayed history re-reduces onto the same monotonic progress.
    const states = new Map<string, JobProgress>()
    const terminalSeen = new Set<string>()
    // Non-reactive access: this hook is a *producer*; it publishes to the store
    // without re-rendering on its own writes. The action is stable.
    const { setJobProgress } = useJobProgressStore.getState()

    async function run(): Promise<void> {
      while (!cancelled && reconnects <= MAX_RECONNECTS) {
        try {
          for await (const frame of streamSse(ownerEventsPath(), controller.signal)) {
            const event = toJobEvent(frame)
            if (!event) continue
            reconnects = 0 // a real event resets the consecutive-failure budget
            const jobId = event.data.job_id
            if (!jobId) continue
            const prev = states.get(jobId) ?? initialJobProgress()
            const next = reduceJobEvent(prev, event)
            states.set(jobId, next)
            setJobProgress(jobId, next)
            if (next.terminal && !terminalSeen.has(jobId)) {
              terminalSeen.add(jobId)
              void queryClient.invalidateQueries({ queryKey: ['jobs'] })
            }
          }
        } catch {
          if (cancelled || controller.signal.aborted) return
        }
        if (!cancelled) {
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
    }
  }, [queryClient])
}
