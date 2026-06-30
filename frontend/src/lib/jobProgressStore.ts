import { create } from 'zustand'
import type { JobProgress } from './jobProgress'

/**
 * Shared, module-singleton store holding each job's latest reduced
 * {@link JobProgress}, keyed by `job_id`. Every mounted `useJobStream` publishes
 * into it, so an ancestor component (the header status bar) can read live
 * per-job progress it could not otherwise reach by props.
 */
export interface JobProgressState {
  /** Live reduced progress, keyed by `job_id`. Mirrors each mounted useJobStream. */
  byId: Record<string, JobProgress>
  /** Insert or replace a job's latest reduced progress. */
  setJobProgress: (jobId: string, progress: JobProgress) => void
  /** Drop a job's entry (stream unmount / job removed). No-op when absent. */
  removeJob: (jobId: string) => void
  /** Reset all entries (test resets / owner switch). */
  clear: () => void
}

/** Stable selector for the live map; lets consumers subscribe with one identity. */
export const selectById = (state: JobProgressState): Record<string, JobProgress> => state.byId

export const useJobProgressStore = create<JobProgressState>((set) => ({
  byId: {},
  setJobProgress: (jobId, progress) => set((state) => ({ byId: { ...state.byId, [jobId]: progress } })),
  removeJob: (jobId) =>
    set((state) => {
      if (!(jobId in state.byId)) return state // keep the reference stable -> no needless re-render
      const next = { ...state.byId }
      delete next[jobId]
      return { byId: next }
    }),
  clear: () => set({ byId: {} }),
}))
