import { describe, expect, it } from 'vitest'
import { summarizeJobs } from './jobStatusSummary'
import type { JobListItem, JobStatus } from '../api/types'
import type { JobProgress, JobProgressStatus } from './jobProgress'

function mkJob(partial: { job_id: string; status: JobStatus } & Partial<JobListItem>): JobListItem {
  return {
    file_name: `${partial.job_id}.wav`,
    stage: null,
    progress: 0,
    error: null,
    created_at: 't',
    started_at: null,
    finished_at: null,
    task: 'transcribe',
    ...partial,
  }
}

function mkProgress(partial: { status: JobProgressStatus } & Partial<JobProgress>): JobProgress {
  return { stageIndex: 0, stageLabel: null, progress: 0, error: null, skipped: false, terminal: false, ...partial }
}

describe('summarizeJobs', () => {
  it('returns all-zero counts and no running job for an empty list', () => {
    expect(summarizeJobs([], {})).toEqual({
      counts: { processing: 0, queued: 0, finished: 0, failed: 0, total: 0 },
      running: null,
    })
  })

  it('prefers the live store status over a stale list snapshot', () => {
    // queued->running never refetches the list, so the snapshot still says queued;
    // the live store knows the job is running mid-stage.
    const jobs = [mkJob({ job_id: 'a', status: 'queued', file_name: 'a.wav' }), mkJob({ job_id: 'b', status: 'completed' })]
    const live = { a: mkProgress({ status: 'running', stageLabel: 'Translating', progress: 0.4 }) }
    const { counts, running } = summarizeJobs(jobs, live)
    expect(counts).toEqual({ processing: 1, queued: 0, finished: 1, failed: 0, total: 2 })
    expect(running).toEqual({ stageLabel: 'Translating', progress: 0.4, fileName: 'a.wav' })
  })

  it('falls back to the list snapshot for the running job when no live entry exists', () => {
    const jobs = [mkJob({ job_id: 'a', status: 'running', stage: 'Transcribing', progress: 0.2, file_name: 'clip.wav' })]
    const { counts, running } = summarizeJobs(jobs, {})
    expect(counts.processing).toBe(1)
    expect(running).toEqual({ stageLabel: 'Transcribing', progress: 0.2, fileName: 'clip.wav' })
  })

  it('counts a mixed batch and preserves the total invariant', () => {
    const jobs = [
      mkJob({ job_id: 'r', status: 'running' }),
      mkJob({ job_id: 'q1', status: 'queued' }),
      mkJob({ job_id: 'q2', status: 'queued' }),
      mkJob({ job_id: 'c1', status: 'completed' }),
      mkJob({ job_id: 'c2', status: 'completed' }),
      mkJob({ job_id: 'c3', status: 'completed' }),
      mkJob({ job_id: 'f', status: 'failed' }),
    ]
    const { counts } = summarizeJobs(jobs, {})
    expect(counts).toEqual({ processing: 1, queued: 2, finished: 3, failed: 1, total: 7 })
    expect(counts.processing + counts.queued + counts.finished + counts.failed).toBe(counts.total)
  })

  it('folds interrupted and cancelled into the failed bucket', () => {
    const jobs = [mkJob({ job_id: 'i', status: 'interrupted' }), mkJob({ job_id: 'x', status: 'cancelled' })]
    expect(summarizeJobs(jobs, {}).counts.failed).toBe(2)
  })

  it('ignores live entries for jobs not in the list', () => {
    const jobs = [mkJob({ job_id: 'a', status: 'completed' })]
    const live = { ghost: mkProgress({ status: 'running' }) }
    const { counts, running } = summarizeJobs(jobs, live)
    expect(counts).toEqual({ processing: 0, queued: 0, finished: 1, failed: 0, total: 1 })
    expect(running).toBeNull()
  })

  it('is defensive about multiple running jobs: counts all, reports the first in list order', () => {
    const jobs = [mkJob({ job_id: 'a', status: 'running', file_name: 'a.wav' }), mkJob({ job_id: 'b', status: 'running', file_name: 'b.wav' })]
    const live = {
      a: mkProgress({ status: 'running', stageLabel: 'Transcribing', progress: 0.1 }),
      b: mkProgress({ status: 'running', stageLabel: 'Summarizing', progress: 0.6 }),
    }
    const { counts, running } = summarizeJobs(jobs, live)
    expect(counts.processing).toBe(2)
    expect(running?.fileName).toBe('a.wav')
  })

  it('passes through a null stage label and zero progress for a just-started running job', () => {
    const jobs = [mkJob({ job_id: 'a', status: 'running', stage: 'should-be-ignored', file_name: 'a.wav' })]
    const live = { a: mkProgress({ status: 'running', stageLabel: null, progress: 0 }) }
    expect(summarizeJobs(jobs, live).running).toEqual({ stageLabel: null, progress: 0, fileName: 'a.wav' })
  })
})
