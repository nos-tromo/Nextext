import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { StatusBar } from './StatusBar'
import { useJobProgressStore } from '../../lib/jobProgressStore'
import type { JobListItem, JobStatus } from '../../api/types'
import type { JobProgress, JobProgressStatus } from '../../lib/jobProgress'

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

function stubJobs(jobs: JobListItem[]): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => new Response(JSON.stringify({ jobs }), { status: 200, headers: { 'content-type': 'application/json' } })),
  )
}

function mountStatusBar() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <StatusBar />
    </QueryClientProvider>,
  )
}

beforeEach(() => useJobProgressStore.getState().clear())
afterEach(() => vi.restoreAllMocks())

describe('StatusBar', () => {
  it('shows count pills derived from the jobs list', async () => {
    stubJobs([
      mkJob({ job_id: 'r', status: 'running' }),
      mkJob({ job_id: 'c1', status: 'completed' }),
      mkJob({ job_id: 'c2', status: 'completed' }),
    ])
    mountStatusBar()
    expect(await screen.findByText('1 processing')).toBeInTheDocument()
    expect(await screen.findByText('2 finished')).toBeInTheDocument()
  })

  it('hides the failed pill when there are no failures, and shows it when there are', async () => {
    stubJobs([mkJob({ job_id: 'c', status: 'completed' })])
    const { unmount } = mountStatusBar()
    await screen.findByText('1 finished')
    expect(screen.queryByText(/failed/)).toBeNull()
    unmount()

    stubJobs([mkJob({ job_id: 'f', status: 'failed' })])
    mountStatusBar()
    expect(await screen.findByText('1 failed')).toBeInTheDocument()
  })

  it('shows batch progress as finished-and-failed over total files', async () => {
    stubJobs([
      mkJob({ job_id: 'c1', status: 'completed' }),
      mkJob({ job_id: 'f1', status: 'failed' }),
      mkJob({ job_id: 'r1', status: 'running' }),
      mkJob({ job_id: 'q1', status: 'queued' }),
    ])
    mountStatusBar()
    // 2 of 4 terminal (1 completed + 1 failed) -> "2/4"
    expect(await screen.findByText('2/4')).toBeInTheDocument()
  })

  it('counts a live-completed job toward batch progress before the list refetches', async () => {
    stubJobs([mkJob({ job_id: 'a', status: 'running', file_name: 'a.wav' })])
    useJobProgressStore.getState().setJobProgress('a', mkProgress({ status: 'completed', progress: 1, terminal: true }))
    mountStatusBar()
    expect(await screen.findByText('1/1')).toBeInTheDocument()
    expect(await screen.findByText('1 finished')).toBeInTheDocument()
  })

  it('renders nothing when there are no jobs', async () => {
    const fetchSpy = vi.fn(
      async () => new Response(JSON.stringify({ jobs: [] }), { status: 200, headers: { 'content-type': 'application/json' } }),
    )
    vi.stubGlobal('fetch', fetchSpy)
    mountStatusBar()
    await waitFor(() => expect(fetchSpy).toHaveBeenCalled())
    expect(screen.queryByText(/processing|queued|finished|failed/)).toBeNull()
  })

  it('shows the batch progress fraction whenever there are jobs, even with none running', async () => {
    stubJobs([mkJob({ job_id: 'c', status: 'completed' }), mkJob({ job_id: 'q', status: 'queued' })])
    mountStatusBar()
    await screen.findByText('1 finished')
    expect(await screen.findByText('1/2')).toBeInTheDocument()
  })
})
