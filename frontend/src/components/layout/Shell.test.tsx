import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Shell } from './Shell'
import { useJobProgressStore } from '../../lib/jobProgressStore'
import type { JobListItem } from '../../api/types'

// The Shell mounts the single owner-multiplexed SSE stream; mock it so the
// stream stays open (never yields) and never touches the network in Shell tests.
const { streamSseMock } = vi.hoisted(() => ({ streamSseMock: vi.fn() }))
vi.mock('../../api/sse', () => ({ streamSse: streamSseMock }))

function stubJobs(jobs: JobListItem[]): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => new Response(JSON.stringify({ jobs }), { status: 200, headers: { 'content-type': 'application/json' } })),
  )
}

function mountShell() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <Shell>
        <div>page-body</div>
      </Shell>
    </QueryClientProvider>,
  )
}

beforeEach(() => {
  useJobProgressStore.getState().clear()
  streamSseMock.mockReset()
  // Default: an owner stream that opens and stays open. It yields one inert
  // (non-job) frame the hook ignores, then blocks — modelling a live SSE
  // connection without emitting real progress into these layout tests.
  streamSseMock.mockImplementation(async function* () {
    yield { event: 'ping', data: '' }
    await new Promise(() => {})
  })
})
afterEach(() => vi.restoreAllMocks())

describe('Shell', () => {
  it('renders the app title and its children', () => {
    stubJobs([])
    mountShell()
    expect(screen.getByText('Nextext')).toBeInTheDocument()
    expect(screen.getByText('page-body')).toBeInTheDocument()
  })

  it('opens exactly one owner-multiplexed job stream', () => {
    stubJobs([])
    mountShell()
    expect(streamSseMock).toHaveBeenCalledTimes(1)
    expect(streamSseMock.mock.calls[0][0]).toBe('/jobs/events')
  })

  it('surfaces the job status bar in the header', async () => {
    stubJobs([
      {
        job_id: 'j1',
        status: 'completed',
        file_name: 'a.wav',
        stage: null,
        progress: 1,
        error: null,
        created_at: 't',
        started_at: null,
        finished_at: null,
        task: 'transcribe',
      },
    ])
    mountShell()
    expect(await screen.findByText('1 finished')).toBeInTheDocument()
  })

  it('pins the header to the top so it stays visible while scrolling', () => {
    stubJobs([])
    mountShell()
    expect(screen.getByRole('banner')).toHaveClass('sticky', 'top-0')
  })
})
