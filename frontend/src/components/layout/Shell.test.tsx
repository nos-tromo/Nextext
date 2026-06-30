import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Shell } from './Shell'
import { useJobProgressStore } from '../../lib/jobProgressStore'
import type { JobListItem } from '../../api/types'

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

beforeEach(() => useJobProgressStore.getState().clear())
afterEach(() => vi.restoreAllMocks())

describe('Shell', () => {
  it('renders the app title and its children', () => {
    stubJobs([])
    mountShell()
    expect(screen.getByText('Nextext')).toBeInTheDocument()
    expect(screen.getByText('page-body')).toBeInTheDocument()
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
})
