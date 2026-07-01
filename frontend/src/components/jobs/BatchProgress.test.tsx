import { afterEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

// The job's SSE stream drives it straight to completion.
vi.mock('../../api/sse', () => ({
  async *streamSse() {
    yield { event: 'job_completed', data: '{"job_id":"j1","skipped":false,"timestamp":"t"}' }
  },
}))

import { BatchProgress } from './BatchProgress'

function jobsResponse(status: string): Response {
  return new Response(
    JSON.stringify({
      jobs: [
        {
          job_id: 'j1',
          status,
          file_name: 'clip.wav',
          progress: status === 'completed' ? 1 : 0.5,
          created_at: 't',
          task: 'transcribe',
        },
      ],
    }),
    { status: 200, headers: { 'content-type': 'application/json' } },
  )
}

function emptyJobsResponse(): Response {
  return new Response(JSON.stringify({ jobs: [] }), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  })
}

function mountBatchProgress() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <BatchProgress />
    </QueryClientProvider>,
  )
}

afterEach(() => vi.restoreAllMocks())

describe('BatchProgress batch-download enablement', () => {
  it('enables the download control after a job completes via SSE, without a reload', async () => {
    let listCalls = 0
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        listCalls += 1
        // The first list load sees the job still running; once the SSE
        // completion refreshes the list, the refetch sees it completed.
        return jobsResponse(listCalls === 1 ? 'running' : 'completed')
      }),
    )

    mountBatchProgress()

    const trigger = () => screen.getByRole('button', { name: /Download all jobs/ })
    // The control must become enabled after the live completion — the bug was
    // that it stayed disabled until a manual page reload.
    await waitFor(() => expect(trigger()).toBeEnabled())
  })
})

describe('BatchProgress clear control', () => {
  it('renders the Clear control alongside the jobs when the list is non-empty', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jobsResponse('completed')))
    mountBatchProgress()
    expect(await screen.findByRole('button', { name: /Clear ▾/ })).toBeInTheDocument()
    expect(screen.getByText('clip.wav')).toBeInTheDocument()
  })

  it('shows the empty state and no Clear control when there are no jobs', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => emptyJobsResponse()))
    mountBatchProgress()
    expect(await screen.findByText('No jobs yet.')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Clear/ })).toBeNull()
  })
})
