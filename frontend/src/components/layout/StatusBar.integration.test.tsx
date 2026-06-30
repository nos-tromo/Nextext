import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

// Script a stream that advances Transcribing(0) -> Transcribing done(0.2) -> Translating(0.2),
// then stays open (a real SSE connection does not close mid-run) so the job holds at Translating.
const { frames } = vi.hoisted(() => ({
  frames: [
    { event: 'stage_started', data: '{"stage":"Transcribing","stage_index":0,"progress":0,"timestamp":"t"}' },
    { event: 'stage_completed', data: '{"stage":"Transcribing","stage_index":0,"progress":0.2,"timestamp":"t","result_delta":null}' },
    { event: 'stage_started', data: '{"stage":"Translating","stage_index":1,"progress":0.2,"timestamp":"t"}' },
  ],
}))
vi.mock('../../api/sse', () => ({
  async *streamSse() {
    for (const f of frames) yield f
    await new Promise(() => {
      /* emulate an open SSE connection that never closes */
    })
  },
}))

import { JobCard } from '../jobs/JobCard'
import { StatusBar } from './StatusBar'
import { useJobProgressStore } from '../../lib/jobProgressStore'
import type { JobListItem } from '../../api/types'

const job: JobListItem = {
  job_id: 'j1',
  status: 'running',
  file_name: 'clip.wav',
  stage: null,
  progress: 0,
  error: null,
  created_at: 't',
  started_at: null,
  finished_at: null,
  task: 'transcribe',
}

function mountBoth() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => new Response(JSON.stringify({ jobs: [job] }), { status: 200, headers: { 'content-type': 'application/json' } })),
  )
  return render(
    <QueryClientProvider client={qc}>
      <StatusBar />
      <JobCard job={job} />
    </QueryClientProvider>,
  )
}

beforeEach(() => useJobProgressStore.getState().clear())
afterEach(() => vi.restoreAllMocks())

describe('StatusBar + JobCard integration', () => {
  it('advances the header step in real time as the job stream progresses', async () => {
    mountBoth()
    // The JobCard's useJobStream reduces the stream and publishes into the shared
    // store; the StatusBar (a header sibling) reads that store and re-renders.
    await waitFor(() => expect(screen.getByText('Translating')).toBeInTheDocument())
    expect(screen.getByText('20%')).toBeInTheDocument()
    expect(screen.getByText('1 processing')).toBeInTheDocument()
  })
})
