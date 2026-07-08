import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

// Script the OWNER-multiplexed stream: Transcribing(0) -> Transcribing done(0.2)
// -> Translating(0.2), each frame tagged with its job_id, then stay open (a real
// SSE connection does not close mid-run) so the job holds at Translating.
const { frames } = vi.hoisted(() => ({
  frames: [
    { event: 'stage_started', data: '{"job_id":"j1","stage":"Transcribing","stage_index":0,"progress":0,"timestamp":"t"}' },
    { event: 'stage_completed', data: '{"job_id":"j1","stage":"Transcribing","stage_index":0,"progress":0.2,"timestamp":"t","result_delta":null}' },
    { event: 'stage_started', data: '{"job_id":"j1","stage":"Translating","stage_index":1,"progress":0.2,"timestamp":"t"}' },
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
import { useOwnerJobStream } from '../../hooks/useOwnerJobStream'
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

/** Mounts the single owner stream (as the Shell does) without any UI. */
function StreamMount() {
  useOwnerJobStream()
  return null
}

function mountAll() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => new Response(JSON.stringify({ jobs: [job] }), { status: 200, headers: { 'content-type': 'application/json' } })),
  )
  return render(
    <QueryClientProvider client={qc}>
      <StreamMount />
      <StatusBar />
      <JobCard job={job} />
    </QueryClientProvider>,
  )
}

beforeEach(() => useJobProgressStore.getState().clear())
afterEach(() => vi.restoreAllMocks())

describe('owner stream + StatusBar + JobCard integration', () => {
  it('drives the job card step in real time while the header shows batch progress', async () => {
    mountAll()
    // The single owner stream reduces the frames and publishes into the shared
    // store; the JobCard reads its own job's live step from that store.
    await waitFor(() => expect(screen.getByText(/Translating/)).toBeInTheDocument())
    expect(screen.getByText(/20%/)).toBeInTheDocument()
    // The header StatusBar (a sibling) reads the same store: the job is
    // processing and 0 of 1 files are done yet.
    expect(screen.getByText('1 processing')).toBeInTheDocument()
    expect(screen.getByText('0/1')).toBeInTheDocument()
  })
})
