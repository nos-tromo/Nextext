import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

// Mock the SSE module so the single owner stream yields a scripted, multiplexed
// frame sequence (events for two jobs interleaved), then stays open — modelling
// the real long-lived connection that never ends on its own.
const { frames, streamSseMock } = vi.hoisted(() => ({
  frames: [
    { event: 'stage_started', data: '{"job_id":"j1","stage":"Transcribing","stage_index":0,"progress":0,"timestamp":"t"}' },
    { event: 'stage_started', data: '{"job_id":"j2","stage":"Transcribing","stage_index":0,"progress":0,"timestamp":"t"}' },
    { event: 'job_completed', data: '{"job_id":"j1","skipped":false,"timestamp":"t"}' },
    { event: 'job_completed', data: '{"job_id":"j2","skipped":false,"timestamp":"t"}' },
  ],
  streamSseMock: vi.fn(),
}))
vi.mock('../api/sse', () => ({ streamSse: streamSseMock }))

import { useOwnerJobStream } from './useOwnerJobStream'
import { useJobProgressStore } from '../lib/jobProgressStore'

function makeWrapper(client: QueryClient) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>
  }
}

afterEach(() => {
  vi.restoreAllMocks()
  streamSseMock.mockReset()
  useJobProgressStore.getState().clear()
})

describe('useOwnerJobStream', () => {
  it('routes multiplexed frames to the store by job_id over a single stream', async () => {
    streamSseMock.mockImplementation(async function* () {
      for (const f of frames) yield f
      await new Promise(() => {}) // stay open like a real owner stream
    })
    const client = new QueryClient()

    renderHook(() => useOwnerJobStream(), { wrapper: makeWrapper(client) })

    await waitFor(() => {
      const byId = useJobProgressStore.getState().byId
      expect(byId.j1?.status).toBe('completed')
      expect(byId.j2?.status).toBe('completed')
    })
    expect(useJobProgressStore.getState().byId.j1?.progress).toBe(1)
    expect(useJobProgressStore.getState().byId.j2?.terminal).toBe(true)
    // One connection for the whole batch, regardless of job count.
    expect(streamSseMock).toHaveBeenCalledTimes(1)
    expect(streamSseMock.mock.calls[0][0]).toBe('/jobs/events')
  })

  it('invalidates the jobs list when a job reaches a terminal state', async () => {
    streamSseMock.mockImplementation(async function* () {
      for (const f of frames) yield f
      await new Promise(() => {})
    })
    const client = new QueryClient()
    const invalidate = vi.spyOn(client, 'invalidateQueries')

    renderHook(() => useOwnerJobStream(), { wrapper: makeWrapper(client) })

    await waitFor(() => expect(invalidate).toHaveBeenCalled())
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ['jobs'] })
  })
})
