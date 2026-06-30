import { afterEach, describe, expect, it, vi } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'

// Mock the SSE module so the hook consumes a scripted stream (no network).
// `vi.hoisted` makes `frames` available inside the hoisted vi.mock factory
// (a plain top-level const would be in the TDZ when the factory runs).
const { frames } = vi.hoisted(() => ({
  frames: [
    { event: 'stage_started', data: '{"stage":"Transcribing","stage_index":0,"progress":0,"timestamp":"t"}' },
    { event: 'stage_completed', data: '{"stage":"Transcribing","stage_index":0,"progress":0.2,"timestamp":"t","result_delta":null}' },
    { event: 'job_completed', data: '{"job_id":"j1","skipped":false,"timestamp":"t"}' },
  ],
}))
vi.mock('../api/sse', () => ({
  async *streamSse() {
    for (const f of frames) yield f
  },
}))

import { useJobStream } from './useJobStream'
import { useJobProgressStore } from '../lib/jobProgressStore'

afterEach(() => {
  vi.restoreAllMocks()
  useJobProgressStore.getState().clear()
})

describe('useJobStream', () => {
  it('drives progress to a terminal completed state', async () => {
    const { result } = renderHook(() => useJobStream('j1'))
    await waitFor(() => expect(result.current.terminal).toBe(true))
    expect(result.current.status).toBe('completed')
    expect(result.current.progress).toBe(1)
  })

  it('publishes its reduced progress into the shared store', async () => {
    const { result } = renderHook(() => useJobStream('j1'))
    await waitFor(() => expect(result.current.terminal).toBe(true))
    const entry = useJobProgressStore.getState().byId.j1
    expect(entry?.status).toBe('completed')
    expect(entry?.terminal).toBe(true)
  })

  it('removes its store entry on unmount', async () => {
    const { unmount } = renderHook(() => useJobStream('j1'))
    await waitFor(() => expect(useJobProgressStore.getState().byId.j1).toBeDefined())
    unmount()
    expect(useJobProgressStore.getState().byId.j1).toBeUndefined()
  })
})
