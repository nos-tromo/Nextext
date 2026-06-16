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

afterEach(() => vi.restoreAllMocks())

describe('useJobStream', () => {
  it('drives progress to a terminal completed state', async () => {
    const { result } = renderHook(() => useJobStream('j1'))
    await waitFor(() => expect(result.current.terminal).toBe(true))
    expect(result.current.status).toBe('completed')
    expect(result.current.progress).toBe(1)
  })
})
