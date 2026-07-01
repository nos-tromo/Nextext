import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'

vi.mock('../api/jobs', () => ({
  deleteJob: vi.fn(),
  listJobs: vi.fn(),
  submitJob: vi.fn(),
}))

import { deleteJob } from '../api/jobs'
import { ApiError } from '../api/client'
import { useClearJobs, useDeleteJob } from './useJobs'

const mockedDelete = vi.mocked(deleteJob)

function makeWrapper() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={qc}>{children}</QueryClientProvider>
  }
}

beforeEach(() => {
  mockedDelete.mockReset()
  mockedDelete.mockResolvedValue(undefined)
})
afterEach(() => vi.restoreAllMocks())

describe('useDeleteJob', () => {
  it('deletes the job', async () => {
    const { result } = renderHook(() => useDeleteJob(), { wrapper: makeWrapper() })
    result.current.mutate('j1')
    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(mockedDelete).toHaveBeenCalledWith('j1')
  })

  it('treats a 404 as already-gone (success)', async () => {
    mockedDelete.mockRejectedValueOnce(new ApiError(404, 'not found'))
    const { result } = renderHook(() => useDeleteJob(), { wrapper: makeWrapper() })
    result.current.mutate('gone')
    await waitFor(() => expect(result.current.isSuccess).toBe(true))
  })

  it('surfaces a non-404 failure', async () => {
    mockedDelete.mockRejectedValueOnce(new ApiError(500, 'boom'))
    const { result } = renderHook(() => useDeleteJob(), { wrapper: makeWrapper() })
    result.current.mutate('j1')
    await waitFor(() => expect(result.current.isError).toBe(true))
  })

  it('invalidates the jobs query on success', async () => {
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
    const invalidateSpy = vi.spyOn(qc, 'invalidateQueries')
    const wrapper = ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={qc}>{children}</QueryClientProvider>
    )
    const { result } = renderHook(() => useDeleteJob(), { wrapper })
    result.current.mutate('j1')
    await waitFor(() => expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ['jobs'] }))
  })
})

describe('useClearJobs', () => {
  it('clears every id and reports counts, tolerating failures', async () => {
    mockedDelete.mockImplementation(async (id: string) => {
      if (id === 'bad') throw new ApiError(500, 'boom')
      if (id === 'gone') throw new ApiError(404, 'not found') // counts as cleared
      return undefined
    })
    const { result } = renderHook(() => useClearJobs(), { wrapper: makeWrapper() })
    result.current.mutate(['ok', 'gone', 'bad'])
    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(result.current.data).toEqual({ cleared: 2, failed: 1 })
    expect(mockedDelete).toHaveBeenCalledTimes(3)
  })
})
