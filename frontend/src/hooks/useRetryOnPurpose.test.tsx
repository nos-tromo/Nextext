import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { clearAllPurposes, setPurpose } from '@infra/ui'
import { ApiError, PURPOSE_SCOPE } from '../api/client'
import { useRetryOnPurpose } from './useRetryOnPurpose'

afterEach(() => {
  clearAllPurposes()
  sessionStorage.clear()
})

function wrapper(client: QueryClient) {
  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  )
}

describe('useRetryOnPurpose', () => {
  it('refetches a query a missing purpose refused once a purpose is stated, and no other', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const refused = vi.fn().mockRejectedValueOnce(new ApiError(428, null)).mockResolvedValue('jobs')
    const broken = vi.fn().mockRejectedValue(new ApiError(500, null))

    const { result } = renderHook(
      () => {
        useRetryOnPurpose()
        return {
          refused: useQuery({ queryKey: ['refused'], queryFn: refused }),
          broken: useQuery({ queryKey: ['broken'], queryFn: broken }),
        }
      },
      { wrapper: wrapper(client) },
    )
    await waitFor(() => expect(result.current.refused.isError && result.current.broken.isError).toBe(true))

    act(() => {
      setPurpose(PURPOSE_SCOPE, { caseRef: 'case-0001' })
    })

    await waitFor(() => expect(result.current.refused.data).toBe('jobs'))
    expect(refused).toHaveBeenCalledTimes(2)
    expect(broken).toHaveBeenCalledTimes(1)
  })

  it('does nothing when the purpose is cleared', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const refused = vi.fn().mockRejectedValue(new ApiError(428, null))
    const { result } = renderHook(
      () => {
        useRetryOnPurpose()
        return useQuery({ queryKey: ['refused'], queryFn: refused })
      },
      { wrapper: wrapper(client) },
    )
    await waitFor(() => expect(result.current.isError).toBe(true))
    setPurpose(PURPOSE_SCOPE, { caseRef: 'case-0001' })
    await waitFor(() => expect(refused).toHaveBeenCalledTimes(2))

    act(() => {
      clearAllPurposes()
    })
    await new Promise((resolve) => setTimeout(resolve, 20))
    expect(refused).toHaveBeenCalledTimes(2)
  })
})
