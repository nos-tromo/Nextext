import { afterEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

vi.mock('../api/sse', () => ({
  // eslint-disable-next-line require-yield
  async *streamSse() {},
}))

import { Home } from './Home'

function mountHome() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <Home />
    </QueryClientProvider>,
  )
}

afterEach(() => vi.restoreAllMocks())

describe('Home', () => {
  it('re-discovers jobs on mount and renders them', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(
          JSON.stringify({
            jobs: [{ job_id: 'j1', status: 'completed', file_name: 'clip.wav', progress: 1, created_at: 't', task: 'transcribe' }],
          }),
          { status: 200, headers: { 'content-type': 'application/json' } },
        ),
      ),
    )
    mountHome()
    expect(screen.getByText('New job')).toBeInTheDocument()
    await waitFor(() => expect(screen.getByText('clip.wav')).toBeInTheDocument())
  })
})
