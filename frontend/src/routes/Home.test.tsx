import { afterEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

vi.mock('../api/sse', () => ({
  async *streamSse() {},
}))

import { Home } from './Home'
import { LanguageContext } from '../i18n/LanguageContext'

function mountHome() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <Home />
    </QueryClientProvider>,
  )
}

function stubEmptyJobs() {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => new Response(JSON.stringify({ jobs: [] }), { status: 200, headers: { 'content-type': 'application/json' } })),
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

describe('Home German locale', () => {
  it('renders the "New job" heading in German', () => {
    stubEmptyJobs()
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={qc}>
        <LanguageContext.Provider value="de">
          <Home />
        </LanguageContext.Provider>
      </QueryClientProvider>,
    )
    expect(screen.getByText('Neuer Auftrag')).toBeInTheDocument()
  })
})
