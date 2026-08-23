import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Shell } from './Shell'
import { useJobProgressStore } from '../../lib/jobProgressStore'
import type { JobListItem } from '../../api/types'

// The Shell mounts the single owner-multiplexed SSE stream; mock it so the
// stream stays open (never yields) and never touches the network in Shell tests.
const { streamSseMock } = vi.hoisted(() => ({ streamSseMock: vi.fn() }))
vi.mock('../../api/sse', () => ({ streamSse: streamSseMock }))

function stubJobs(
  jobs: JobListItem[],
  whoami: { username: string; display_name: string | null } | null = null,
): void {
  const jobsUrl = '/jobs'
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.includes('/version')) {
        return new Response(JSON.stringify({ version: '1.2.3' }), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        })
      }
      if (url.includes('/whoami')) {
        if (!whoami) {
          return new Response('{}', { status: 401, headers: { 'content-type': 'application/json' } })
        }
        return new Response(JSON.stringify(whoami), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        })
      }
      if (url.includes(jobsUrl)) {
        return new Response(JSON.stringify({ jobs }), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        })
      }
      return new Response('{}', { status: 200, headers: { 'content-type': 'application/json' } })
    }),
  )
}

function mountShell() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <Shell>
        <div>page-body</div>
      </Shell>
    </QueryClientProvider>,
  )
}

beforeEach(() => {
  useJobProgressStore.getState().clear()
  streamSseMock.mockReset()
  // Default: an owner stream that opens and stays open. It yields one inert
  // (non-job) frame the hook ignores, then blocks — modelling a live SSE
  // connection without emitting real progress into these layout tests.
  streamSseMock.mockImplementation(async function* () {
    yield { event: 'ping', data: '' }
    await new Promise(() => {})
  })
})
afterEach(() => vi.restoreAllMocks())

describe('Shell', () => {
  it('renders exactly one header row with the app title and its children', () => {
    stubJobs([])
    mountShell()
    // A single AppHeader carries the title; the shell no longer duplicates
    // it in a second header row.
    expect(screen.getAllByText('Nextext')).toHaveLength(1)
    expect(screen.getByText('page-body')).toBeInTheDocument()
  })

  it('renders the fetched version in the AppHeader', async () => {
    stubJobs([])
    mountShell()
    expect(await screen.findByText('v1.2.3')).toBeInTheDocument()
  })

  it('opens exactly one owner-multiplexed job stream', () => {
    stubJobs([])
    mountShell()
    expect(streamSseMock).toHaveBeenCalledTimes(1)
    expect(streamSseMock.mock.calls[0][0]).toBe('/jobs/events')
  })

  it('renders the chrome header and the scrolling canvas main', () => {
    stubJobs([])
    mountShell()
    expect(screen.getByRole('banner')).toBeInTheDocument()
    expect(screen.getByRole('main')).toHaveTextContent('page-body')
  })

  it('shows the display name in the header when whoami resolves', async () => {
    stubJobs([], { username: 'alice', display_name: 'Alice Example' })
    mountShell()
    expect(await screen.findByRole('button', { name: /Alice Example/ })).toBeInTheDocument()
  })

  it('falls back to the username when no display name is set', async () => {
    stubJobs([], { username: 'alice', display_name: null })
    mountShell()
    expect(await screen.findByRole('button', { name: /alice/ })).toBeInTheDocument()
  })

  it('omits the identity slot when whoami fails (dev, no gateway)', () => {
    stubJobs([], null)
    mountShell()
    expect(screen.queryByText('alice')).not.toBeInTheDocument()
  })

  it('lets the padded canvas grow past the fold so its bottom padding renders', () => {
    stubJobs([])
    mountShell()
    const canvas = screen.getByRole('main').firstElementChild
    // `h-full` would pin the canvas to exactly one viewport slice, stranding
    // `p-8`'s bottom half at the fold instead of below the last job card —
    // an ancestor's padding is not part of the scroll container's overflow
    // region. `min-h-full` still fills a short page but grows with content.
    expect(canvas).toHaveClass('min-h-full', 'p-8')
    expect(canvas).not.toHaveClass('h-full')
    expect(canvas).not.toHaveClass('min-h-0')
  })
})
