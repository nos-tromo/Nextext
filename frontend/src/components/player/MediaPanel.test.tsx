import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { clearAllPurposes, setPurpose } from '@infra/ui'
import { MediaPanel, REGRANT_INTERVAL_MS } from './MediaPanel'
import { PURPOSE_SCOPE } from '../../api/client'
import { useMediaPlayerStore } from '../../lib/mediaPlayerStore'

/** Drive the store from outside React; act() flushes the re-render. */
function dispatch(fn: () => void) {
  act(fn)
}

const video = { jobId: 'j1', fileName: 'clip.mp4', mediaUrl: '/api/v1/jobs/j1/media?token=t1' }
const audio = { jobId: 'j2', fileName: 'talk.wav', mediaUrl: '/api/v1/jobs/j2/media?token=t2' }

/** The grant URL the stubbed backend hands out for a job's nth grant. */
const grantUrl = (jobId: string, n: number) => `/api/v1/jobs/${jobId}/media?grant=g${n}`

/** Every POST …/media-grant answered with a fresh grant; other requests 404. */
let grantRequests: string[]
let grantStatus: number
function stubGrants() {
  grantRequests = []
  grantStatus = 201
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      const match = /\/jobs\/([^/]+)\/media-grant$/.exec(url)
      if (!match || init?.method !== 'POST') return new Response('{}', { status: 404 })
      if (grantStatus !== 201) return new Response('{}', { status: grantStatus })
      grantRequests.push(match[1])
      const n = grantRequests.filter((id) => id === match[1]).length
      return new Response(
        JSON.stringify({ media_url: grantUrl(match[1], n), expires_at: '2026-01-02T03:19:05Z' }),
        { status: 201, headers: { 'content-type': 'application/json' } },
      )
    }),
  )
}

/** Wait for the panel's media element, which appears once its grant arrived. */
async function mediaElement(container: HTMLElement, tag: 'video' | 'audio' = 'video', src?: string) {
  let found: HTMLMediaElement | null = null
  await waitFor(() => {
    found = container.querySelector<HTMLMediaElement>(tag)
    expect(found).not.toBeNull()
    if (src !== undefined) expect(found).toHaveAttribute('src', src)
  })
  return found as unknown as HTMLMediaElement
}

beforeEach(() => {
  useMediaPlayerStore.getState().clear()
  stubGrants()
  // happy-dom implements neither playback nor seeking; stub the surface the
  // panel touches so effects run without unhandled promise rejections.
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined)
  vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {})
})

afterEach(() => {
  clearAllPurposes()
  sessionStorage.clear()
  vi.restoreAllMocks()
  useMediaPlayerStore.getState().clear()
})

describe('MediaPanel', () => {
  it('renders nothing visible until a recording is opened', () => {
    const { container } = render(<MediaPanel />)
    const panel = container.querySelector('aside')
    // Kept mounted (so the slide transition has something to animate) but
    // hidden from assistive tech and taken out of the tab order.
    expect(panel).not.toBeNull()
    expect(panel).toHaveAttribute('aria-hidden', 'true')
  })

  it('reveals the panel when a recording is opened', () => {
    const { container } = render(<MediaPanel />)
    dispatch(() => useMediaPlayerStore.getState().open(video))
    expect(container.querySelector('aside')).not.toHaveAttribute('aria-hidden', 'true')
  })

  it('shows the recording name', () => {
    render(<MediaPanel />)
    dispatch(() => useMediaPlayerStore.getState().open(video))
    expect(screen.getByText('clip.mp4')).toBeInTheDocument()
  })

  it('uses a video element for a video upload', async () => {
    const { container } = render(<MediaPanel />)
    dispatch(() => useMediaPlayerStore.getState().open(video))
    await mediaElement(container, 'video', grantUrl('j1', 1))
  })

  it('uses an audio element for an audio upload', async () => {
    // A <video> for an audio file renders a large black rectangle; the
    // element has to follow the file, not the panel.
    const { container } = render(<MediaPanel />)
    dispatch(() => useMediaPlayerStore.getState().open(audio))
    await mediaElement(container, 'audio')
    expect(container.querySelector('video')).toBeNull()
  })

  it('seeks the element to the requested timestamp', async () => {
    const { container } = render(<MediaPanel />)
    dispatch(() => useMediaPlayerStore.getState().open(video, 42))
    const el = await mediaElement(container)
    fireEvent.loadedMetadata(el)
    expect(el.currentTime).toBe(42)
  })

  it('plays from a timestamp clicked before its grant arrived', async () => {
    // The element only exists once the grant is there, so the click's seek
    // lands on `loadedmetadata` — and must still start playback.
    const play = vi.mocked(HTMLMediaElement.prototype.play)
    const { container } = render(<MediaPanel />)
    dispatch(() => useMediaPlayerStore.getState().open(video, 42))
    const el = await mediaElement(container)
    play.mockClear()
    fireEvent.loadedMetadata(el)
    expect(el.currentTime).toBe(42)
    expect(play).toHaveBeenCalledOnce()
  })

  it('opens without playing when no timestamp was clicked', async () => {
    const play = vi.mocked(HTMLMediaElement.prototype.play)
    const { container } = render(<MediaPanel />)
    dispatch(() => useMediaPlayerStore.getState().open(video))
    const el = await mediaElement(container)
    play.mockClear()
    fireEvent.loadedMetadata(el)
    expect(play).not.toHaveBeenCalled()
  })

  it('applies a seek that arrives after metadata is ready', async () => {
    const { container } = render(<MediaPanel />)
    dispatch(() => useMediaPlayerStore.getState().open(video))
    const el = await mediaElement(container)
    fireEvent.loadedMetadata(el)
    dispatch(() => useMediaPlayerStore.getState().seek(77))
    expect(el.currentTime).toBe(77)
  })

  it('reports the playhead to the store', async () => {
    const { container } = render(<MediaPanel />)
    dispatch(() => useMediaPlayerStore.getState().open(video))
    const el = await mediaElement(container)
    el.currentTime = 12
    fireEvent.timeUpdate(el)
    expect(useMediaPlayerStore.getState().currentTime).toBe(12)
  })

  it('closes from the close button', () => {
    render(<MediaPanel />)
    dispatch(() => useMediaPlayerStore.getState().open(video))
    fireEvent.click(screen.getByRole('button', { name: 'Close player' }))
    expect(useMediaPlayerStore.getState().session).toBeNull()
  })

  it('closes on Escape', () => {
    render(<MediaPanel />)
    dispatch(() => useMediaPlayerStore.getState().open(video))
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(useMediaPlayerStore.getState().session).toBeNull()
  })

  it('returns focus to whatever opened it', () => {
    // Closing an overlay that dumps focus on <body> strands keyboard users
    // at the top of the page.
    const opener = document.createElement('button')
    document.body.appendChild(opener)
    opener.focus()

    render(<MediaPanel />)
    dispatch(() => useMediaPlayerStore.getState().open(video))
    fireEvent.keyDown(document, { key: 'Escape' })

    expect(document.activeElement).toBe(opener)
    opener.remove()
  })

  it('explains an unplayable file instead of showing a dead player', async () => {
    // Browsers refuse containers they have no decoder for (Firefox and .mkv,
    // say) — silently rendering a broken element would look like our bug.
    // The first failure could be an expired grant, so it costs one fresh
    // grant; the second, right after, is the file's.
    const { container } = render(<MediaPanel />)
    dispatch(() => useMediaPlayerStore.getState().open(video))
    fireEvent.error(await mediaElement(container, 'video', grantUrl('j1', 1)))
    fireEvent.error(await mediaElement(container, 'video', grantUrl('j1', 2)))
    expect(await screen.findByText(/can’t be played|cannot be played/i)).toBeInTheDocument()
    expect(grantRequests).toEqual(['j1', 'j1'])
  })

  it('clears the error when another recording is opened', async () => {
    const { container } = render(<MediaPanel />)
    dispatch(() => useMediaPlayerStore.getState().open(video))
    fireEvent.error(await mediaElement(container, 'video', grantUrl('j1', 1)))
    fireEvent.error(await mediaElement(container, 'video', grantUrl('j1', 2)))
    await screen.findByText(/can’t be played|cannot be played/i)
    dispatch(() => useMediaPlayerStore.getState().open(audio))
    expect(screen.queryByText(/can’t be played|cannot be played/i)).not.toBeInTheDocument()
    await mediaElement(container, 'audio', grantUrl('j2', 1))
  })
})

describe('MediaPanel playback grant', () => {
  it('asks for a grant before it sets src, and plays the URL the grant names', async () => {
    const { container } = render(<MediaPanel />)
    dispatch(() => useMediaPlayerStore.getState().open(video))
    // No element — and so no request the purpose could not ride on — until
    // the grant is there.
    expect(container.querySelector('video')).toBeNull()
    expect(screen.getAllByText('Loading the recording…').length).toBeGreaterThan(0)

    await mediaElement(container, 'video', grantUrl('j1', 1))
    const [url, init] = vi.mocked(fetch).mock.calls[0] as unknown as [string, RequestInit]
    expect(url).toBe('/api/v1/jobs/j1/media-grant')
    expect(init.method).toBe('POST')
    // The snapshot's token URL is never played.
    expect(container.querySelector('video')?.getAttribute('src')).not.toContain('token=')
  })

  it('replaces an expired grant once and picks playback up where it stopped', async () => {
    const { container } = render(<MediaPanel />)
    dispatch(() => useMediaPlayerStore.getState().open(video, 10))
    const first = await mediaElement(container, 'video', grantUrl('j1', 1))
    // Mid-playback: metadata loaded, playing at 95 s, when a range request
    // past the grant's lifetime is refused.
    Object.defineProperty(first, 'readyState', { configurable: true, value: 1 /* HAVE_METADATA */ })
    first.currentTime = 95
    Object.defineProperty(first, 'paused', { configurable: true, value: false })
    const play = vi.mocked(HTMLMediaElement.prototype.play)
    play.mockClear()
    fireEvent.error(first)

    const second = await mediaElement(container, 'video', grantUrl('j1', 2))
    fireEvent.loadedMetadata(second)
    expect(second.currentTime).toBe(95) // not back to the 10 s timestamp
    expect(play).toHaveBeenCalled()
    expect(grantRequests).toEqual(['j1', 'j1'])
  })

  it('a timestamp clicked while the fresh grant loads wins over the resume point', async () => {
    const { container } = render(<MediaPanel />)
    dispatch(() => useMediaPlayerStore.getState().open(video))
    const first = await mediaElement(container, 'video', grantUrl('j1', 1))
    Object.defineProperty(first, 'readyState', { configurable: true, value: 1 /* HAVE_METADATA */ })
    first.currentTime = 95
    fireEvent.error(first)
    dispatch(() => useMediaPlayerStore.getState().seek(30))

    const second = await mediaElement(container, 'video', grantUrl('j1', 2))
    fireEvent.loadedMetadata(second)
    expect(second.currentTime).toBe(30)
  })

  it('asks again on a later expiry, a grant lifetime on', async () => {
    let now = 1_000
    vi.spyOn(performance, 'now').mockImplementation(() => now)
    const { container } = render(<MediaPanel />)
    dispatch(() => useMediaPlayerStore.getState().open(video))
    fireEvent.error(await mediaElement(container, 'video', grantUrl('j1', 1)))
    const second = await mediaElement(container, 'video', grantUrl('j1', 2))
    fireEvent.loadedMetadata(second)
    now += REGRANT_INTERVAL_MS

    fireEvent.error(second)

    await mediaElement(container, 'video', grantUrl('j1', 3))
    expect(screen.queryByText(/can’t be played|cannot be played/i)).not.toBeInTheDocument()
  })

  it('counts a repeated error from the same element once', async () => {
    const { container } = render(<MediaPanel />)
    dispatch(() => useMediaPlayerStore.getState().open(video))
    const first = await mediaElement(container, 'video', grantUrl('j1', 1))
    fireEvent.error(first)
    fireEvent.error(first)
    await mediaElement(container, 'video', grantUrl('j1', 2))
    expect(screen.queryByText(/can’t be played|cannot be played/i)).not.toBeInTheDocument()
  })

  it('says a purpose is needed when the grant is refused for want of one, and plays once one is stated', async () => {
    // No purpose bar is mounted to ask with, so the 428 stands.
    grantStatus = 428
    const { container } = render(<MediaPanel />)
    dispatch(() => useMediaPlayerStore.getState().open(video))
    expect(await screen.findByText(/Playing this recording needs a stated purpose/)).toBeInTheDocument()
    expect(container.querySelector('video')).toBeNull()

    // Stated in the header bar: the panel asks again, and the grant plays.
    grantStatus = 201
    act(() => {
      setPurpose(PURPOSE_SCOPE, { caseRef: 'case-0001' })
    })
    await mediaElement(container, 'video', grantUrl('j1', 1))
    expect(screen.queryByText(/needs a stated purpose/)).not.toBeInTheDocument()
  })

  it('reports other grant failures as an error, not as a missing purpose', async () => {
    grantStatus = 404
    render(<MediaPanel />)
    dispatch(() => useMediaPlayerStore.getState().open(video))
    expect(await screen.findByText(/Something went wrong \(404\)/)).toBeInTheDocument()
  })

  it('drops the grant of a recording that is no longer open', async () => {
    const { container } = render(<MediaPanel />)
    dispatch(() => useMediaPlayerStore.getState().open(video))
    dispatch(() => useMediaPlayerStore.getState().open(audio))
    await mediaElement(container, 'audio', grantUrl('j2', 1))
    await new Promise((resolve) => setTimeout(resolve, 20))
    expect(container.querySelector('video')).toBeNull()
    expect(container.querySelector('audio')).toHaveAttribute('src', grantUrl('j2', 1))
  })
})

describe('MediaPanel chrome', () => {
  it('gives the header band the app chrome colour and height', () => {
    // The panel's header sits beside the AppShell header (`h-12`, on
    // `bg-chrome`); a different colour or height there reads as a stray
    // block in the top-right corner rather than one continuous bar.
    const { container } = render(<MediaPanel />)
    dispatch(() => useMediaPlayerStore.getState().open(video))
    const band = container.querySelector('aside > div') as HTMLElement
    expect(band.className).toContain('h-12')
    expect(band.className).toContain('bg-chrome')
  })

  it('rules the body rather than the band, so the seam lines up', () => {
    // `border-box` sizing draws a `border-b` *inside* the band's 48px, one
    // pixel above the AppShell's own rule (a borderless h-12 header over a
    // `border-t` main). Carrying it on the body keeps the line continuous.
    const { container } = render(<MediaPanel />)
    dispatch(() => useMediaPlayerStore.getState().open(video))
    const [band, body] = Array.from(
      (container.querySelector('aside') as HTMLElement).children,
    ) as HTMLElement[]
    expect(band.className).not.toContain('border-b')
    expect(body.className).toContain('border-t')
  })

  it('separates the panel by its shadow, not a hard left border', () => {
    // `border-l` drew a grey line the full height of the viewport, cutting
    // the header bar in two; the shadow gives a smoother transition.
    const { container } = render(<MediaPanel />)
    const panel = container.querySelector('aside') as HTMLElement
    expect(panel.className).not.toContain('border-l')
    expect(panel.className).toContain('shadow-lg')
  })
})
