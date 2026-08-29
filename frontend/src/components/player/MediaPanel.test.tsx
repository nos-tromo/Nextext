import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, fireEvent, render, screen } from '@testing-library/react'
import { MediaPanel } from './MediaPanel'
import { useMediaPlayerStore } from '../../lib/mediaPlayerStore'

/** Drive the store from outside React; act() flushes the re-render. */
function dispatch(fn: () => void) {
  act(fn)
}

const video = { jobId: 'j1', fileName: 'clip.mp4', mediaUrl: '/api/v1/jobs/j1/media?token=t1' }
const audio = { jobId: 'j2', fileName: 'talk.wav', mediaUrl: '/api/v1/jobs/j2/media?token=t2' }

beforeEach(() => {
  useMediaPlayerStore.getState().clear()
  // happy-dom implements neither playback nor seeking; stub the surface the
  // panel touches so effects run without unhandled promise rejections.
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined)
  vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {})
})

afterEach(() => {
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

  it('uses a video element for a video upload', () => {
    const { container } = render(<MediaPanel />)
    dispatch(() => useMediaPlayerStore.getState().open(video))
    const el = container.querySelector('video')
    expect(el).not.toBeNull()
    expect(el).toHaveAttribute('src', video.mediaUrl)
  })

  it('uses an audio element for an audio upload', () => {
    // A <video> for an audio file renders a large black rectangle; the
    // element has to follow the file, not the panel.
    const { container } = render(<MediaPanel />)
    dispatch(() => useMediaPlayerStore.getState().open(audio))
    expect(container.querySelector('audio')).not.toBeNull()
    expect(container.querySelector('video')).toBeNull()
  })

  it('seeks the element to the requested timestamp', () => {
    const { container } = render(<MediaPanel />)
    dispatch(() => useMediaPlayerStore.getState().open(video, 42))
    const el = container.querySelector('video') as HTMLVideoElement
    fireEvent.loadedMetadata(el)
    expect(el.currentTime).toBe(42)
  })

  it('applies a seek that arrives after metadata is ready', () => {
    const { container } = render(<MediaPanel />)
    dispatch(() => useMediaPlayerStore.getState().open(video))
    const el = container.querySelector('video') as HTMLVideoElement
    fireEvent.loadedMetadata(el)
    dispatch(() => useMediaPlayerStore.getState().seek(77))
    expect(el.currentTime).toBe(77)
  })

  it('reports the playhead to the store', () => {
    const { container } = render(<MediaPanel />)
    dispatch(() => useMediaPlayerStore.getState().open(video))
    const el = container.querySelector('video') as HTMLVideoElement
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

  it('explains an unplayable file instead of showing a dead player', () => {
    // Browsers refuse containers they have no decoder for (Firefox and .mkv,
    // say) — silently rendering a broken element would look like our bug.
    const { container } = render(<MediaPanel />)
    dispatch(() => useMediaPlayerStore.getState().open(video))
    fireEvent.error(container.querySelector('video') as HTMLVideoElement)
    expect(screen.getByText(/can’t be played|cannot be played/i)).toBeInTheDocument()
  })

  it('clears the error when another recording is opened', () => {
    const { container } = render(<MediaPanel />)
    dispatch(() => useMediaPlayerStore.getState().open(video))
    fireEvent.error(container.querySelector('video') as HTMLVideoElement)
    dispatch(() => useMediaPlayerStore.getState().open(audio))
    expect(screen.queryByText(/can’t be played|cannot be played/i)).not.toBeInTheDocument()
  })
})
