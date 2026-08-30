import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, fireEvent, render, screen } from '@testing-library/react'
import { useFollowActiveRow } from './useFollowActiveRow'
import { useMediaPlayerStore } from '../lib/mediaPlayerStore'

const clip = { jobId: 'j1', fileName: 'clip.mp4', mediaUrl: '/api/v1/jobs/j1/media?token=t1' }

/** Minimal stand-in for a transcript table: one row carries the ref. */
function Rows({ activeIndex }: { activeIndex: number }) {
  const activeRef = useFollowActiveRow(activeIndex)
  return (
    <ul>
      {['a', 'b', 'c'].map((name, i) => (
        <li key={name} data-testid={name} ref={i === activeIndex ? activeRef : undefined}>
          {name}
        </li>
      ))}
    </ul>
  )
}

afterEach(() => {
  useMediaPlayerStore.getState().clear()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('useFollowActiveRow', () => {
  it('scrolls the newly active row into view', () => {
    const { rerender } = render(<Rows activeIndex={0} />)
    const spy = vi.spyOn(screen.getByTestId('b'), 'scrollIntoView')
    rerender(<Rows activeIndex={1} />)
    expect(spy).toHaveBeenCalledWith({ block: 'nearest', inline: 'nearest', behavior: 'smooth' })
  })

  it('scrolls on mount when a row is already active', () => {
    // Switching to the transcript tab mid-playback must land on the playhead.
    const spy = vi.spyOn(Element.prototype, 'scrollIntoView')
    render(<Rows activeIndex={2} />)
    expect(spy).toHaveBeenCalledTimes(1)
  })

  it('does nothing when no row is active', () => {
    const spy = vi.spyOn(Element.prototype, 'scrollIntoView')
    render(<Rows activeIndex={-1} />)
    expect(spy).not.toHaveBeenCalled()
  })

  it('stops following once the reader scrolls by hand', () => {
    // Reading back over an earlier passage must not be yanked away by the
    // next segment boundary.
    const { rerender } = render(<Rows activeIndex={0} />)
    const spy = vi.spyOn(Element.prototype, 'scrollIntoView')
    fireEvent.wheel(document)
    rerender(<Rows activeIndex={1} />)
    expect(spy).not.toHaveBeenCalled()
  })

  it('stops following on a scroll keypress', () => {
    const { rerender } = render(<Rows activeIndex={0} />)
    const spy = vi.spyOn(Element.prototype, 'scrollIntoView')
    fireEvent.keyDown(document, { key: 'PageDown' })
    rerender(<Rows activeIndex={1} />)
    expect(spy).not.toHaveBeenCalled()
  })

  it('keeps following through a keypress that does not scroll', () => {
    const { rerender } = render(<Rows activeIndex={0} />)
    const spy = vi.spyOn(Element.prototype, 'scrollIntoView')
    fireEvent.keyDown(document, { key: 'a' })
    rerender(<Rows activeIndex={1} />)
    expect(spy).toHaveBeenCalledTimes(1)
  })

  it('resumes following when the player is seeked', () => {
    // Clicking a timestamp is a deliberate "take me there".
    render(<Rows activeIndex={0} />)
    fireEvent.wheel(document)
    const spy = vi.spyOn(screen.getByTestId('a'), 'scrollIntoView')
    act(() => {
      useMediaPlayerStore.getState().open(clip, 5)
    })
    expect(spy).toHaveBeenCalledTimes(1)
  })

  it('jumps without animation when the reader prefers reduced motion', () => {
    vi.stubGlobal('matchMedia', (query: string) => ({
      matches: query.includes('prefers-reduced-motion'),
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
    }))
    const { rerender } = render(<Rows activeIndex={0} />)
    const spy = vi.spyOn(screen.getByTestId('b'), 'scrollIntoView')
    rerender(<Rows activeIndex={1} />)
    expect(spy).toHaveBeenCalledWith({ block: 'nearest', inline: 'nearest', behavior: 'auto' })
  })
})
