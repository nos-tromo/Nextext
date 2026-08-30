import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, fireEvent, render, screen } from '@testing-library/react'
import { useFollowActiveRow } from './useFollowActiveRow'
import { useMediaPlayerStore } from '../lib/mediaPlayerStore'

const clip = { jobId: 'j1', fileName: 'clip.mp4', mediaUrl: '/api/v1/jobs/j1/media?token=t1' }

/** Viewport of the stand-in scroll container, in px from the top of the page. */
const SCROLLER = { top: 0, bottom: 500 }

/** Row geometry: 'a' and 'b' sit inside the viewport, 'c' is below the fold. */
const ROW_RECTS: Record<string, { top: number; bottom: number }> = {
  a: { top: 10, bottom: 40 },
  b: { top: 100, bottom: 130 },
  c: { top: 900, bottom: 930 },
}

function rect(r: { top: number; bottom: number }): DOMRect {
  return { ...r, left: 0, right: 100, width: 100, height: r.bottom - r.top, x: 0, y: r.top, toJSON: () => ({}) } as DOMRect
}

/**
 * Minimal stand-in for AppShell's `<main>` plus a transcript table. happy-dom
 * lays nothing out, so the scroll container and the rows report stubbed
 * geometry, and the container is made scrollable by hand.
 */
function Rows({ activeIndex, rects = ROW_RECTS }: { activeIndex: number; rects?: typeof ROW_RECTS }) {
  const activeRef = useFollowActiveRow(activeIndex)
  // Stubbed in the ref callback: refs attach before any effect runs, so the
  // hook sees a scrollable container the first time it looks for one.
  const stubScroller = (el: HTMLDivElement | null) => {
    if (!el) return
    Object.defineProperty(el, 'scrollHeight', { value: 2000, configurable: true })
    Object.defineProperty(el, 'clientHeight', { value: 500, configurable: true })
    el.getBoundingClientRect = () => rect(SCROLLER)
  }
  return (
    <div>
      <div data-testid="scroller" ref={stubScroller} style={{ overflowY: 'auto' }}>
        <ul>
          {['a', 'b', 'c'].map((name, i) => (
            <li
              key={name}
              data-testid={name}
              ref={(el) => {
                if (el) el.getBoundingClientRect = () => rect(rects[name])
                if (i === activeIndex) activeRef(el)
              }}
            >
              {name}
            </li>
          ))}
        </ul>
      </div>
      <aside data-testid="panel">
        <video data-testid="video" />
        <button type="button" data-testid="button">
          x
        </button>
      </aside>
    </div>
  )
}

afterEach(() => {
  useMediaPlayerStore.getState().clear()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('useFollowActiveRow', () => {
  it('centres a newly active row that is below the fold', () => {
    const { rerender } = render(<Rows activeIndex={0} />)
    const spy = vi.spyOn(screen.getByTestId('c'), 'scrollIntoView')
    rerender(<Rows activeIndex={2} />)
    expect(spy).toHaveBeenCalledWith({ block: 'center', inline: 'nearest', behavior: 'smooth' })
  })

  it('leaves a row that is already on screen alone', () => {
    // Scrolling on every row change would make the page twitch under a reader
    // who can already see the highlight.
    const { rerender } = render(<Rows activeIndex={0} />)
    const spy = vi.spyOn(screen.getByTestId('b'), 'scrollIntoView')
    rerender(<Rows activeIndex={1} />)
    expect(spy).not.toHaveBeenCalled()
  })

  it('scrolls on mount when the active row is off screen', () => {
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

  it('stops following once the reader scrolls the page by hand', () => {
    // Reading back over an earlier passage must not be yanked away by the
    // next segment boundary.
    const { rerender } = render(<Rows activeIndex={0} />)
    const spy = vi.spyOn(Element.prototype, 'scrollIntoView')
    fireEvent.wheel(screen.getByTestId('a'))
    rerender(<Rows activeIndex={2} />)
    expect(spy).not.toHaveBeenCalled()
  })

  it('keeps following through a wheel over the media panel', () => {
    // The panel has its own scroller; scrolling it says nothing about
    // where the reader is on the page.
    const { rerender } = render(<Rows activeIndex={0} />)
    const spy = vi.spyOn(Element.prototype, 'scrollIntoView')
    fireEvent.wheel(screen.getByTestId('panel'))
    rerender(<Rows activeIndex={2} />)
    expect(spy).toHaveBeenCalledTimes(1)
  })

  it('stops following on a scroll keypress', () => {
    const { rerender } = render(<Rows activeIndex={0} />)
    const spy = vi.spyOn(Element.prototype, 'scrollIntoView')
    fireEvent.keyDown(screen.getByTestId('a'), { key: 'PageDown' })
    rerender(<Rows activeIndex={2} />)
    expect(spy).not.toHaveBeenCalled()
  })

  it.each([
    ['video', ' '],
    ['video', 'ArrowRight'],
    ['button', ' '],
  ])('keeps following through %s keys on a %s control', (target, key) => {
    // Space and the arrows drive the player's own controls; they are not
    // the reader scrolling away.
    const { rerender } = render(<Rows activeIndex={0} />)
    const spy = vi.spyOn(Element.prototype, 'scrollIntoView')
    fireEvent.keyDown(screen.getByTestId(target), { key })
    rerender(<Rows activeIndex={2} />)
    expect(spy).toHaveBeenCalledTimes(1)
  })

  it('keeps following through a keypress that does not scroll', () => {
    const { rerender } = render(<Rows activeIndex={0} />)
    const spy = vi.spyOn(Element.prototype, 'scrollIntoView')
    fireEvent.keyDown(screen.getByTestId('a'), { key: 'a' })
    rerender(<Rows activeIndex={2} />)
    expect(spy).toHaveBeenCalledTimes(1)
  })

  it('resumes following when the player is seeked', () => {
    // Clicking a timestamp is a deliberate "take me there".
    const { rerender } = render(<Rows activeIndex={0} />)
    fireEvent.wheel(screen.getByTestId('a'))
    rerender(<Rows activeIndex={2} />)
    const spy = vi.spyOn(screen.getByTestId('c'), 'scrollIntoView')
    act(() => {
      useMediaPlayerStore.getState().open(clip, 5)
    })
    expect(spy).toHaveBeenCalledTimes(1)
  })

  it('resumes following once the playhead row is back on screen', () => {
    // A reader who scrolls back to the highlight has caught up; the pause
    // must not outlive its reason, or following looks broken.
    const { rerender } = render(<Rows activeIndex={0} />)
    fireEvent.wheel(screen.getByTestId('a'))
    const spy = vi.spyOn(Element.prototype, 'scrollIntoView')
    rerender(<Rows activeIndex={1} />) // 'b' is on screen: re-arms, no scroll needed
    expect(spy).not.toHaveBeenCalled()
    rerender(<Rows activeIndex={2} />) // 'c' is off screen: follows again
    expect(spy).toHaveBeenCalledTimes(1)
  })

  it('resumes as soon as the reader scrolls the playhead back into view', () => {
    // Waiting for the next row boundary to re-arm would make following look
    // dead for however long the current segment lasts.
    const { rerender } = render(<Rows activeIndex={0} />)
    fireEvent.wheel(screen.getByTestId('a'))
    const spy = vi.spyOn(Element.prototype, 'scrollIntoView')
    fireEvent.scroll(screen.getByTestId('scroller'))
    rerender(<Rows activeIndex={2} />)
    expect(spy).toHaveBeenCalledTimes(1)
  })

  it('stays paused when the reader scrolls somewhere else', () => {
    const { rerender } = render(<Rows activeIndex={2} />) // 'c' is off screen
    const spy = vi.spyOn(Element.prototype, 'scrollIntoView')
    spy.mockClear()
    fireEvent.wheel(screen.getByTestId('c'))
    fireEvent.scroll(screen.getByTestId('scroller'))
    rerender(<Rows activeIndex={2} rects={{ ...ROW_RECTS, c: { top: 950, bottom: 980 } }} />)
    expect(spy).not.toHaveBeenCalled()
  })

  it('stays paused while the playhead row is off screen', () => {
    const { rerender } = render(<Rows activeIndex={0} />)
    fireEvent.wheel(screen.getByTestId('a'))
    const spy = vi.spyOn(Element.prototype, 'scrollIntoView')
    rerender(<Rows activeIndex={2} />)
    rerender(<Rows activeIndex={1} rects={{ ...ROW_RECTS, b: { top: 700, bottom: 730 } }} />)
    expect(spy).not.toHaveBeenCalled()
  })

  it('jumps without animation when the reader prefers reduced motion', () => {
    vi.stubGlobal('matchMedia', (query: string) => ({
      matches: query.includes('prefers-reduced-motion'),
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
    }))
    const { rerender } = render(<Rows activeIndex={0} />)
    const spy = vi.spyOn(screen.getByTestId('c'), 'scrollIntoView')
    rerender(<Rows activeIndex={2} />)
    expect(spy).toHaveBeenCalledWith({ block: 'center', inline: 'nearest', behavior: 'auto' })
  })
})
