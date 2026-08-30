import { useCallback, useEffect, useRef } from 'react'
import { useMediaPlayerStore } from '../lib/mediaPlayerStore'

/** Keys that scroll a page, and so count as reading by hand. */
const SCROLL_KEYS = new Set([
  'ArrowUp',
  'ArrowDown',
  'PageUp',
  'PageDown',
  'Home',
  'End',
  ' ',
  'Spacebar',
])

/**
 * Resolves the scroll animation to the reader's motion preference.
 *
 * @returns `'auto'` when reduced motion is requested, otherwise `'smooth'`.
 */
function scrollBehavior(): ScrollBehavior {
  const query = typeof window !== 'undefined' && typeof window.matchMedia === 'function'
    ? window.matchMedia('(prefers-reduced-motion: reduce)')
    : null
  return query?.matches ? 'auto' : 'smooth'
}

/**
 * Keeps the row under the playhead in view as playback advances.
 *
 * The caller computes which row is active and attaches the returned ref to
 * that row only; whenever the active row changes, it is scrolled into view.
 * `block: 'nearest'` means a row already on screen is left alone, so the page
 * moves only when it has to.
 *
 * Following is suspended as soon as the reader scrolls by hand — a wheel,
 * touch drag, or scroll keypress anywhere on the page — and resumes on the
 * next deliberate jump (`open`/`seek` in {@link useMediaPlayerStore}, i.e. a
 * timestamp click). Listening for the input events rather than for `scroll`
 * is what keeps our own programmatic scrolling from cancelling itself. The
 * listeners sit on the document rather than on the table, so scrolling with
 * the pointer anywhere over the page counts.
 *
 * @param activeIndex - Index of the row under the playhead, or `-1` when
 *   nothing is playing for this view.
 * @returns A ref callback for the active row.
 */
export function useFollowActiveRow(activeIndex: number): (el: HTMLElement | null) => void {
  const followSeq = useMediaPlayerStore((s) => s.followSeq)
  const rowRef = useRef<HTMLElement | null>(null)
  const followingRef = useRef(true)

  const setRow = useCallback((el: HTMLElement | null) => {
    rowRef.current = el
  }, [])

  // Declared before the scrolling effect so a deliberate jump has re-engaged
  // following by the time that effect reads it.
  useEffect(() => {
    followingRef.current = true
  }, [followSeq])

  useEffect(() => {
    const pause = () => {
      followingRef.current = false
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (SCROLL_KEYS.has(event.key)) pause()
    }
    document.addEventListener('wheel', pause, { passive: true })
    document.addEventListener('touchmove', pause, { passive: true })
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('wheel', pause)
      document.removeEventListener('touchmove', pause)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [])

  useEffect(() => {
    const row = rowRef.current
    if (!followingRef.current || activeIndex < 0 || !row) return
    row.scrollIntoView({ block: 'nearest', inline: 'nearest', behavior: scrollBehavior() })
  }, [activeIndex, followSeq])

  return setRow
}
