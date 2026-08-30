import { useCallback, useEffect, useRef } from 'react'
import { useMediaPlayerStore } from '../lib/mediaPlayerStore'

/** Keys that scroll a page, and so count as reading by hand. */
const SCROLL_KEYS = new Set([
  'ArrowUp',
  'ArrowDown',
  'ArrowLeft',
  'ArrowRight',
  'PageUp',
  'PageDown',
  'Home',
  'End',
  ' ',
  'Spacebar',
])

/**
 * Controls that consume the scroll keys themselves.
 *
 * Space and the arrows drive the player (play/pause, skip) and the timestamp
 * buttons; pressing them there is not the reader scrolling away.
 */
const INTERACTIVE = 'input, textarea, select, button, a, video, audio, [contenteditable]'

/**
 * Finds the scroll container the row actually sits in.
 *
 * Skips ancestors that merely *could* scroll — the transcript table's
 * `overflow-x-auto` wrapper computes to `overflow-y: auto` but has no
 * vertical overflow — so the search lands on the app shell's `<main>`.
 *
 * @param el - Element to search upward from.
 * @returns The nearest scrollable ancestor, or the document's scrolling
 *   element when there is none.
 */
function findScrollParent(el: HTMLElement): Element | null {
  for (let node = el.parentElement; node; node = node.parentElement) {
    const overflowY = getComputedStyle(node).overflowY
    if ((overflowY === 'auto' || overflowY === 'scroll') && node.scrollHeight > node.clientHeight) {
      return node
    }
  }
  return document.scrollingElement
}

/**
 * Reports whether a row is vertically within the scroller's viewport.
 *
 * @param row - The row to test.
 * @param scroller - The scroll container, or `null` when unresolved.
 * @returns `true` when any part of the row is visible.
 */
function isInView(row: HTMLElement, scroller: Element | null): boolean {
  const view = scroller ? scroller.getBoundingClientRect() : null
  const top = view?.top ?? 0
  const bottom = view?.bottom ?? window.innerHeight
  const rect = row.getBoundingClientRect()
  return rect.bottom > top && rect.top < bottom
}

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
 * that row only. Whenever the active row changes and has left the viewport,
 * it is scrolled to the middle — one jump per screenful rather than a nudge
 * per row, and the reader gets the lines either side of the playhead as
 * context. A row still on screen is left where it is.
 *
 * Following pauses while the reader scrolls the page by hand, and resumes on
 * either of two signals: a deliberate jump (`open`/`seek` in
 * {@link useMediaPlayerStore} — a timestamp click), or the playhead's row
 * coming back into view on its own. That second signal is what keeps the
 * pause from outliving its reason: a one-way switch would leave following
 * dead for the rest of the session after a single trackpad flick.
 *
 * @param activeIndex - Index of the row under the playhead, or `-1` when
 *   nothing is playing for this view.
 * @returns A ref callback for the active row.
 */
export function useFollowActiveRow(activeIndex: number): (el: HTMLElement | null) => void {
  const followSeq = useMediaPlayerStore((s) => s.followSeq)
  const rowRef = useRef<HTMLElement | null>(null)
  const scrollerRef = useRef<Element | null>(null)
  const followingRef = useRef(true)

  const setRow = useCallback((el: HTMLElement | null) => {
    rowRef.current = el
  }, [])

  // Resolved on first use rather than when the row attaches: `findScrollParent`
  // measures `scrollHeight`/`clientHeight`, which are only meaningful once the
  // commit that attached the row has been laid out.
  const resolveScroller = useCallback((): Element | null => {
    if (!scrollerRef.current && rowRef.current) {
      scrollerRef.current = findScrollParent(rowRef.current)
    }
    return scrollerRef.current
  }, [])

  // Declared before the scrolling effect so a deliberate jump has re-engaged
  // following by the time that effect reads it.
  useEffect(() => {
    followingRef.current = true
  }, [followSeq])

  useEffect(() => {
    const pausedBy = (target: EventTarget | null): boolean => {
      const scroller = resolveScroller()
      // Before the first row attaches there is no scroller to compare
      // against; treat the event as page scrolling.
      if (!scroller || !(target instanceof Node)) return true
      return scroller.contains(target)
    }
    const onScrollInput = (event: Event) => {
      if (pausedBy(event.target)) followingRef.current = false
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (!SCROLL_KEYS.has(event.key)) return
      const target = event.target
      if (target instanceof Element && target.closest(INTERACTIVE)) return
      if (pausedBy(target)) followingRef.current = false
    }
    // Scrolling the playhead's row back into view says the reader has caught
    // up, so following resumes there and then rather than at the next row
    // boundary — which could be a whole segment away.
    const onScroll = () => {
      if (followingRef.current) return
      const row = rowRef.current
      if (row && isInView(row, resolveScroller())) followingRef.current = true
    }
    // `scroll` does not bubble, so listen in the capture phase.
    document.addEventListener('scroll', onScroll, { passive: true, capture: true })
    document.addEventListener('wheel', onScrollInput, { passive: true })
    document.addEventListener('touchmove', onScrollInput, { passive: true })
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('wheel', onScrollInput)
      document.removeEventListener('touchmove', onScrollInput)
      document.removeEventListener('scroll', onScroll, { capture: true })
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [resolveScroller])

  useEffect(() => {
    const row = rowRef.current
    if (activeIndex < 0 || !row) return
    const onScreen = isInView(row, resolveScroller())
    // The reader has caught up with the playhead: following is wanted again.
    if (onScreen) {
      followingRef.current = true
      return
    }
    if (!followingRef.current) return
    row.scrollIntoView({ block: 'center', inline: 'nearest', behavior: scrollBehavior() })
  }, [activeIndex, followSeq, resolveScroller])

  return setRow
}
