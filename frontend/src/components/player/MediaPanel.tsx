import { useEffect, useRef, useState } from 'react'
import { IconButton, XIcon } from '@infra/ui'
import { useMediaPlayerStore } from '../../lib/mediaPlayerStore'
import { mediaSrcUrl } from '../../lib/mediaSrc'
import { useT } from '../../i18n/LanguageContext'

/** Upload extensions that carry a picture and therefore need a `<video>`. */
const VIDEO_EXTENSIONS = new Set(['mp4', 'mkv', 'webm', 'mov', 'avi', 'm4v'])

/**
 * Decide whether a recording needs a video surface.
 *
 * Driven by the filename rather than the served MIME type: the element has to
 * exist before the first byte arrives, and a `<video>` for an audio file
 * renders as a large black rectangle.
 *
 * @param fileName - Original upload name.
 * @returns True when the file is expected to carry a picture.
 */
function isVideoFile(fileName: string): boolean {
  return VIDEO_EXTENSIONS.has(fileName.split('.').pop()?.toLowerCase() ?? '')
}

/**
 * The app-wide media player: a non-modal panel that slides in from the right.
 *
 * Mounted once in the Shell and driven by {@link useMediaPlayerStore}, so any
 * transcript row or frame caption can start playback at its own timestamp
 * without owning player state.
 *
 * Its header band carries the app chrome's own colour and height, so it
 * continues the AppShell header rather than sitting beside it as a separate
 * block; the panel is set off by its shadow alone, with no left border to cut
 * that bar in two.
 *
 * Deliberately **not** a dialog: no backdrop and no `aria-modal`, because the
 * point is that the jobs list and the result tabs stay usable while a
 * recording plays. It keeps only the parts of the overlay contract that still
 * apply — Escape closes, and focus returns to whatever opened it.
 *
 * The element stays mounted while closed so the slide has something to
 * animate; `aria-hidden` plus `inert` keep it out of the accessibility tree
 * and the tab order in that state.
 */
export function MediaPanel() {
  const t = useT()
  const session = useMediaPlayerStore((s) => s.session)
  const seekRequest = useMediaPlayerStore((s) => s.seekRequest)
  const close = useMediaPlayerStore((s) => s.close)
  const setCurrentTime = useMediaPlayerStore((s) => s.setCurrentTime)

  const mediaRef = useRef<HTMLVideoElement | HTMLAudioElement | null>(null)
  const panelRef = useRef<HTMLElement | null>(null)
  const openerRef = useRef<HTMLElement | null>(null)
  const [failed, setFailed] = useState(false)

  const open = session !== null
  // The store holds the backend's server-relative URL; the element needs one
  // resolved against the sub-path the SPA is mounted under.
  const src = session ? mediaSrcUrl(session.mediaUrl) : undefined

  // Escape to close, plus focus return — the docint PreviewDialog contract
  // minus the modal parts.
  useEffect(() => {
    if (!open) return
    openerRef.current = document.activeElement as HTMLElement | null
    panelRef.current?.focus()

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') close()
    }
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('keydown', onKeyDown)
      openerRef.current?.focus?.()
      openerRef.current = null
    }
  }, [open, close])

  // A new recording starts clean: last one's playback error is not this one's.
  useEffect(() => setFailed(false), [session?.mediaUrl])

  // Apply seeks. Setting `currentTime` before metadata is loaded is discarded
  // by the browser, so a request that arrives early is re-applied on
  // `loadedmetadata` (see `onLoadedMetadata` below).
  useEffect(() => {
    const element = mediaRef.current
    if (!element || !seekRequest) return
    element.currentTime = seekRequest.seconds
    void element.play?.()?.catch?.(() => {
      // Autoplay can be refused until the user interacts with the page; the
      // seek still landed, so leave the player paused at the right spot.
    })
  }, [seekRequest])

  return (
    <aside
      ref={panelRef}
      role="complementary"
      aria-label={session ? session.fileName : t('player.title')}
      aria-hidden={open ? undefined : true}
      // `inert` keeps the closed panel's controls off the tab path. React 19
      // renders the boolean attribute natively.
      inert={open ? undefined : true}
      tabIndex={-1}
      className={[
        'fixed inset-y-0 right-0 z-40 flex w-full flex-col',
        'bg-background shadow-lg outline-none md:w-[28rem]',
        'transition-transform duration-200 motion-reduce:transition-none',
        open ? 'translate-x-0' : 'translate-x-full',
      ].join(' ')}
    >
      {/* Matches the AppShell header beside it — same `h-12`, same
          `bg-chrome` — so the two read as one bar and the bottom borders
          meet as a single line across the viewport. */}
      <div className="flex h-12 shrink-0 items-center justify-between gap-3 border-b border-border bg-chrome px-4">
        <h2 className="truncate text-sm font-medium">{session?.fileName ?? ''}</h2>
        {/* The base IconButton, not RemoveButton: closing the player takes
            nothing away, so it must not warn in red. */}
        <IconButton icon={<XIcon />} label={t('player.close')} onClick={close} />
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-4">
        {session &&
          (failed ? (
            <p className="text-sm text-muted-foreground">{t('player.unplayable')}</p>
          ) : isVideoFile(session.fileName) ? (
            <video
              ref={mediaRef as React.RefObject<HTMLVideoElement>}
              key={src}
              src={src}
              controls
              playsInline
              preload="metadata"
              className="w-full rounded-md bg-black"
              onLoadedMetadata={(e) => {
                // Re-apply a seek requested before the media was ready.
                if (seekRequest) e.currentTarget.currentTime = seekRequest.seconds
              }}
              onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
              onError={() => setFailed(true)}
            />
          ) : (
            <audio
              ref={mediaRef as React.RefObject<HTMLAudioElement>}
              key={src}
              src={src}
              controls
              preload="metadata"
              className="w-full"
              onLoadedMetadata={(e) => {
                if (seekRequest) e.currentTarget.currentTime = seekRequest.seconds
              }}
              onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
              onError={() => setFailed(true)}
            />
          ))}
      </div>
    </aside>
  )
}
