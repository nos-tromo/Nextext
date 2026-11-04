import { useCallback, useEffect, useRef, useState, type SyntheticEvent } from 'react'
import { IconButton, PURPOSE_REQUIRED_STATUS, Spinner, XIcon, subscribePurpose } from '@infra/ui'
import { ApiError, PURPOSE_SCOPE } from '../../api/client'
import { requestMediaGrant } from '../../api/jobs'
import { describeError, type ErrorDescriptor } from '../../api/errorMessage'
import { useMediaPlayerStore } from '../../lib/mediaPlayerStore'
import { mediaSrcUrl } from '../../lib/mediaSrc'
import { useT } from '../../i18n/LanguageContext'

/** Upload extensions that carry a picture and therefore need a `<video>`. */
const VIDEO_EXTENSIONS = new Set(['mp4', 'mkv', 'webm', 'mov', 'avi', 'm4v'])

/**
 * Least time between two grants asked for because the element failed.
 *
 * A grant expires once per lifetime, so an error soon after a fresh grant is
 * the file's (a container the browser cannot decode, say), not the grant's:
 * the player then reports it instead of asking for grants in a loop.
 */
export const REGRANT_INTERVAL_MS = 30_000

/** `HTMLMediaElement.HAVE_METADATA`, spelled out: not every DOM implementation defines the static. */
const HAVE_METADATA = 1

/** The grant the element plays, for the job it was issued for. */
interface Grant {
  jobId: string
  src: string
}

/** Where to pick playback up once a fresh grant replaced an expired one. */
interface Resume {
  time: number
  play: boolean
}

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
 * Its header band repeats the AppShell chrome — same colour, same height,
 * same borderless band over a ruled body — so it continues the app header
 * rather than sitting beside it as a separate block; the panel is set off by
 * its shadow alone, with no left border to cut that bar in two.
 *
 * Deliberately **not** a dialog: no backdrop and no `aria-modal`, because the
 * point is that the jobs list and the result tabs stay usable while a
 * recording plays. It keeps only the parts of the overlay contract that still
 * apply — Escape closes, and focus returns to whatever opened it.
 *
 * The element stays mounted while closed so the slide has something to
 * animate; `aria-hidden` plus `inert` keep it out of the accessibility tree
 * and the tab order in that state.
 *
 * A media element can send neither the identity nor the stated purpose, so
 * the panel asks for a playback grant first — an ordinary API request, which
 * states the purpose like any other — and only then sets `src`. A grant
 * expires; when the element fails, the panel asks for a fresh one once and
 * picks playback up where it stopped ({@link REGRANT_INTERVAL_MS}). A grant
 * refused because no purpose is stated leaves the panel saying so, and it
 * plays as soon as a purpose is stated in the header.
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
  const [grant, setGrant] = useState<Grant | null>(null)
  const [grantError, setGrantError] = useState<ErrorDescriptor | null>(null)
  // The grant was refused (428) and the user declined to state a purpose.
  const [purposeNeeded, setPurposeNeeded] = useState(false)
  const [failed, setFailed] = useState(false)
  // Bumped per grant request and when the recording changes, so a reply that
  // is no longer wanted is dropped.
  const requestSeqRef = useRef(0)
  const lastRegrantAtRef = useRef<number | null>(null)
  // The `src` a fresh grant was already asked for, so a second error from the
  // same element does not count twice.
  const regrantedSrcRef = useRef<string | null>(null)
  const resumeAtRef = useRef<Resume | null>(null)

  const open = session !== null
  const jobId = session?.jobId ?? null
  // The backend hands out a server-relative URL; the element needs one
  // resolved against the sub-path the SPA is mounted under.
  const src = grant !== null && grant.jobId === jobId ? grant.src : undefined

  const fetchGrant = useCallback(async (id: string) => {
    const seq = ++requestSeqRef.current
    try {
      const { media_url } = await requestMediaGrant(id)
      if (seq === requestSeqRef.current) setGrant({ jobId: id, src: mediaSrcUrl(media_url) })
    } catch (err) {
      if (seq !== requestSeqRef.current) return
      if (err instanceof ApiError && err.status === PURPOSE_REQUIRED_STATUS) setPurposeNeeded(true)
      else setGrantError(describeError(err))
    }
  }, [])

  // A new recording starts clean — last one's grant and playback error are
  // not this one's — and asks for its own grant.
  useEffect(() => {
    setGrant(null)
    setGrantError(null)
    setPurposeNeeded(false)
    setFailed(false)
    lastRegrantAtRef.current = null
    regrantedSrcRef.current = null
    resumeAtRef.current = null
    if (jobId === null) return
    void fetchGrant(jobId)
    return () => {
      requestSeqRef.current += 1
    }
  }, [jobId, fetchGrant])

  // Waiting for a purpose: the moment one is stated in the header, ask again.
  useEffect(() => {
    if (!purposeNeeded || jobId === null) return
    return subscribePurpose(PURPOSE_SCOPE, (purpose) => {
      if (!purpose) return
      setPurposeNeeded(false)
      void fetchGrant(jobId)
    })
  }, [purposeNeeded, jobId, fetchGrant])

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

  // The element failed: most often because its grant expired and the next
  // range request was refused. Ask for a fresh grant — once: an error soon
  // after the last fresh grant is the file's, and is reported.
  const onError = (event: SyntheticEvent<HTMLMediaElement>) => {
    if (jobId === null || src === undefined || regrantedSrcRef.current === src) return
    const element = event.currentTarget
    const now = performance.now()
    if (lastRegrantAtRef.current !== null && now - lastRegrantAtRef.current < REGRANT_INTERVAL_MS) {
      setFailed(true)
      return
    }
    lastRegrantAtRef.current = now
    regrantedSrcRef.current = src
    // Mid-playback, pick up where it stopped; before metadata, a pending
    // timestamp seek still applies.
    resumeAtRef.current =
      element.readyState >= HAVE_METADATA ? { time: element.currentTime, play: !element.paused } : null
    void fetchGrant(jobId)
  }

  const onLoadedMetadata = (event: SyntheticEvent<HTMLMediaElement>) => {
    const element = event.currentTarget
    const resume = resumeAtRef.current
    resumeAtRef.current = null
    if (resume) {
      element.currentTime = resume.time
      if (resume.play) void element.play?.()?.catch?.(() => {})
    } else if (seekRequest) {
      // A timestamp clicked before the media was ready — the element only
      // exists once its grant is there — seeks and plays, as a click on a
      // ready player does.
      element.currentTime = seekRequest.seconds
      void element.play?.()?.catch?.(() => {
        // Autoplay can be refused; the seek still landed.
      })
    }
  }

  // Apply seeks. Setting `currentTime` before metadata is loaded is discarded
  // by the browser, so a request that arrives early is re-applied on
  // `loadedmetadata` (see `onLoadedMetadata`).
  useEffect(() => {
    if (!seekRequest) return
    // A timestamp clicked while a fresh grant loads outranks picking up where
    // the expired one stopped.
    resumeAtRef.current = null
    const element = mediaRef.current
    if (!element) return
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
      {/* Mirrors the AppShell chrome beside it: a borderless `h-12` band on
          `bg-chrome`, with the rule carried by the region below (see the
          body's `border-t`). Putting the rule here instead would land it a
          pixel high — `border-box` sizing draws it *inside* the 48px — and
          the seam would visibly step at the panel edge. */}
      <div className="flex h-12 shrink-0 items-center justify-between gap-3 bg-chrome px-4">
        <h2 className="truncate text-sm font-medium">{session?.fileName ?? ''}</h2>
        {/* The base IconButton, not RemoveButton: closing the player takes
            nothing away, so it must not warn in red. */}
        <IconButton icon={<XIcon />} label={t('player.close')} onClick={close} />
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto border-t border-border p-4">
        {session &&
          (failed ? (
            <p className="text-sm text-muted-foreground">{t('player.unplayable')}</p>
          ) : purposeNeeded ? (
            <p className="text-sm text-muted-foreground">{t('player.purpose_needed')}</p>
          ) : grantError ? (
            <p className="text-sm text-danger">{t(grantError.key, grantError.vars)}</p>
          ) : src === undefined ? (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Spinner label={t('player.loading')} />
              <span aria-hidden>{t('player.loading')}</span>
            </div>
          ) : isVideoFile(session.fileName) ? (
            <video
              ref={mediaRef as React.RefObject<HTMLVideoElement>}
              key={src}
              src={src}
              controls
              playsInline
              preload="metadata"
              className="w-full rounded-md bg-black"
              onLoadedMetadata={onLoadedMetadata}
              onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
              onError={onError}
            />
          ) : (
            <audio
              ref={mediaRef as React.RefObject<HTMLAudioElement>}
              key={src}
              src={src}
              controls
              preload="metadata"
              className="w-full"
              onLoadedMetadata={onLoadedMetadata}
              onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
              onError={onError}
            />
          ))}
      </div>
    </aside>
  )
}
