import { create } from 'zustand'

/** The recording currently loaded in the player. */
export interface MediaSession {
  jobId: string
  /** Original upload name — the panel's accessible label and visible title. */
  fileName: string
  /** Capability URL from the job result; carries its own auth token. */
  mediaUrl: string
}

/**
 * A request to move the playhead.
 *
 * The `id` is what makes a repeat click work: seeking to a second the player
 * is already at is not a state change, so without a monotonic marker the
 * effect that applies the seek would never re-run.
 */
export interface SeekRequest {
  seconds: number
  id: number
}

export interface MediaPlayerState {
  /** Open recording, or `null` when the panel is closed. */
  session: MediaSession | null
  /** Pending seek, consumed by the player element. */
  seekRequest: SeekRequest | null
  /** Latest playhead position in seconds; drives row highlighting. */
  currentTime: number
  /**
   * Counter bumped by every deliberate jump (`open`, `seek`).
   *
   * Views that scroll the active row into view stop following once the
   * reader scrolls by hand; a change here is the signal to resume, so a
   * timestamp click always brings the reader back to the playhead.
   */
  followSeq: number
  /** Load a recording, optionally jumping straight to a timestamp. */
  open: (session: MediaSession, seekSeconds?: number) => void
  /** Move the playhead of the already-open recording. No-op when closed. */
  seek: (seconds: number) => void
  /** Close the panel and forget the recording. */
  close: () => void
  /** Report the playhead (from the media element's `timeupdate`). */
  setCurrentTime: (seconds: number) => void
  /** Reset everything (test resets). */
  clear: () => void
}

let nextSeekId = 0

/** Mint a monotonic seek id so repeat seeks to the same second still apply. */
function seekRequestFor(seconds: number): SeekRequest {
  nextSeekId += 1
  return { seconds, id: nextSeekId }
}

/**
 * Shared, module-singleton store for the slide-in media player.
 *
 * There is exactly one player for the whole app, but several job result
 * panels can be open at once, so the session records which job is playing —
 * a transcript row only highlights when its own job is the one loaded.
 *
 * Modelled on {@link useJobProgressStore}.
 */
export const useMediaPlayerStore = create<MediaPlayerState>((set, get) => ({
  session: null,
  seekRequest: null,
  currentTime: 0,
  followSeq: 0,
  open: (session, seekSeconds) =>
    set((state) => {
      // Switching recordings restarts the playhead; re-opening the same one
      // (a second timestamp click) must not make the highlight jump to 0.
      const sameJob = state.session?.jobId === session.jobId
      return {
        session,
        seekRequest: seekSeconds === undefined ? null : seekRequestFor(seekSeconds),
        currentTime: sameJob ? state.currentTime : 0,
        followSeq: state.followSeq + 1,
      }
    }),
  seek: (seconds) => {
    if (!get().session) return
    set((state) => ({ seekRequest: seekRequestFor(seconds), followSeq: state.followSeq + 1 }))
  },
  close: () => set({ session: null, seekRequest: null, currentTime: 0 }),
  setCurrentTime: (seconds) => set({ currentTime: seconds }),
  clear: () => set({ session: null, seekRequest: null, currentTime: 0, followSeq: 0 }),
}))
