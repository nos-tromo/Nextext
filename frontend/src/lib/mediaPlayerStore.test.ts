import { beforeEach, describe, expect, it } from 'vitest'
import { useMediaPlayerStore } from './mediaPlayerStore'

const clip = { jobId: 'j1', fileName: 'clip.mp4', mediaUrl: '/api/v1/jobs/j1/media?token=t1' }
const other = { jobId: 'j2', fileName: 'talk.wav', mediaUrl: '/api/v1/jobs/j2/media?token=t2' }

function store() {
  return useMediaPlayerStore.getState()
}

describe('mediaPlayerStore', () => {
  beforeEach(() => {
    useMediaPlayerStore.getState().clear()
  })

  it('starts closed', () => {
    expect(store().session).toBeNull()
    expect(store().seekRequest).toBeNull()
  })

  it('opens a session at a timestamp', () => {
    store().open(clip, 42)
    expect(store().session).toEqual(clip)
    expect(store().seekRequest?.seconds).toBe(42)
  })

  it('opens without a timestamp when the player is opened from the tab bar', () => {
    store().open(clip)
    expect(store().session).toEqual(clip)
    expect(store().seekRequest).toBeNull()
  })

  it('re-seeks when the same timestamp is clicked twice', () => {
    // Without a monotonic id the second click would be a no-op state update,
    // so clicking a row you already jumped to would do nothing.
    store().open(clip, 42)
    const first = store().seekRequest?.id
    store().seek(42)
    expect(store().seekRequest?.id).not.toBe(first)
    expect(store().seekRequest?.seconds).toBe(42)
  })

  it('replaces the session when another job is opened', () => {
    // One player, one recording: opening a second job's transcript must not
    // leave the first job's audio playing under it.
    store().open(clip, 10)
    store().open(other, 5)
    expect(store().session).toEqual(other)
    expect(store().seekRequest?.seconds).toBe(5)
  })

  it('resets the playhead when the session changes', () => {
    store().open(clip, 10)
    store().setCurrentTime(30)
    store().open(other, 0)
    expect(store().currentTime).toBe(0)
  })

  it('keeps the playhead when re-seeking the same job', () => {
    store().open(clip, 10)
    store().setCurrentTime(30)
    store().open(clip, 90)
    expect(store().currentTime).toBe(30)
    expect(store().seekRequest?.seconds).toBe(90)
  })

  it('tracks the playhead', () => {
    store().open(clip)
    store().setCurrentTime(12.5)
    expect(store().currentTime).toBe(12.5)
  })

  it('closes and forgets the session', () => {
    store().open(clip, 10)
    store().setCurrentTime(30)
    store().close()
    expect(store().session).toBeNull()
    expect(store().seekRequest).toBeNull()
    expect(store().currentTime).toBe(0)
  })

  it('ignores a seek when nothing is open', () => {
    store().seek(10)
    expect(store().seekRequest).toBeNull()
  })

  it('starts with a zero follow counter', () => {
    expect(store().followSeq).toBe(0)
  })

  it('bumps the follow counter when a session is opened', () => {
    // Opening the player is a deliberate act, so it re-engages following
    // even if the reader had scrolled away during an earlier session.
    store().open(clip, 42)
    expect(store().followSeq).toBe(1)
  })

  it('bumps the follow counter on every seek', () => {
    store().open(clip)
    store().seek(42)
    store().seek(42)
    expect(store().followSeq).toBe(3)
  })

  it('leaves the follow counter alone as the playhead advances', () => {
    // Only an explicit jump resumes following; ordinary playback must not
    // yank a reader who has scrolled off to read something else.
    store().open(clip)
    const seq = store().followSeq
    store().setCurrentTime(12)
    store().setCurrentTime(13)
    expect(store().followSeq).toBe(seq)
  })

  it('resets the follow counter on clear', () => {
    store().open(clip, 1)
    store().clear()
    expect(store().followSeq).toBe(0)
  })
})
