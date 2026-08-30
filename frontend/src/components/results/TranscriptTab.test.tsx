import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, fireEvent, render, screen } from '@testing-library/react'
import { TranscriptTab } from './TranscriptTab'
import type { TranscriptSegment } from '../../api/types'
import { useMediaPlayerStore } from '../../lib/mediaPlayerStore'

const transcribeSegments: TranscriptSegment[] = [
  { start: '0.00', end: '2.00', start_seconds: 0, end_seconds: 2, speaker: null, text: 'Hello world', translation: null },
]

const translateSegments: TranscriptSegment[] = [
  { start: '0.00', end: '2.00', start_seconds: 0, end_seconds: 2, speaker: null, text: 'Hello world', translation: 'Hallo Welt' },
]

describe('TranscriptTab download buttons', () => {
  it('shows a single TXT button for a transcribe-only transcript', () => {
    render(<TranscriptTab jobId="j1" segments={transcribeSegments} stem="clip" fileName="clip.mp4" mediaUrl={null} />)
    expect(screen.getByRole('button', { name: 'Download TXT' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Download Transcript TXT' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Download Translation TXT' })).not.toBeInTheDocument()
  })

  it('splits into Transcript TXT and Translation TXT when a translation exists', () => {
    render(<TranscriptTab jobId="j1" segments={translateSegments} stem="clip" fileName="clip.mp4" mediaUrl={null} />)
    expect(screen.getByRole('button', { name: 'Download Transcript TXT' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Download Translation TXT' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Download TXT' })).not.toBeInTheDocument()
  })
})


const MEDIA_URL = '/api/v1/jobs/j1/media?token=t1'

const playable: TranscriptSegment[] = [
  { start: '0:00:00', end: '0:00:05', start_seconds: 0, end_seconds: 5, speaker: null, text: 'First line', translation: null },
  { start: '0:00:05', end: '0:00:09', start_seconds: 5, end_seconds: 9, speaker: null, text: 'Second line', translation: null },
]

afterEach(() => useMediaPlayerStore.getState().clear())

describe('TranscriptTab playback', () => {
  it('makes the start time a button when the recording is playable', () => {
    render(
      <TranscriptTab jobId="j1" segments={playable} stem="clip" fileName="clip.mp4" mediaUrl={MEDIA_URL} />,
    )
    expect(screen.getByRole('button', { name: 'Play from 0:00:00' })).toBeInTheDocument()
  })

  it('leaves the start time as plain text when there is no media', () => {
    // A job whose upload is gone must not offer a control that cannot work.
    render(<TranscriptTab jobId="j1" segments={playable} stem="clip" fileName="clip.mp4" mediaUrl={null} />)
    expect(screen.queryByRole('button', { name: /Play from/ })).not.toBeInTheDocument()
    expect(screen.getByText('0:00:00')).toBeInTheDocument()
  })

  it('leaves a row without a parsed offset unclickable', () => {
    const unparsed: TranscriptSegment[] = [
      { start: 'garbled', end: null, start_seconds: null, end_seconds: null, speaker: null, text: 'x', translation: null },
    ]
    render(<TranscriptTab jobId="j1" segments={unparsed} stem="clip" fileName="clip.mp4" mediaUrl={MEDIA_URL} />)
    expect(screen.queryByRole('button', { name: /Play from/ })).not.toBeInTheDocument()
  })

  it('opens the player at that row on click', () => {
    render(
      <TranscriptTab jobId="j1" segments={playable} stem="clip" fileName="clip.mp4" mediaUrl={MEDIA_URL} />,
    )
    fireEvent.click(screen.getByRole('button', { name: 'Play from 0:00:05' }))
    const state = useMediaPlayerStore.getState()
    expect(state.session).toEqual({ jobId: 'j1', fileName: 'clip.mp4', mediaUrl: MEDIA_URL })
    expect(state.seekRequest?.seconds).toBe(5)
  })

  it('marks the row the playhead is inside', () => {
    render(
      <TranscriptTab jobId="j1" segments={playable} stem="clip" fileName="clip.mp4" mediaUrl={MEDIA_URL} />,
    )
    act(() => {
      useMediaPlayerStore.getState().open({ jobId: 'j1', fileName: 'clip.mp4', mediaUrl: MEDIA_URL })
      useMediaPlayerStore.getState().setCurrentTime(6)
    })
    expect(screen.getByText('Second line').closest('tr')).toHaveAttribute('aria-current', 'true')
    expect(screen.getByText('First line').closest('tr')).not.toHaveAttribute('aria-current')
  })

  it('scrolls the row under the playhead into view', () => {
    // The highlight is useless once it has scrolled past the fold.
    render(
      <TranscriptTab jobId="j1" segments={playable} stem="clip" fileName="clip.mp4" mediaUrl={MEDIA_URL} />,
    )
    const row = screen.getByText('Second line').closest('tr')!
    // happy-dom lays nothing out; put the row below the fold so the hook has
    // a reason to scroll it in.
    row.getBoundingClientRect = () =>
      ({ top: 2000, bottom: 2040, left: 0, right: 100, width: 100, height: 40, x: 0, y: 2000, toJSON: () => ({}) }) as DOMRect
    const spy = vi.spyOn(row, 'scrollIntoView')
    act(() => {
      useMediaPlayerStore.getState().open({ jobId: 'j1', fileName: 'clip.mp4', mediaUrl: MEDIA_URL })
      useMediaPlayerStore.getState().setCurrentTime(6)
    })
    expect(spy).toHaveBeenCalledWith(expect.objectContaining({ block: 'center' }))
    spy.mockRestore()
  })

  it('never marks a row while another job is playing', () => {
    // Several result panels can be open at once; only the playing one follows.
    render(
      <TranscriptTab jobId="j1" segments={playable} stem="clip" fileName="clip.mp4" mediaUrl={MEDIA_URL} />,
    )
    act(() => {
      useMediaPlayerStore.getState().open({ jobId: 'other', fileName: 'x.mp4', mediaUrl: '/m' })
      useMediaPlayerStore.getState().setCurrentTime(6)
    })
    expect(screen.getByText('Second line').closest('tr')).not.toHaveAttribute('aria-current')
  })
})
