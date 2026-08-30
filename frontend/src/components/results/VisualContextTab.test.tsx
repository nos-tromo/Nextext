import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, fireEvent, render, screen } from '@testing-library/react'
import { VisualContextTab } from './VisualContextTab'
import type { FrameCaption, JobResult } from '../../api/types'
import { useMediaPlayerStore } from '../../lib/mediaPlayerStore'

const captions: FrameCaption[] = [
  { time_sec: 0, caption: 'A slide titled Roadmap' },
  { time_sec: 75, caption: 'A speaker at a lectern' },
]

function makeResult(frame_captions: FrameCaption[] | null): JobResult {
  return {
    transcript: [],
    transcript_language: 'en',
    resolved_src_lang: 'en',
    summary: 'They discussed the roadmap.',
    word_counts: null,
    named_entities: null,
    wordcloud_url: null,
    keyframes_url: null,
    media_url: null,
    frame_captions,
    hate_speech_findings: null,
    skipped: false,
    skip_reason: null,
    skip_reason_code: null,
    task: 'transcribe',
  }
}

describe('VisualContextTab', () => {
  it('lists every caption', () => {
    render(<VisualContextTab jobId="j1" result={makeResult(captions)} stem="clip" fileName="clip.mp4" mediaUrl={null} />)
    expect(screen.getByText('A slide titled Roadmap')).toBeInTheDocument()
    expect(screen.getByText('A speaker at a lectern')).toBeInTheDocument()
  })

  it('stamps each caption with its moment in the clip', () => {
    render(<VisualContextTab jobId="j1" result={makeResult(captions)} stem="clip" fileName="clip.mp4" mediaUrl={null} />)
    expect(screen.getByText('00:00')).toBeInTheDocument()
    expect(screen.getByText('01:15')).toBeInTheDocument()
  })

  it('shows the captions without needing a disclosure to be opened', () => {
    // The tab is now the dedicated place for this, so its content is the page —
    // not something folded away behind a summary.
    const { container } = render(<VisualContextTab jobId="j1" result={makeResult(captions)} stem="clip" fileName="clip.mp4" mediaUrl={null} />)
    expect(container.querySelector('details')).toBeNull()
  })

  it('offers the visual context as a download', () => {
    render(<VisualContextTab jobId="j1" result={makeResult(captions)} stem="clip" fileName="clip.mp4" mediaUrl={null} />)
    expect(screen.getByRole('button', { name: 'Download TXT' })).toBeInTheDocument()
  })

  it('renders an empty state when the job produced no captions', () => {
    render(<VisualContextTab jobId="j1" result={makeResult(null)} stem="clip" fileName="clip.mp4" mediaUrl={null} />)
    expect(screen.getByText('No visual context produced for this job.')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Download TXT' })).not.toBeInTheDocument()
  })

  it('renders the empty state for an empty caption list too', () => {
    render(<VisualContextTab jobId="j1" result={makeResult([])} stem="clip" fileName="clip.mp4" mediaUrl={null} />)
    expect(screen.getByText('No visual context produced for this job.')).toBeInTheDocument()
  })
})


const VC_MEDIA_URL = '/api/v1/jobs/j1/media?token=t1'

afterEach(() => useMediaPlayerStore.getState().clear())

describe('VisualContextTab playback', () => {
  it('makes each timestamp a button when the recording is playable', () => {
    render(
      <VisualContextTab
        jobId="j1"
        result={makeResult(captions)}
        stem="clip"
        fileName="clip.mp4"
        mediaUrl={VC_MEDIA_URL}
      />,
    )
    expect(screen.getByRole('button', { name: 'Play from 01:15' })).toBeInTheDocument()
  })

  it('leaves timestamps as plain text without media', () => {
    render(
      <VisualContextTab jobId="j1" result={makeResult(captions)} stem="clip" fileName="clip.mp4" mediaUrl={null} />,
    )
    expect(screen.queryByRole('button', { name: /Play from/ })).not.toBeInTheDocument()
    expect(screen.getByText('01:15')).toBeInTheDocument()
  })

  it('seeks to the frame on click', () => {
    render(
      <VisualContextTab
        jobId="j1"
        result={makeResult(captions)}
        stem="clip"
        fileName="clip.mp4"
        mediaUrl={VC_MEDIA_URL}
      />,
    )
    fireEvent.click(screen.getByRole('button', { name: 'Play from 01:15' }))
    expect(useMediaPlayerStore.getState().seekRequest?.seconds).toBe(75)
  })

  it('scrolls the reached frame into view', () => {
    render(
      <VisualContextTab
        jobId="j1"
        result={makeResult(captions)}
        stem="clip"
        fileName="clip.mp4"
        mediaUrl={VC_MEDIA_URL}
      />,
    )
    const row = screen.getByText('A speaker at a lectern').closest('tr')!
    // happy-dom lays nothing out; put the row below the fold so the hook has
    // a reason to scroll it in.
    row.getBoundingClientRect = () =>
      ({ top: 2000, bottom: 2040, left: 0, right: 100, width: 100, height: 40, x: 0, y: 2000, toJSON: () => ({}) }) as DOMRect
    const spy = vi.spyOn(row, 'scrollIntoView')
    act(() => {
      useMediaPlayerStore.getState().open({ jobId: 'j1', fileName: 'clip.mp4', mediaUrl: VC_MEDIA_URL })
      useMediaPlayerStore.getState().setCurrentTime(80)
    })
    expect(spy).toHaveBeenCalledWith(expect.objectContaining({ block: 'center' }))
    spy.mockRestore()
  })

  it('marks the frame the playhead has reached', () => {
    // Frames have no end time, so the active one is the latest already passed.
    render(
      <VisualContextTab
        jobId="j1"
        result={makeResult(captions)}
        stem="clip"
        fileName="clip.mp4"
        mediaUrl={VC_MEDIA_URL}
      />,
    )
    act(() => {
      useMediaPlayerStore.getState().open({ jobId: 'j1', fileName: 'clip.mp4', mediaUrl: VC_MEDIA_URL })
      useMediaPlayerStore.getState().setCurrentTime(80)
    })
    expect(screen.getByText('A speaker at a lectern').closest('tr')).toHaveAttribute('aria-current', 'true')
    expect(screen.getByText('A slide titled Roadmap').closest('tr')).not.toHaveAttribute('aria-current')
  })
})
