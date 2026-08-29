import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import { VisualContextTab } from './VisualContextTab'
import type { FrameCaption, JobResult } from '../../api/types'

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
    render(<VisualContextTab jobId="j1" result={makeResult(captions)} stem="clip" />)
    expect(screen.getByText('A slide titled Roadmap')).toBeInTheDocument()
    expect(screen.getByText('A speaker at a lectern')).toBeInTheDocument()
  })

  it('stamps each caption with its moment in the clip', () => {
    render(<VisualContextTab jobId="j1" result={makeResult(captions)} stem="clip" />)
    expect(screen.getByText('00:00')).toBeInTheDocument()
    expect(screen.getByText('01:15')).toBeInTheDocument()
  })

  it('shows the captions without needing a disclosure to be opened', () => {
    // The tab is now the dedicated place for this, so its content is the page —
    // not something folded away behind a summary.
    const { container } = render(<VisualContextTab jobId="j1" result={makeResult(captions)} stem="clip" />)
    expect(container.querySelector('details')).toBeNull()
  })

  it('offers the visual context as a download', () => {
    render(<VisualContextTab jobId="j1" result={makeResult(captions)} stem="clip" />)
    expect(screen.getByRole('button', { name: 'Download TXT' })).toBeInTheDocument()
  })

  it('renders an empty state when the job produced no captions', () => {
    render(<VisualContextTab jobId="j1" result={makeResult(null)} stem="clip" />)
    expect(screen.getByText('No visual context produced for this job.')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Download TXT' })).not.toBeInTheDocument()
  })

  it('renders the empty state for an empty caption list too', () => {
    render(<VisualContextTab jobId="j1" result={makeResult([])} stem="clip" />)
    expect(screen.getByText('No visual context produced for this job.')).toBeInTheDocument()
  })
})
