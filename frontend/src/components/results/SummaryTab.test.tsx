import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import { SummaryTab } from './SummaryTab'
import type { FrameCaption, JobResult } from '../../api/types'

function makeResult(overrides: Partial<JobResult> = {}): JobResult {
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
    frame_captions: null,
    hate_speech_findings: null,
    skipped: false,
    skip_reason: null,
    skip_reason_code: null,
    task: 'transcribe',
    ...overrides,
  }
}

const captions: FrameCaption[] = [
  { time_sec: 0, caption: 'A slide titled Roadmap' },
  { time_sec: 75, caption: 'A speaker at a lectern' },
]

describe('SummaryTab', () => {
  it('no longer carries the visual context — it has its own tab now', () => {
    render(<SummaryTab jobId="j1" result={makeResult({ frame_captions: captions })} stem="clip" />)
    expect(screen.queryByText('A slide titled Roadmap')).not.toBeInTheDocument()
    expect(screen.queryByText(/visual context/i)).not.toBeInTheDocument()
  })

  it('keeps a single plain TXT download whether or not captions exist', () => {
    const { unmount } = render(
      <SummaryTab jobId="j1" result={makeResult({ frame_captions: captions })} stem="clip" />,
    )
    expect(screen.getByRole('button', { name: 'Download TXT' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /visual context/i })).not.toBeInTheDocument()
    unmount()

    render(<SummaryTab jobId="j1" result={makeResult()} stem="clip" />)
    expect(screen.getByRole('button', { name: 'Download TXT' })).toBeInTheDocument()
  })

  it('still renders the summary text itself', () => {
    render(<SummaryTab jobId="j1" result={makeResult({ frame_captions: captions })} stem="clip" />)
    expect(screen.getByText('They discussed the roadmap.')).toBeInTheDocument()
  })

  it('renders the empty state when there is no summary at all', () => {
    render(<SummaryTab jobId="j1" result={makeResult({ summary: null })} stem="clip" />)
    expect(screen.getByText('No summary produced for this job.')).toBeInTheDocument()
  })
})
