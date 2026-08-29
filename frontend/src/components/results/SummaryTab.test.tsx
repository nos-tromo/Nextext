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

describe('SummaryTab visual context', () => {
  it('lists the frame captions behind a video summary', () => {
    render(<SummaryTab jobId="j1" result={makeResult({ frame_captions: captions })} stem="clip" />)
    expect(screen.getByText('A slide titled Roadmap')).toBeInTheDocument()
    expect(screen.getByText('A speaker at a lectern')).toBeInTheDocument()
  })

  it('stamps each caption with its moment in the clip', () => {
    render(<SummaryTab jobId="j1" result={makeResult({ frame_captions: captions })} stem="clip" />)
    expect(screen.getByText('01:15')).toBeInTheDocument()
  })

  it('labels both downloads distinctly when a visual context exists', () => {
    // Two bare "TXT" chips side by side would be a guessing game — the same
    // reason TranscriptTab splits into "Transcript TXT" / "Translation TXT".
    render(<SummaryTab jobId="j1" result={makeResult({ frame_captions: captions })} stem="clip" />)
    expect(screen.getByRole('button', { name: 'Download Summary TXT' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Download Visual context TXT' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Download TXT' })).not.toBeInTheDocument()
  })

  it('keeps the lone summary download a plain TXT chip', () => {
    render(<SummaryTab jobId="j1" result={makeResult()} stem="clip" />)
    expect(screen.getByRole('button', { name: 'Download TXT' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Download Summary TXT' })).not.toBeInTheDocument()
  })

  it('shows no visual-context section for an audio-only summary', () => {
    render(<SummaryTab jobId="j1" result={makeResult()} stem="clip" />)
    expect(screen.queryByText(/visual context/i)).not.toBeInTheDocument()
  })

  it('shows no visual-context section when the caption list is empty', () => {
    render(<SummaryTab jobId="j1" result={makeResult({ frame_captions: [] })} stem="clip" />)
    expect(screen.queryByText(/visual context/i)).not.toBeInTheDocument()
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
