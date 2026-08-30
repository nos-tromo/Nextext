import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import { WordCountsTab } from './WordCountsTab'
import type { JobResult, WordCount } from '../../api/types'

const counts: WordCount[] = [
  { word: 'alpha', count: 40 },
  { word: 'beta', count: 10 },
  { word: 'gamma', count: 5 },
]

function makeResult(word_counts: WordCount[] | null): JobResult {
  return {
    transcript: [],
    transcript_language: 'en',
    resolved_src_lang: 'en',
    summary: null,
    word_counts,
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
  }
}

/** The proportional bar fill rendered for each word row, in document order. */
function barFills(container: HTMLElement): HTMLElement[] {
  return Array.from(container.querySelectorAll<HTMLElement>('[data-testid="word-count-bar"]'))
}

describe('WordCountsTab', () => {
  it('lists every word with its count', () => {
    render(<WordCountsTab jobId="j1" result={makeResult(counts)} stem="clip" />)
    expect(screen.getByText('alpha')).toBeInTheDocument()
    expect(screen.getByText('40')).toBeInTheDocument()
    expect(screen.getByText('gamma')).toBeInTheDocument()
    expect(screen.getByText('5')).toBeInTheDocument()
  })

  it('draws a bar per word scaled against the most frequent one', () => {
    // The magnitude is what makes this a histogram rather than a word list:
    // the top word fills the track and the rest are a share of it.
    const { container } = render(<WordCountsTab jobId="j1" result={makeResult(counts)} stem="clip" />)
    expect(barFills(container).map((bar) => bar.style.width)).toEqual(['100%', '25%', '12.5%'])
  })

  it('keeps the bars out of the accessibility tree, since each row already reads its count', () => {
    const { container } = render(<WordCountsTab jobId="j1" result={makeResult(counts)} stem="clip" />)
    for (const bar of barFills(container)) {
      expect(bar.closest('[aria-hidden="true"]')).not.toBeNull()
    }
  })

  it('scales against the largest count even when the list is not sorted', () => {
    const unsorted: WordCount[] = [
      { word: 'small', count: 2 },
      { word: 'big', count: 8 },
    ]
    const { container } = render(<WordCountsTab jobId="j1" result={makeResult(unsorted)} stem="clip" />)
    expect(barFills(container).map((bar) => bar.style.width)).toEqual(['25%', '100%'])
  })

  it('offers the word counts as CSV and XLSX downloads', () => {
    render(<WordCountsTab jobId="j1" result={makeResult(counts)} stem="clip" />)
    expect(screen.getByRole('button', { name: 'Download CSV' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Download XLSX' })).toBeInTheDocument()
  })

  it('renders an empty state when the job produced no word counts', () => {
    const { container } = render(<WordCountsTab jobId="j1" result={makeResult(null)} stem="clip" />)
    expect(screen.getByText('No word counts available for this job.')).toBeInTheDocument()
    expect(barFills(container)).toHaveLength(0)
  })
})
