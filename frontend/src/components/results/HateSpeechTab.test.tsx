import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import { HateSpeechTab } from './HateSpeechTab'
import type { HateSpeechFinding, JobResult, TranscriptSegment } from '../../api/types'

function segment(text: string): TranscriptSegment {
  return {
    start: '0:00:01',
    end: '0:00:02',
    start_seconds: 1,
    end_seconds: 2,
    speaker: null,
    text,
    translation: null,
  }
}

function finding(overrides: Partial<HateSpeechFinding> = {}): HateSpeechFinding {
  return {
    hate_speech: true,
    category: 'ethnicity',
    confidence: 'high',
    reason: 'Calls for expelling a group.',
    text: 'Genau, raus mit denen.',
    start: '0:00:04',
    speaker: null,
    translation: null,
    ...overrides,
  }
}

function makeResult(overrides: Partial<JobResult> = {}): JobResult {
  return {
    transcript: [segment('Im Bus rief jemand etwas.'), segment('Genau, raus mit denen.'), segment('Weiter.')],
    transcript_language: 'de',
    resolved_src_lang: 'de',
    summary: null,
    word_counts: null,
    named_entities: null,
    wordcloud_url: null,
    keyframes_url: null,
    media_url: null,
    frame_captions: null,
    hate_speech_findings: [finding()],
    skipped: false,
    skip_reason: null,
    skip_reason_code: null,
    task: 'transcribe',
    ...overrides,
  }
}

describe('HateSpeechTab', () => {
  it('counts flagged segments against the whole transcript, not against the findings', () => {
    render(<HateSpeechTab jobId="j1" result={makeResult()} stem="talk" />)
    expect(screen.getByText('1 of 3 segments flagged.')).toBeInTheDocument()
  })

  it('shows who said it and the translation aid next to the original wording', () => {
    const result = makeResult({
      hate_speech_findings: [finding({ speaker: 'Speaker 2', translation: 'Exactly, get them out.' })],
    })
    render(<HateSpeechTab jobId="j1" result={result} stem="talk" />)
    expect(screen.getByText('Speaker 2')).toBeInTheDocument()
    expect(screen.getByText('Genau, raus mit denen.')).toBeInTheDocument()
    expect(screen.getByText('Translation: Exactly, get them out.')).toBeInTheDocument()
  })

  it('renders no speaker or translation line for an undiarized, untranslated job', () => {
    render(<HateSpeechTab jobId="j1" result={makeResult()} stem="talk" />)
    expect(screen.queryByText(/^Speaker/)).not.toBeInTheDocument()
    expect(screen.queryByText(/^Translation:/)).not.toBeInTheDocument()
  })

  it('keeps the empty state when nothing was flagged', () => {
    render(<HateSpeechTab jobId="j1" result={makeResult({ hate_speech_findings: [] })} stem="talk" />)
    expect(screen.getByText('No hate-speech findings for this job.')).toBeInTheDocument()
  })
})
