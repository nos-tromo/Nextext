import { describe, expect, it } from 'vitest'
import { downloadsFor } from './resultDownloads'
import type { JobResult, TranscriptSegment } from '../api/types'

const t = (key: string) => key

const seg = (translation: string | null = null): TranscriptSegment => ({
  start: null,
  end: null,
  start_seconds: null,
  end_seconds: null,
  speaker: null,
  text: 'hallo',
  translation,
})

function result(overrides: Partial<JobResult> = {}): JobResult {
  return {
    transcript: [seg()],
    transcript_language: 'de',
    resolved_src_lang: 'de',
    summary: null,
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
    ...overrides,
  } as JobResult
}

const names = (tab: Parameters<typeof downloadsFor>[0], r: JobResult) =>
  downloadsFor(tab, r, 'clip', t).map((d) => [d.name, d.label, d.fileName])

describe('downloadsFor', () => {
  it('offers the transcript as CSV, XLSX, TXT and JSONL', () => {
    expect(names('transcript', result())).toEqual([
      ['transcript.csv', 'CSV', 'clip_transcript.csv'],
      ['transcript.xlsx', 'XLSX', 'clip_transcript.xlsx'],
      ['transcript.txt', 'TXT', 'clip_transcript.txt'],
      ['docint.jsonl', 'JSONL', 'clip_docint.jsonl'],
    ])
  })

  it('splits transcript TXT when a translation exists', () => {
    expect(names('transcript', result({ transcript: [seg('hello')] }))).toEqual([
      ['transcript.csv', 'CSV', 'clip_transcript.csv'],
      ['transcript.xlsx', 'XLSX', 'clip_transcript.xlsx'],
      ['transcript.txt', 'downloads.transcript_txt', 'clip_transcript.txt'],
      ['translation.txt', 'downloads.translation_txt', 'clip_translation.txt'],
      ['docint.jsonl', 'JSONL', 'clip_docint.jsonl'],
    ])
  })

  it('offers the described frames as TXT and the frames as a ZIP', () => {
    const r = result({ frame_captions: [{ time_sec: 1, caption: 'Ein Tisch' }] as JobResult['frame_captions'], keyframes_url: '/k.zip' })
    expect(names('visual_context', r)).toEqual([
      ['visual_context.txt', 'TXT', 'clip_visual_context.txt'],
      ['keyframes.zip', 'downloads.keyframes_zip', 'clip_keyframes.zip'],
    ])
    expect(downloadsFor('visual_context', r, 'clip', t)[1].title).toBe('artifacts.download_keyframes')
  })

  it('offers only the keyframes ZIP when frames have no captions', () => {
    expect(names('visual_context', result({ keyframes_url: '/k.zip' }))).toEqual([
      ['keyframes.zip', 'downloads.keyframes_zip', 'clip_keyframes.zip'],
    ])
  })

  it('offers the summary, word counts, word cloud, entities and hate speech', () => {
    const r = result()
    expect(names('summary', r)).toEqual([['summary.txt', 'TXT', 'clip_summary.txt']])
    expect(names('words', r)).toEqual([
      ['wordcounts.csv', 'CSV', 'clip_wordcounts.csv'],
      ['wordcounts.xlsx', 'XLSX', 'clip_wordcounts.xlsx'],
    ])
    expect(names('wordcloud', r)).toEqual([['wordcloud.png', 'PNG', 'clip_wordcloud.png']])
    expect(names('entities', r)).toEqual([
      ['entities.csv', 'CSV', 'clip_entities.csv'],
      ['entities.xlsx', 'XLSX', 'clip_entities.xlsx'],
    ])
    expect(names('hate_speech', r)).toEqual([
      ['hate_speech.csv', 'CSV', 'clip_hate_speech.csv'],
      ['hate_speech.xlsx', 'XLSX', 'clip_hate_speech.xlsx'],
    ])
  })

  it('offers the faces and their table only while there are faces', () => {
    const face = {
      number: 1,
      assigned: true,
      first_seen_seconds: 0,
      last_seen_seconds: 1,
      appearances: [{ start_seconds: 0, end_seconds: 1 }],
      detections: 1,
      thumbnail: 'data:image/gif;base64,R0lGODlhAQABAAAAACw=',
      best_artifact: 'face_001.jpg',
    }
    const faces = (overrides: Partial<NonNullable<JobResult['faces']>> = {}) =>
      result({ faces: { outcome: 'ok', grouped: true, truncated: false, entries: [face], ...overrides } })
    expect(names('faces', faces())).toEqual([
      ['faces.zip', 'ZIP', 'clip_faces.zip'],
      ['faces.csv', 'CSV', 'clip_faces.csv'],
    ])
    expect(downloadsFor('faces', faces({ entries: [] }), 'clip', t)).toEqual([])
    expect(downloadsFor('faces', faces({ outcome: 'unavailable' }), 'clip', t)).toEqual([])
    expect(downloadsFor('faces', result({ faces: null }), 'clip', t)).toEqual([])
  })

  it('offers no transcript downloads for an empty transcript', () => {
    expect(downloadsFor('transcript', result({ transcript: [] }), 'clip', t)).toEqual([])
  })
})
