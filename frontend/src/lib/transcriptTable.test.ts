import { describe, expect, it } from 'vitest'
import { transcriptHasSpeaker, transcriptHasTranslation } from './transcriptTable'
import type { TranscriptSegment } from '../api/types'

const seg = (
  speaker: string | null,
  text = 'hi',
  translation: string | null = null,
): TranscriptSegment => ({
  start: null,
  end: null,
  speaker,
  text,
  translation,
})

describe('transcriptHasSpeaker', () => {
  it('returns false for an empty transcript', () => {
    expect(transcriptHasSpeaker([])).toBe(false)
  })

  it('returns false when all speakers are null', () => {
    expect(transcriptHasSpeaker([seg(null), seg(null)])).toBe(false)
  })

  it('returns false when all speakers are empty string', () => {
    expect(transcriptHasSpeaker([seg(''), seg('')])).toBe(false)
  })

  it('returns true when at least one segment has a non-empty speaker', () => {
    expect(transcriptHasSpeaker([seg(null), seg('SPEAKER_00')])).toBe(true)
  })

  it('returns true when all segments have speakers', () => {
    expect(transcriptHasSpeaker([seg('SPEAKER_00'), seg('SPEAKER_01')])).toBe(true)
  })
})

describe('transcriptHasTranslation', () => {
  it('returns false for an empty transcript', () => {
    expect(transcriptHasTranslation([])).toBe(false)
  })

  it('returns false when all translations are null', () => {
    expect(transcriptHasTranslation([seg(null), seg(null)])).toBe(false)
  })

  it('returns false when all translations are empty string', () => {
    expect(transcriptHasTranslation([seg(null, 'hi', ''), seg(null, 'hi', '')])).toBe(false)
  })

  it('returns true when at least one segment has a non-empty translation', () => {
    expect(transcriptHasTranslation([seg(null), seg(null, 'hi', 'hallo')])).toBe(true)
  })

  it('returns true when all segments have translations', () => {
    expect(
      transcriptHasTranslation([seg(null, 'hi', 'hallo'), seg(null, 'bye', 'tschüss')]),
    ).toBe(true)
  })
})
