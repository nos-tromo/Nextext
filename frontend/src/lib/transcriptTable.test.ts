import { describe, expect, it } from 'vitest'
import { transcriptHasSpeaker } from './transcriptTable'
import type { TranscriptSegment } from '../api/types'

const seg = (speaker: string | null, text = 'hi'): TranscriptSegment => ({
  start: null,
  end: null,
  speaker,
  text,
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
