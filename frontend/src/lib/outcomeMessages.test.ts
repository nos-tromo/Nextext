import { describe, expect, it } from 'vitest'
import { failureMessageKey, jobSkipMessageKey, resultSkipMessageKey } from './outcomeMessages'
import { en } from '../i18n/en'

describe('jobSkipMessageKey', () => {
  it('maps each typed reason to its own short card message', () => {
    expect(jobSkipMessageKey('vad_no_speech')).toBe('jobs.skipped_vad_no_speech')
    expect(jobSkipMessageKey('asr_empty_transcript')).toBe('jobs.skipped_asr_empty')
    expect(jobSkipMessageKey('asr_all_segments_filtered')).toBe('jobs.skipped_asr_filtered')
  })

  it('falls back to the generic message for a missing or unknown code', () => {
    expect(jobSkipMessageKey(null)).toBe('jobs.skipped')
    // An older or newer backend may send a code this build does not know.
    expect(jobSkipMessageKey('something_new' as never)).toBe('jobs.skipped')
  })
})

describe('resultSkipMessageKey', () => {
  it('maps each typed reason to its own explanation', () => {
    expect(resultSkipMessageKey('vad_no_speech')).toBe('results.skipped_vad_no_speech')
    expect(resultSkipMessageKey('asr_empty_transcript')).toBe('results.skipped_asr_empty')
    expect(resultSkipMessageKey('asr_all_segments_filtered')).toBe('results.skipped_asr_filtered')
    expect(resultSkipMessageKey(null)).toBe('results.skipped')
  })
})

describe('failureMessageKey', () => {
  it('explains an undecodable upload instead of calling it unknown', () => {
    expect(failureMessageKey('undecodable_media', false)).toBe('jobs.error_undecodable')
  })

  it('keeps the generic message for internal failures and unknown codes', () => {
    expect(failureMessageKey('internal', false)).toBe('jobs.unknown_error')
    expect(failureMessageKey(null, false)).toBe('jobs.unknown_error')
  })

  it('prefers the interrupted message, which describes the job not the cause', () => {
    expect(failureMessageKey('undecodable_media', true)).toBe('jobs.interrupted')
    expect(failureMessageKey(null, true)).toBe('jobs.interrupted')
  })
})

describe('message catalog', () => {
  it('has an English string for every key these mappers can return', () => {
    const keys = [
      jobSkipMessageKey('vad_no_speech'),
      jobSkipMessageKey('asr_empty_transcript'),
      jobSkipMessageKey('asr_all_segments_filtered'),
      jobSkipMessageKey(null),
      resultSkipMessageKey('vad_no_speech'),
      resultSkipMessageKey('asr_empty_transcript'),
      resultSkipMessageKey('asr_all_segments_filtered'),
      resultSkipMessageKey(null),
      failureMessageKey('undecodable_media', false),
      failureMessageKey('internal', false),
      failureMessageKey(null, true),
    ]
    for (const key of keys) {
      expect(en[key]).toBeTruthy()
    }
  })
})
