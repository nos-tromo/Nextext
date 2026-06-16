import { describe, expect, it } from 'vitest'
import { toJobEvent } from './jobEvents'

describe('toJobEvent', () => {
  it('maps a known frame to a typed event', () => {
    const event = toJobEvent({
      event: 'stage_completed',
      data: '{"stage":"Transcribing","stage_index":0,"progress":0.2,"timestamp":"t","result_delta":null}',
    })
    expect(event).toEqual({
      name: 'stage_completed',
      data: { stage: 'Transcribing', stage_index: 0, progress: 0.2, timestamp: 't', result_delta: null },
    })
  })

  it('maps a terminal job_completed frame', () => {
    const event = toJobEvent({ event: 'job_completed', data: '{"job_id":"j1","skipped":false,"timestamp":"t"}' })
    expect(event?.name).toBe('job_completed')
  })

  it('returns null for an unknown event name', () => {
    expect(toJobEvent({ event: 'mystery', data: '{}' })).toBeNull()
  })

  it('returns null for malformed JSON', () => {
    expect(toJobEvent({ event: 'stage_started', data: 'not json' })).toBeNull()
  })
})
