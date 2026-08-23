import { describe, expect, it } from 'vitest'
import { initialJobProgress, reduceJobEvent } from './jobProgress'
import type { JobEvent } from '../api/types'

const started = (i: number, p: number): JobEvent => ({
  name: 'stage_started',
  data: { job_id: 'j', stage: `S${i}`, stage_index: i, progress: p, timestamp: 't' },
})
const completed = (i: number, p: number): JobEvent => ({
  name: 'stage_completed',
  data: { job_id: 'j', stage: `S${i}`, stage_index: i, progress: p, timestamp: 't', result_delta: null },
})

describe('jobProgress reducer', () => {
  it('starts queued at 0', () => {
    const s = initialJobProgress()
    expect(s).toMatchObject({ status: 'queued', stageIndex: 0, progress: 0, terminal: false })
  })

  it('advances through stages, monotonic progress', () => {
    let s = initialJobProgress()
    s = reduceJobEvent(s, started(0, 0))
    expect(s).toMatchObject({ status: 'running', stageLabel: 'S0', progress: 0 })
    s = reduceJobEvent(s, completed(0, 0.2))
    expect(s.progress).toBe(0.2)
  })

  it('marks completed + carries skipped', () => {
    const s = reduceJobEvent(initialJobProgress(), {
      name: 'job_completed',
      data: { job_id: 'j1', skipped: true, skip_reason_code: null, timestamp: 't' },
    })
    expect(s).toMatchObject({ status: 'completed', progress: 1, skipped: true, terminal: true })
  })

  it('marks failed + carries error', () => {
    const s = reduceJobEvent(initialJobProgress(), {
      name: 'job_failed',
      data: { job_id: 'j1', error: 'boom', error_code: null, timestamp: 't' },
    })
    expect(s).toMatchObject({ status: 'failed', error: 'boom', terminal: true })
  })

  it('is idempotent under replay (reconnect re-sends history)', () => {
    const events = [started(0, 0), completed(0, 0.2), started(1, 0.2)]
    let once = initialJobProgress()
    for (const e of events) once = reduceJobEvent(once, e)
    let twice = initialJobProgress()
    for (const e of [...events, ...events]) twice = reduceJobEvent(twice, e)
    expect(twice).toEqual(once)
  })

  it('seeds the skip and failure codes from a list item so they survive a reload', () => {
    const seeded = initialJobProgress('completed', null, {
      skipped: true,
      skipReason: 'vad_no_speech',
      errorCode: null,
    })
    expect(seeded).toMatchObject({ status: 'completed', skipped: true, skipReason: 'vad_no_speech' })

    const failed = initialJobProgress('failed', 'Job failed.', {
      skipped: false,
      skipReason: null,
      errorCode: 'undecodable_media',
    })
    expect(failed).toMatchObject({ status: 'failed', errorCode: 'undecodable_media' })
  })

  it('defaults the codes to unset when the caller passes none', () => {
    expect(initialJobProgress()).toMatchObject({ skipped: false, skipReason: null, errorCode: null })
  })

  it('carries the typed codes off the terminal events', () => {
    const done = reduceJobEvent(initialJobProgress(), {
      name: 'job_completed',
      data: { job_id: 'j1', skipped: true, skip_reason_code: 'asr_empty_transcript', timestamp: 't' },
    })
    expect(done).toMatchObject({ skipped: true, skipReason: 'asr_empty_transcript' })

    const failed = reduceJobEvent(initialJobProgress(), {
      name: 'job_failed',
      data: { job_id: 'j1', error: 'Job failed.', error_code: 'undecodable_media', timestamp: 't' },
    })
    expect(failed).toMatchObject({ status: 'failed', errorCode: 'undecodable_media' })
  })

  it('seeds error only for a failed initial status', () => {
    expect(initialJobProgress('failed', 'boom')).toMatchObject({ status: 'failed', error: 'boom', terminal: true })
    expect(initialJobProgress('completed', 'boom')).toMatchObject({ status: 'completed', error: null })
    expect(initialJobProgress()).toMatchObject({ status: 'queued', error: null })
  })
})
