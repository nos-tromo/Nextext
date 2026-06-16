import { describe, expect, it } from 'vitest'
import { initialJobProgress, reduceJobEvent } from './jobProgress'
import type { JobEvent } from '../api/types'

const started = (i: number, p: number): JobEvent => ({
  name: 'stage_started',
  data: { stage: `S${i}`, stage_index: i, progress: p, timestamp: 't' },
})
const completed = (i: number, p: number): JobEvent => ({
  name: 'stage_completed',
  data: { stage: `S${i}`, stage_index: i, progress: p, timestamp: 't', result_delta: null },
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
    let s = reduceJobEvent(initialJobProgress(), {
      name: 'job_completed',
      data: { job_id: 'j1', skipped: true, timestamp: 't' },
    })
    expect(s).toMatchObject({ status: 'completed', progress: 1, skipped: true, terminal: true })
  })

  it('marks failed + carries error', () => {
    const s = reduceJobEvent(initialJobProgress(), {
      name: 'job_failed',
      data: { job_id: 'j1', error: 'boom', timestamp: 't' },
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
})
