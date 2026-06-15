import { describe, expect, it } from 'vitest'
import { parseSseFrames, type SseFrame } from './sse'

function framesFrom(chunks: string[]): SseFrame[] {
  const out: SseFrame[] = []
  const parser = parseSseFrames()
  for (const chunk of chunks) out.push(...parser.push(chunk))
  out.push(...parser.flush())
  return out
}

describe('parseSseFrames', () => {
  it('parses a complete event with name and JSON data', () => {
    const frames = framesFrom([
      'event: stage_completed\ndata: {"stage":"Transcribing","stage_index":0}\n\n',
    ])
    expect(frames).toEqual([
      { event: 'stage_completed', data: '{"stage":"Transcribing","stage_index":0}' },
    ])
  })

  it('joins data across chunk boundaries split mid-frame', () => {
    const frames = framesFrom(['event: job_completed\nda', 'ta: {"job_id":"j1"}\n\n'])
    expect(frames).toEqual([{ event: 'job_completed', data: '{"job_id":"j1"}' }])
  })

  it('ignores heartbeat comment lines', () => {
    const frames = framesFrom([': ping\n\n', 'event: x\ndata: 1\n\n'])
    expect(frames).toEqual([{ event: 'x', data: '1' }])
  })

  it('concatenates multiple data lines with newlines', () => {
    const frames = framesFrom(['event: e\ndata: a\ndata: b\n\n'])
    expect(frames).toEqual([{ event: 'e', data: 'a\nb' }])
  })
})
