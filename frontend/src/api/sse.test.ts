import { afterEach, describe, expect, it, vi } from 'vitest'
import { clearAllPurposes, encodePurpose, PURPOSE_HEADER, setPurpose } from '@infra/ui'
import { OWNER_HEADER, PURPOSE_SCOPE } from './client'
import { parseSseFrames, streamSse, type SseFrame } from './sse'

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

describe('streamSse', () => {
  afterEach(() => {
    clearAllPurposes()
    sessionStorage.clear()
    vi.restoreAllMocks()
  })

  it('reads the stream through the API client, with the identity and the stated purpose', async () => {
    setPurpose(PURPOSE_SCOPE, { caseRef: 'case-0001' })
    const fetchFn = vi.fn(
      async () => new Response('event: job_completed\ndata: {"job_id":"j1"}\n\n', { status: 200 }),
    )
    vi.stubGlobal('fetch', fetchFn)

    const frames: SseFrame[] = []
    for await (const frame of streamSse('/jobs/events')) frames.push(frame)

    expect(frames).toEqual([{ event: 'job_completed', data: '{"job_id":"j1"}' }])
    const [url, init] = fetchFn.mock.calls[0] as unknown as [string, RequestInit]
    expect(url).toBe('/api/v1/jobs/events')
    const headers = init.headers as Record<string, string>
    expect(headers.accept).toBe('text/event-stream')
    expect(headers).toHaveProperty(OWNER_HEADER)
    expect(headers[PURPOSE_HEADER]).toBe(encodePurpose({ caseRef: 'case-0001' }))
  })
})
