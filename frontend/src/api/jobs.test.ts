import { afterEach, describe, expect, it, vi } from 'vitest'
import { deleteJob, getJob, jobEventsPath, listJobs, submitJob } from './jobs'
import { OWNER_STORAGE_KEY } from '../identity/owner'
import type { JobOptions } from './types'

const OPTS: JobOptions = {
  src_lang: null,
  trg_lang: 'de',
  task: 'transcribe',
  diarize: true,
  words: false,
  summarization: false,
  hate_speech: false,
}

afterEach(() => {
  localStorage.clear()
  window.history.replaceState({}, '', '/')
  vi.restoreAllMocks()
})

function stub(status: number, body: unknown) {
  const fn = vi.fn(async () =>
    new Response(body === null ? '' : JSON.stringify(body), {
      status,
      headers: body === null ? {} : { 'content-type': 'application/json' },
    }),
  )
  vi.stubGlobal('fetch', fn)
  return fn
}

describe('jobs api', () => {
  it('jobEventsPath builds the SSE path', () => {
    expect(jobEventsPath('j1')).toBe('/jobs/j1/events')
  })

  it('submitJob posts multipart with file + options and returns the id', async () => {
    localStorage.setItem(OWNER_STORAGE_KEY, 'a'.repeat(32))
    const fetchFn = stub(201, { job_id: 'j1', status: 'queued', created_at: 't' })
    const file = new File([new Uint8Array([1, 2, 3])], 'clip.wav', { type: 'audio/wav' })

    const res = await submitJob('clip.wav', file, OPTS)

    expect(res.job_id).toBe('j1')
    const [url, init] = fetchFn.mock.calls[0] as unknown as [string, RequestInit]
    expect(url).toBe('/api/v1/jobs')
    expect(init.method).toBe('POST')
    expect(init.body).toBeInstanceOf(FormData)
    const form = init.body as FormData
    expect((form.get('file') as File).name).toBe('clip.wav')
    expect(JSON.parse(form.get('options') as string)).toMatchObject({ task: 'transcribe', trg_lang: 'de' })
    // multipart: the browser sets content-type with boundary, so we must NOT set it
    expect(init.headers).not.toHaveProperty('content-type')
  })

  it('listJobs and getJob GET the right paths', async () => {
    const fetchFn = stub(200, { jobs: [] })
    await listJobs()
    expect((fetchFn.mock.calls[0] as unknown as [string])[0]).toBe('/api/v1/jobs')

    stub(200, { job_id: 'j1', status: 'completed' })
    const snap = await getJob('j1')
    expect(snap.job_id).toBe('j1')
  })

  it('deleteJob tolerates a 204 empty body', async () => {
    stub(204, null)
    await expect(deleteJob('j1')).resolves.toBeNull()
  })
})
