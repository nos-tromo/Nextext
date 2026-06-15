import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, apiGet, apiSend, OWNER_HEADER } from './client'
import { OWNER_STORAGE_KEY } from '../identity/owner'

afterEach(() => {
  localStorage.clear()
  window.history.replaceState({}, '', '/')
  vi.restoreAllMocks()
})

function mockFetch(status: number, body: unknown) {
  const fn = vi.fn(async () =>
    new Response(JSON.stringify(body), {
      status,
      headers: { 'content-type': 'application/json' },
    }),
  )
  vi.stubGlobal('fetch', fn)
  return fn
}

describe('apiGet', () => {
  it('calls the /api/v1-prefixed path with the owner identity header', async () => {
    localStorage.setItem(OWNER_STORAGE_KEY, 'a'.repeat(32))
    const fetchFn = mockFetch(200, { status: 'ok' })

    const body = await apiGet<{ status: string }>('/health')

    expect(body.status).toBe('ok')
    const [url, init] = fetchFn.mock.calls[0]
    expect(url).toBe('/api/v1/health')
    expect((init as RequestInit).headers).toMatchObject({
      [OWNER_HEADER]: 'a'.repeat(32),
    })
  })

  it('throws ApiError carrying status and detail on non-2xx', async () => {
    mockFetch(404, { detail: 'nope' })
    await expect(apiGet('/jobs/x')).rejects.toMatchObject({
      name: 'ApiError',
      status: 404,
      detail: 'nope',
    } satisfies Partial<ApiError>)
  })
})

describe('apiSend', () => {
  it('serializes a JSON body and sets content-type', async () => {
    const fetchFn = mockFetch(201, { job_id: 'j1' })
    await apiSend('POST', '/jobs', { json: { a: 1 } })
    const [, init] = fetchFn.mock.calls[0]
    const ri = init as RequestInit
    expect(ri.method).toBe('POST')
    expect(ri.body).toBe(JSON.stringify({ a: 1 }))
    expect(ri.headers).toMatchObject({ 'content-type': 'application/json' })
  })

  it('passes a FormData body through without a content-type header', async () => {
    const fetchFn = mockFetch(201, { job_id: 'j1' })
    const fd = new FormData()
    fd.append('x', 'y')
    await apiSend('POST', '/jobs', { form: fd })
    const [, init] = fetchFn.mock.calls[0]
    const ri = init as RequestInit
    expect(ri.body).toBe(fd)
    expect(ri.headers).not.toHaveProperty('content-type')
  })
})
