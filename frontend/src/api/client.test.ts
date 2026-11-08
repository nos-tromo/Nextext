import { createElement } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { clearAllPurposes, encodePurpose, PURPOSE_HEADER, PurposeBar, setPurpose } from '@infra/ui'
import { apiBase, ApiError, apiFetch, apiGet, apiRequest, apiSend, OWNER_HEADER, PURPOSE_SCOPE } from './client'
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
    const [url, init] = fetchFn.mock.calls[0] as unknown as [string, RequestInit]
    expect(url).toBe('/api/v1/health')
    expect((init).headers).toMatchObject({
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
    const [, init] = fetchFn.mock.calls[0] as unknown as [string, RequestInit]
    const ri = init
    expect(ri.method).toBe('POST')
    expect(ri.body).toBe(JSON.stringify({ a: 1 }))
    expect(ri.headers).toMatchObject({ 'content-type': 'application/json' })
  })

  it('passes a FormData body through without a content-type header', async () => {
    const fetchFn = mockFetch(201, { job_id: 'j1' })
    const fd = new FormData()
    fd.append('x', 'y')
    await apiSend('POST', '/jobs', { form: fd })
    const [, init] = fetchFn.mock.calls[0] as unknown as [string, RequestInit]
    const ri = init
    expect(ri.body).toBe(fd)
    expect(ri.headers).not.toHaveProperty('content-type')
  })
})

describe('the gateway sign-in page', () => {
  const signInPage = '<!doctype html><title>Sign in</title><form></form>'

  /** A response as `fetch` hands it back, marked redirected when it followed one. */
  function respond(body: BodyInit, contentType: string, { status = 200, redirected = false } = {}): Response {
    const res = new Response(body, { status, headers: { 'content-type': contentType } })
    if (redirected) Object.defineProperty(res, 'redirected', { value: true })
    return res
  }

  function fetchReturning(res: Response) {
    const fn = vi.fn(async () => res)
    vi.stubGlobal('fetch', fn)
    return fn
  }

  it('fails as signed out when fetch followed the redirect to it, and keeps the page out of the error', async () => {
    fetchReturning(respond(signInPage, 'text/html; charset=utf-8', { redirected: true }))

    await expect(apiFetch('/jobs/j1/artifacts/archive.zip')).rejects.toMatchObject({
      name: 'ApiError',
      status: 401,
      detail: null,
    } satisfies Partial<ApiError>)
  })

  it('fails as signed out when it answers without a redirect', async () => {
    fetchReturning(respond(signInPage, 'text/html'))

    await expect(apiRequest('/jobs')).rejects.toMatchObject({ name: 'ApiError', status: 401, detail: null })
  })

  it('fails a JSON read as signed out instead of returning the page as text', async () => {
    fetchReturning(respond(signInPage, 'text/html', { redirected: true }))

    await expect(apiGet('/jobs')).rejects.toMatchObject({ name: 'ApiError', status: 401 })
  })

  it('fails as signed out on any redirect: the backend never sends one', async () => {
    fetchReturning(respond('{}', 'application/json', { redirected: true }))

    await expect(apiSend('DELETE', '/jobs/j1')).rejects.toMatchObject({ name: 'ApiError', status: 401 })
  })

  it('keeps the status of an HTML error page, which is not the sign-in page', async () => {
    fetchReturning(respond('<html><body>502 Bad Gateway</body></html>', 'text/html', { status: 502 }))

    await expect(apiFetch('/jobs/j1/artifacts/archive.zip')).rejects.toMatchObject({ name: 'ApiError', status: 502 })
  })

  it('passes a JSON answer through', async () => {
    fetchReturning(respond(JSON.stringify({ jobs: [] }), 'application/json'))

    await expect(apiGet('/jobs')).resolves.toEqual({ jobs: [] })
  })

  it.each(['image/jpeg', 'application/zip', 'text/csv; charset=utf-8'])('passes a %s download through', async (type) => {
    fetchReturning(respond(new Blob([new Uint8Array([1, 2, 3])]), type))

    const res = await apiFetch('/jobs/j1/artifacts/some.bin')
    expect(res.ok).toBe(true)
    expect((await res.blob()).size).toBe(3)
  })
})

describe('apiBase', () => {
  it('derives the API base from a sub-path BASE_URL', () => {
    expect(apiBase('/nextext/')).toBe('/nextext/api/v1')
  })
  it('falls back to root for BASE_URL "/"', () => {
    expect(apiBase('/')).toBe('/api/v1')
  })
})

describe('the stated purpose', () => {
  const purpose = { caseRef: 'case-0001', reason: 'Review of a reported clip' }

  afterEach(() => {
    clearAllPurposes()
    sessionStorage.clear()
  })

  /** Answer each fetch with the next status in turn (JSON bodies, fresh per call). */
  function fetchAnswering(...statuses: number[]) {
    const fn = vi.fn(async () => {
      const status = statuses.shift() ?? 200
      return new Response(JSON.stringify(status === 428 ? { detail: 'needs a purpose' } : { jobs: [] }), {
        status,
        headers: { 'content-type': 'application/json' },
      })
    })
    vi.stubGlobal('fetch', fn)
    return fn
  }

  function headersOf(fn: ReturnType<typeof fetchAnswering>, call: number): Record<string, string> {
    const [, init] = fn.mock.calls[call] as unknown as [string, RequestInit]
    return init.headers as Record<string, string>
  }

  /** Mount the header's bar, which a 428 opens. */
  function mountBar() {
    render(createElement(PurposeBar, { scope: PURPOSE_SCOPE, required: true }))
  }

  /** Save a reason in the bar's dialog, which a 428 opened. */
  async function answerPrompt() {
    const user = userEvent.setup()
    await screen.findByRole('dialog')
    await user.type(screen.getByLabelText('Reason'), purpose.reason)
    await user.click(screen.getByRole('button', { name: 'Save' }))
  }

  it('is sent with every request once stated, and not at all before', async () => {
    const fetchFn = fetchAnswering(200, 200)
    await apiGet('/jobs')
    setPurpose(PURPOSE_SCOPE, purpose)
    await apiSend('DELETE', '/jobs/j1')

    expect(headersOf(fetchFn, 0)).not.toHaveProperty(PURPOSE_HEADER)
    expect(headersOf(fetchFn, 1)[PURPOSE_HEADER]).toBe(encodePurpose(purpose))
    expect(headersOf(fetchFn, 1)[PURPOSE_HEADER]).toBe('case_ref=case-0001&reason=Review+of+a+reported+clip')
    // The identity still travels beside it.
    expect(headersOf(fetchFn, 1)).toHaveProperty(OWNER_HEADER)
  })

  it('is asked for through the header bar on a 428, and the request is sent once more with it', async () => {
    mountBar()
    const fetchFn = fetchAnswering(428, 200)

    const pending = apiGet<{ jobs: unknown[] }>('/jobs')
    await answerPrompt()

    await expect(pending).resolves.toEqual({ jobs: [] })
    expect(fetchFn).toHaveBeenCalledTimes(2)
    expect(headersOf(fetchFn, 0)).not.toHaveProperty(PURPOSE_HEADER)
    expect(headersOf(fetchFn, 1)[PURPOSE_HEADER]).toBe(encodePurpose({ reason: purpose.reason }))
  })

  it('is resent only once: a second 428 is reported, not asked again', async () => {
    mountBar()
    const fetchFn = fetchAnswering(428, 428)

    // Settled into a value at once, so the rejection is never left unhandled.
    const outcome = apiGet('/jobs').catch((err: unknown) => err)
    await answerPrompt()

    expect(await outcome).toMatchObject({ name: 'ApiError', status: 428 })
    expect(fetchFn).toHaveBeenCalledTimes(2)
  })

  it('reports the 428 when the prompt is cancelled, without sending again', async () => {
    mountBar()
    const fetchFn = fetchAnswering(428)
    const user = userEvent.setup()

    const outcome = apiGet('/jobs').catch((err: unknown) => err)
    await screen.findByRole('dialog')
    await user.click(screen.getByRole('button', { name: 'Cancel' }))

    expect(await outcome).toBeInstanceOf(ApiError)
    expect(fetchFn).toHaveBeenCalledTimes(1)
  })

  it('resends an upload with the same form body', async () => {
    mountBar()
    const fetchFn = fetchAnswering(428, 201)
    const form = new FormData()
    form.append('options', '{}')

    const pending = apiSend('POST', '/jobs', { form })
    await answerPrompt()
    await pending

    const bodies = fetchFn.mock.calls.map((call) => (call as unknown as [string, RequestInit])[1].body)
    expect(bodies).toEqual([form, form])
  })

  it('carries the purpose on a raw fetch, which throws ApiError on a non-2xx', async () => {
    setPurpose(PURPOSE_SCOPE, purpose)
    const fetchFn = fetchAnswering(200, 404)

    const res = await apiFetch('/jobs/j1/artifacts/transcript.csv')
    expect(res.ok).toBe(true)
    expect(headersOf(fetchFn, 0)[PURPOSE_HEADER]).toBe(encodePurpose(purpose))
    await expect(apiFetch('/jobs/j1/artifacts/missing.csv')).rejects.toMatchObject({ status: 404 })
  })
})
