import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError, API_BASE, OWNER_HEADER } from '../api/client'
import { fetchArtifact, fetchArtifactObjectUrl, downloadArtifact } from './download'

const mockBlob = new Blob(['hello'], { type: 'text/csv' })

function makeResponse(ok: boolean, status = ok ? 200 : 404, body: BodyInit = mockBlob): Response {
  return new Response(body, { status, headers: { 'content-type': 'text/csv' } })
}

describe('fetchArtifact', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })
  afterEach(() => vi.restoreAllMocks())

  it('calls the correct URL with the owner header', async () => {
    const fetchMock = vi.mocked(fetch)
    fetchMock.mockResolvedValueOnce(makeResponse(true))

    const res = await fetchArtifact('job1', 'transcript.csv')

    expect(fetchMock).toHaveBeenCalledOnce()
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(url).toBe(`${API_BASE}/jobs/job1/artifacts/transcript.csv`)
    expect((init.headers as Record<string, string>)[OWNER_HEADER]).toBeTruthy()
    expect(res.ok).toBe(true)
  })

  it('throws ApiError on non-2xx', async () => {
    vi.mocked(fetch).mockResolvedValueOnce(makeResponse(false, 404, JSON.stringify({ detail: 'not found' })))

    await expect(fetchArtifact('job1', 'missing.csv')).rejects.toThrow(ApiError)
  })
})

describe('fetchArtifactObjectUrl', () => {
  let createObjectUrl: ReturnType<typeof vi.spyOn>
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
    // Stub URL.createObjectURL/revokeObjectURL (happy-dom may behave differently
    // from a real browser; spying keeps the URL constructor intact).
    createObjectUrl = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:mock-url')
    vi.spyOn(URL, 'revokeObjectURL').mockReturnValue(undefined)
  })
  afterEach(() => vi.restoreAllMocks())

  it('returns a blob URL for a successful fetch', async () => {
    vi.mocked(fetch).mockResolvedValueOnce(makeResponse(true))

    const url = await fetchArtifactObjectUrl('job1', 'transcript.csv')

    expect(createObjectUrl).toHaveBeenCalledWith(expect.any(Blob))
    expect(url).toBe('blob:mock-url')
  })
})

describe('downloadArtifact', () => {
  let revokeObjectUrl: ReturnType<typeof vi.spyOn>
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:mock-url')
    revokeObjectUrl = vi.spyOn(URL, 'revokeObjectURL').mockReturnValue(undefined)
  })
  afterEach(() => vi.restoreAllMocks())

  it('creates and clicks a hidden anchor then revokes the object URL', async () => {
    vi.mocked(fetch).mockResolvedValueOnce(makeResponse(true))

    const anchor = {
      href: '',
      download: '',
      style: { display: '' },
      click: vi.fn(),
    }
    const createElementSpy = vi.spyOn(document, 'createElement').mockReturnValueOnce(anchor as unknown as HTMLAnchorElement)
    const appendChildSpy = vi.spyOn(document.body, 'appendChild').mockImplementation((n) => n)
    const removeChildSpy = vi.spyOn(document.body, 'removeChild').mockImplementation((n) => n)

    await downloadArtifact('job1', 'transcript.csv', 'my_transcript.csv')

    expect(createElementSpy).toHaveBeenCalledWith('a')
    expect(anchor.href).toBe('blob:mock-url')
    expect(anchor.download).toBe('my_transcript.csv')
    expect(anchor.click).toHaveBeenCalledOnce()
    expect(revokeObjectUrl).toHaveBeenCalledWith('blob:mock-url')
    expect(appendChildSpy).toHaveBeenCalled()
    expect(removeChildSpy).toHaveBeenCalled()
  })
})
