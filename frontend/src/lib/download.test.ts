import { createElement } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { clearAllPurposes, encodePurpose, PURPOSE_HEADER, PurposeBar, setPurpose } from '@infra/ui'
import { ApiError, API_BASE, OWNER_HEADER, PURPOSE_SCOPE } from '../api/client'
import {
  fetchArtifact,
  fetchArtifactObjectUrl,
  downloadArtifact,
  fetchBatchArtifact,
  downloadBatchArtifact,
} from './download'

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

describe('fetchBatchArtifact', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })
  afterEach(() => vi.restoreAllMocks())

  it('calls the batch URL with the owner header', async () => {
    const fetchMock = vi.mocked(fetch)
    fetchMock.mockResolvedValueOnce(makeResponse(true))

    const res = await fetchBatchArtifact('docint.jsonl')

    expect(fetchMock).toHaveBeenCalledOnce()
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(url).toBe(`${API_BASE}/jobs/batch/docint.jsonl`)
    expect((init.headers as Record<string, string>)[OWNER_HEADER]).toBeTruthy()
    expect(res.ok).toBe(true)
  })

  it('throws ApiError on non-2xx', async () => {
    vi.mocked(fetch).mockResolvedValueOnce(makeResponse(false, 404, JSON.stringify({ detail: 'no jobs' })))

    await expect(fetchBatchArtifact('archive.zip')).rejects.toThrow(ApiError)
  })
})

describe('downloadBatchArtifact', () => {
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
    const createElementSpy = vi
      .spyOn(document, 'createElement')
      .mockReturnValueOnce(anchor as unknown as HTMLAnchorElement)
    vi.spyOn(document.body, 'appendChild').mockImplementation((n) => n)
    vi.spyOn(document.body, 'removeChild').mockImplementation((n) => n)

    await downloadBatchArtifact('archive.zip', 'nextext_batch.zip')

    expect(createElementSpy).toHaveBeenCalledWith('a')
    expect(anchor.href).toBe('blob:mock-url')
    expect(anchor.download).toBe('nextext_batch.zip')
    expect(anchor.click).toHaveBeenCalledOnce()
    expect(revokeObjectUrl).toHaveBeenCalledWith('blob:mock-url')
  })
})

describe('downloads and the stated purpose', () => {
  // The anchor a download clicks, read off the real element rather than a
  // mocked document.createElement, which React needs for the purpose bar.
  let clicked: { href: string; download: string }[]
  beforeEach(() => {
    clicked = []
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:mock-url')
    vi.spyOn(URL, 'revokeObjectURL').mockReturnValue(undefined)
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      clicked.push({ href: this.href, download: this.download })
    })
  })
  afterEach(() => {
    clearAllPurposes()
    sessionStorage.clear()
    vi.restoreAllMocks()
  })

  it('sends the stated purpose with an artifact download and keeps its file name', async () => {
    setPurpose(PURPOSE_SCOPE, { caseRef: 'case-0001' })
    const fetchMock = vi.fn(async () => makeResponse(true))
    vi.stubGlobal('fetch', fetchMock)

    await downloadArtifact('job1', 'transcript.csv', 'clip_transcript.csv')

    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect(url).toBe(`${API_BASE}/jobs/job1/artifacts/transcript.csv`)
    expect((init.headers as Record<string, string>)[PURPOSE_HEADER]).toBe(encodePurpose({ caseRef: 'case-0001' }))
    expect(clicked).toEqual([{ href: 'blob:mock-url', download: 'clip_transcript.csv' }])
  })

  it('asks for a purpose when a batch download is refused with 428, then downloads', async () => {
    render(createElement(PurposeBar, { scope: PURPOSE_SCOPE, required: true }))
    const statuses = [428, 200]
    const fetchMock = vi.fn(async () => {
      const status = statuses.shift() ?? 200
      return makeResponse(status === 200, status)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()

    const pending = downloadBatchArtifact('archive.zip', 'nextext_batch.zip')
    await screen.findByRole('dialog')
    await user.type(screen.getByLabelText('Case reference'), 'case-0001')
    await user.click(screen.getByRole('button', { name: 'Save' }))
    await pending

    expect(fetchMock).toHaveBeenCalledTimes(2)
    const [url, init] = fetchMock.mock.calls[1] as unknown as [string, RequestInit]
    expect(url).toBe(`${API_BASE}/jobs/batch/archive.zip`)
    expect((init.headers as Record<string, string>)[PURPOSE_HEADER]).toBe(encodePurpose({ caseRef: 'case-0001' }))
    expect(clicked).toEqual([{ href: 'blob:mock-url', download: 'nextext_batch.zip' }])
  })
})
