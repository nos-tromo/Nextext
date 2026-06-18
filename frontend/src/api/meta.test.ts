import { afterEach, describe, expect, it, vi } from 'vitest'
import { getHealth, getLanguages } from './meta'

afterEach(() => vi.restoreAllMocks())

function mockJson(body: unknown) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () =>
      new Response(JSON.stringify(body), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      }),
    ),
  )
}

describe('meta endpoints', () => {
  it('getHealth returns the decoded health payload', async () => {
    mockJson({ status: 'ok', inference: true, version: '1.2.3' })
    const health = await getHealth()
    expect(health).toEqual({ status: 'ok', inference: true, version: '1.2.3' })
  })

  it('getLanguages returns both language lists', async () => {
    mockJson({
      whisper: [{ code: 'en', name: 'English' }],
      target: [{ code: 'de-DE', name: 'German (Germany)' }],
      default_target: 'en',
    })
    const langs = await getLanguages()
    expect(langs.whisper[0].code).toBe('en')
    expect(langs.target[0].code).toBe('de-DE')
    expect(langs.default_target).toBe('en')
  })
})
