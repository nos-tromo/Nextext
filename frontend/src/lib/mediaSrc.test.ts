import { describe, expect, it } from 'vitest'
import { mediaSrcUrl } from './mediaSrc'

describe('mediaSrcUrl', () => {
  it('rebases a backend URL onto the SPA sub-path', () => {
    // The backend answers with a server-relative path and knows nothing about
    // the /nextext/ mount, so a raw <video src> would resolve to /api/v1/…
    // and 404 in production.
    expect(mediaSrcUrl('/api/v1/jobs/j1/media?token=t', '/nextext/api/v1')).toBe(
      '/nextext/api/v1/jobs/j1/media?token=t',
    )
  })

  it('is a no-op when the app is mounted at the root', () => {
    expect(mediaSrcUrl('/api/v1/jobs/j1/media?token=t', '/api/v1')).toBe(
      '/api/v1/jobs/j1/media?token=t',
    )
  })

  it('preserves the token query verbatim', () => {
    const url = mediaSrcUrl('/api/v1/jobs/j1/media?token=a-b_c', '/nextext/api/v1')
    expect(url.endsWith('?token=a-b_c')).toBe(true)
  })

  it('leaves an already-absolute URL alone', () => {
    const abs = 'https://host.example/api/v1/jobs/j1/media?token=t'
    expect(mediaSrcUrl(abs, '/nextext/api/v1')).toBe(abs)
  })
})
