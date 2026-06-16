import { afterEach, describe, expect, it, vi } from 'vitest'
import { OWNER_STORAGE_KEY, resolveOwnerId } from './owner'

function setUrl(search: string) {
  window.history.replaceState({}, '', `/${search}`)
}

afterEach(() => {
  localStorage.clear()
  setUrl('')
  vi.restoreAllMocks()
})

describe('resolveOwnerId', () => {
  it('mints a 32-char hex id, persists it, and writes it to the URL', () => {
    const id = resolveOwnerId()
    expect(id).toMatch(/^[0-9a-f]{32}$/)
    expect(localStorage.getItem(OWNER_STORAGE_KEY)).toBe(id)
    expect(new URLSearchParams(window.location.search).get('owner')).toBe(id)
  })

  it('reuses the id from the URL when present', () => {
    const existing = 'a'.repeat(32)
    setUrl(`?owner=${existing}`)
    expect(resolveOwnerId()).toBe(existing)
    expect(localStorage.getItem(OWNER_STORAGE_KEY)).toBe(existing)
  })

  it('reuses the id from localStorage when the URL has none', () => {
    const existing = 'b'.repeat(32)
    localStorage.setItem(OWNER_STORAGE_KEY, existing)
    expect(resolveOwnerId()).toBe(existing)
    expect(new URLSearchParams(window.location.search).get('owner')).toBe(existing)
  })

  it('ignores a malformed URL owner and mints a fresh one', () => {
    setUrl('?owner=not-valid')
    const id = resolveOwnerId()
    expect(id).toMatch(/^[0-9a-f]{32}$/)
    expect(id).not.toBe('not-valid')
  })
})
