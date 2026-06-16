export const OWNER_STORAGE_KEY = 'nextext-owner'
const OWNER_RE = /^[0-9a-f]{32}$/

function mint(): string {
  const bytes = new Uint8Array(16)
  crypto.getRandomValues(bytes)
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')
}

function writeUrl(id: string): void {
  const url = new URL(window.location.href)
  if (url.searchParams.get('owner') !== id) {
    url.searchParams.set('owner', id)
    window.history.replaceState({}, '', url)
  }
}

/**
 * Resolve the per-browser owner id, preferring (1) a valid `?owner=` URL param,
 * then (2) localStorage, then (3) a freshly minted id. The resolved id is
 * always persisted to both localStorage and the URL.
 */
export function resolveOwnerId(): string {
  const fromUrl = new URLSearchParams(window.location.search).get('owner')
  const fromStore = localStorage.getItem(OWNER_STORAGE_KEY)
  const id =
    (fromUrl && OWNER_RE.test(fromUrl) && fromUrl) ||
    (fromStore && OWNER_RE.test(fromStore) && fromStore) ||
    mint()
  localStorage.setItem(OWNER_STORAGE_KEY, id)
  writeUrl(id)
  return id
}
