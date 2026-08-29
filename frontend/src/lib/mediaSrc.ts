import { API_BASE } from '../api/client'

/** The path prefix the backend uses in the URLs it hands out. */
const BACKEND_API_PREFIX = '/api/v1'

/**
 * Rebase a backend-issued media URL onto the SPA's own API base.
 *
 * The backend answers with a server-relative path (`/api/v1/jobs/…`) and knows
 * nothing about the sub-path the SPA is mounted under. Every other request
 * goes through `API_BASE`, which carries that prefix — but a `<video src>` is
 * resolved by the browser against the document, so handing it the backend's
 * path verbatim would request `/api/v1/…` and 404 wherever the app is not at
 * the root (i.e. in production, under `/nextext/`).
 *
 * @param mediaUrl - The `media_url` from a job result.
 * @param apiBase - API base to rebase onto; defaults to the app's own.
 * @returns A URL the browser can load from wherever the SPA is mounted.
 *   Absolute URLs are returned unchanged.
 */
export function mediaSrcUrl(mediaUrl: string, apiBase: string = API_BASE): string {
  if (!mediaUrl.startsWith(BACKEND_API_PREFIX)) return mediaUrl
  return `${apiBase}${mediaUrl.slice(BACKEND_API_PREFIX.length)}`
}
