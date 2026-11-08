import { withPurpose } from '@infra/ui'
import { resolveOwnerId } from '../identity/owner'

/** API base derived from Vite's base path so the SPA works under its
 *  canonical sub-path (/nextext/) and at root (dev) alike. */
export function apiBase(base: string = import.meta.env.BASE_URL): string {
  return `${base.replace(/\/+$/, '')}/api/v1`
}

export const API_BASE = apiBase()
/** Trusted identity header; mirrors the backend default NEXTEXT_AUTH_HEADER. */
export const OWNER_HEADER = 'X-Auth-User'
/** The app's purpose scope: the compose project name, shared with its PurposeBar. */
export const PURPOSE_SCOPE = 'nextext'
/** Status of the {@link ApiError} a request fails with when the gateway's sign-in page answered it. */
export const SIGNED_OUT_STATUS = 401

export class ApiError extends Error {
  readonly status: number
  readonly detail: unknown
  constructor(status: number, detail: unknown) {
    super(`API ${status}`)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

function identityHeaders(): Record<string, string> {
  return { [OWNER_HEADER]: resolveOwnerId() }
}

/** Fetch options for {@link apiRequest}; headers are a plain record, merged with the app's own. */
export type ApiRequestInit = Omit<RequestInit, 'headers'> & { headers?: Record<string, string> }

/**
 * Whether a response is the edge gateway's sign-in page rather than the backend's answer.
 *
 * Once the gateway session has expired, the gateway redirects a request to its
 * sign-in page and `fetch` follows, ending on an HTML page with a `200`. The
 * backend never redirects and never answers with HTML, so a followed redirect
 * or a successful HTML response means the request never reached it. An HTML
 * error page (a proxy's `502`, say) keeps its own status.
 *
 * @param res - The response `fetch` ended on.
 * @returns `true` when the response is not the backend's.
 */
export function isSignInPage(res: Response): boolean {
  const type = res.headers.get('content-type') ?? ''
  return res.redirected || (res.ok && type.toLowerCase().startsWith('text/html'))
}

/**
 * Send a request to the backend: the one place every API call goes through.
 *
 * Adds the identity header and the purpose stated in the header's
 * `PurposeBar` (`X-Audit-Purpose`); when the backend answers `428` — the
 * request needs a stated purpose — the bar asks for one and the request is
 * sent once more with it. `init` may therefore be used twice, so its body
 * must be one that can be sent again (a string or `FormData`).
 *
 * @param path - Path relative to {@link API_BASE}, e.g. `/jobs`.
 * @param init - Fetch options.
 * @returns The backend's raw response, whatever its status.
 * @throws {ApiError} With {@link SIGNED_OUT_STATUS} and no detail when the
 *   gateway's sign-in page answered instead (see {@link isSignInPage}), so no
 *   caller reads the page as data or saves it as a download.
 */
export async function apiRequest(path: string, init: ApiRequestInit = {}): Promise<Response> {
  const res = await withPurpose(PURPOSE_SCOPE, (purpose) =>
    fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { ...identityHeaders(), ...init.headers, ...purpose },
    }),
  )
  if (isSignInPage(res)) throw new ApiError(SIGNED_OUT_STATUS, null)
  return res
}

async function errorFrom(res: Response): Promise<ApiError> {
  const text = await res.text()
  let body: unknown = text
  try {
    body = text ? JSON.parse(text) : null
  } catch {
    /* keep raw text */
  }
  const detail =
    body && typeof body === 'object' && 'detail' in body
      ? (body as { detail: unknown }).detail
      : body
  return new ApiError(res.status, detail)
}

async function parse<T>(res: Response): Promise<T> {
  if (!res.ok) throw await errorFrom(res)
  const text = await res.text()
  try {
    return (text ? JSON.parse(text) : null) as T
  } catch {
    return text as T
  }
}

/**
 * Fetch a backend path and hand back the response for the caller to read —
 * a download's blob, say — throwing {@link ApiError} on a non-2xx, and with
 * {@link SIGNED_OUT_STATUS} when the gateway's sign-in page answered.
 *
 * @param path - Path relative to {@link API_BASE}.
 * @returns The successful response, its body unread.
 */
export async function apiFetch(path: string): Promise<Response> {
  const res = await apiRequest(path)
  if (!res.ok) throw await errorFrom(res)
  return res
}

export async function apiGet<T>(path: string, signal?: AbortSignal): Promise<T> {
  return parse<T>(await apiRequest(path, { signal }))
}

type SendOpts = { json?: unknown; form?: FormData; signal?: AbortSignal }

export async function apiSend<T>(
  method: 'POST' | 'PUT' | 'DELETE',
  path: string,
  opts: SendOpts = {},
): Promise<T> {
  const headers: Record<string, string> = {}
  let body: BodyInit | undefined
  if (opts.form) {
    body = opts.form // browser sets multipart content-type + boundary
  } else if (opts.json !== undefined) {
    headers['content-type'] = 'application/json'
    body = JSON.stringify(opts.json)
  }
  return parse<T>(await apiRequest(path, { method, headers, body, signal: opts.signal }))
}
