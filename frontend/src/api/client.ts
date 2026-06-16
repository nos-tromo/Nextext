import { resolveOwnerId } from '../identity/owner'

export const API_BASE = '/api/v1'
/** Trusted identity header; mirrors the backend default NEXTEXT_AUTH_HEADER. */
export const OWNER_HEADER = 'X-Auth-User'

export class ApiError extends Error {
  readonly status: number
  readonly detail: unknown
  constructor(status: number, detail: unknown) {
    super(`API ${status}: ${typeof detail === 'string' ? detail : JSON.stringify(detail)}`)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

function identityHeaders(): Record<string, string> {
  return { [OWNER_HEADER]: resolveOwnerId() }
}

async function parse<T>(res: Response): Promise<T> {
  const text = await res.text()
  let body: unknown = text
  try {
    body = text ? JSON.parse(text) : null
  } catch {
    /* keep raw text */
  }
  if (!res.ok) {
    const detail =
      body && typeof body === 'object' && 'detail' in body
        ? (body as { detail: unknown }).detail
        : body
    throw new ApiError(res.status, detail)
  }
  return body as T
}

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { headers: identityHeaders() })
  return parse<T>(res)
}

type SendOpts = { json?: unknown; form?: FormData; signal?: AbortSignal }

export async function apiSend<T>(
  method: 'POST' | 'PUT' | 'DELETE',
  path: string,
  opts: SendOpts = {},
): Promise<T> {
  const headers: Record<string, string> = { ...identityHeaders() }
  let body: BodyInit | undefined
  if (opts.form) {
    body = opts.form // browser sets multipart content-type + boundary
  } else if (opts.json !== undefined) {
    headers['content-type'] = 'application/json'
    body = JSON.stringify(opts.json)
  }
  const res = await fetch(`${API_BASE}${path}`, { method, headers, body, signal: opts.signal })
  return parse<T>(res)
}
