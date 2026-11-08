import { PURPOSE_REQUIRED_STATUS } from '@infra/ui'
import { ApiError, SIGNED_OUT_STATUS } from './client'

export type ErrorKey =
  | 'common.error_request'
  | 'common.error_unknown'
  | 'common.error_network'
  | 'common.error_purpose_required'
  | 'common.error_signed_out'

export type ErrorDescriptor = {
  key: ErrorKey
  vars?: Record<string, string | number>
}

/** The only sanctioned path from a thrown error to user-visible text.
 *  Never render err.message, err.detail, or response bodies. */
export function describeError(err: unknown): ErrorDescriptor {
  if (err instanceof ApiError) {
    // Dev-only visibility; the body is generic post-backend-fix anyway.
    console.debug('API error detail', err.status, err.detail)
    // The request needed a stated purpose and the user dismissed the prompt:
    // say what is missing rather than "something went wrong".
    if (err.status === PURPOSE_REQUIRED_STATUS) return { key: 'common.error_purpose_required' }
    // The gateway's sign-in session has expired: only signing in again helps.
    if (err.status === SIGNED_OUT_STATUS) return { key: 'common.error_signed_out' }
    return { key: 'common.error_request', vars: { status: err.status } }
  }
  if (err instanceof TypeError) {
    // fetch() network-level failure (server unreachable, DNS, CORS)
    return { key: 'common.error_network' }
  }
  return { key: 'common.error_unknown' }
}
