import { describe, expect, it } from 'vitest'
import { ApiError } from './client'
import { describeError } from './errorMessage'

describe('describeError', () => {
  it('maps ApiError to the generic request key with the status', () => {
    expect(describeError(new ApiError(502, 'upstream exploded: /srv/secret'))).toEqual({
      key: 'common.error_request',
      vars: { status: 502 },
    })
  })
  it('maps a 428 to the purpose-required key, naming what is missing', () => {
    expect(describeError(new ApiError(428, 'This request needs a stated purpose.'))).toEqual({
      key: 'common.error_purpose_required',
    })
  })
  it('maps a 401 to the signed-out key', () => {
    expect(describeError(new ApiError(401, null))).toEqual({ key: 'common.error_signed_out' })
  })
  it('maps network TypeError to the network key', () => {
    expect(describeError(new TypeError('Failed to fetch'))).toEqual({
      key: 'common.error_network',
    })
  })
  it('maps anything else to the unknown key', () => {
    expect(describeError('weird')).toEqual({ key: 'common.error_unknown' })
    expect(describeError(new Error('boom'))).toEqual({ key: 'common.error_unknown' })
  })
  it('never exposes detail in the ApiError message', () => {
    const e = new ApiError(500, { detail: 'stacktrace /etc/passwd' })
    expect(e.message).toBe('API 500')
  })
})
