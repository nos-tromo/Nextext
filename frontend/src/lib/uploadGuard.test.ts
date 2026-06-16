import { describe, expect, it } from 'vitest'
import { checkUploadAcceptable, DEFAULT_MAX_FILE_MB } from './uploadGuard'

function file(name: string, bytes: number): File {
  return new File([new Uint8Array(bytes)], name)
}

describe('checkUploadAcceptable', () => {
  it('accepts files within the per-file cap', () => {
    expect(checkUploadAcceptable([file('a.wav', 10)], 1)).toBeNull()
  })

  it('rejects a file over the per-file cap with an actionable message', () => {
    const oneMb = 1 << 20
    const msg = checkUploadAcceptable([file('big.wav', 2 * oneMb)], 1)
    expect(msg).not.toBeNull()
    expect(msg).toContain('big.wav')
    expect(msg).toContain('nextext-cli')
  })

  it('accepts an empty selection', () => {
    expect(checkUploadAcceptable([], 1)).toBeNull()
  })

  it('exposes a sane default cap', () => {
    expect(DEFAULT_MAX_FILE_MB).toBeGreaterThan(0)
  })
})
