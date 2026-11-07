/**
 * Formats an offset in the clip as `mm:ss`, with minutes growing past 60.
 *
 * Mirrors the backend's `_timestamp` so on-screen stamps match the
 * downloaded `visual_context.txt` exactly.
 *
 * @param seconds - Offset from the start of the clip.
 * @returns Zero-padded `mm:ss` stamp.
 */
export function formatStamp(seconds: number): string {
  const total = Number.isFinite(seconds) && seconds > 0 ? Math.floor(seconds) : 0
  const minutes = Math.floor(total / 60)
  return `${String(minutes).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`
}
