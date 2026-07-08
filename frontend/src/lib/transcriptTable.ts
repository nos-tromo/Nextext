import type { TranscriptSegment } from '../api/types'

/**
 * Return `true` when at least one segment in the transcript has a non-null,
 * non-empty speaker label. Used to decide whether to render the Speaker column.
 *
 * @param segments - The transcript segment array from a completed job result.
 * @returns Whether any segment carries a speaker label.
 */
export function transcriptHasSpeaker(segments: TranscriptSegment[]): boolean {
  return segments.some((s) => s.speaker != null && s.speaker !== '')
}

/**
 * Return `true` when at least one segment in the transcript has a non-null,
 * non-empty translation. Used to decide whether to render the Translation
 * column alongside the original transcript text, so the two can be
 * cross-referenced side by side.
 *
 * @param segments - The transcript segment array from a completed job result.
 * @returns Whether any segment carries a translation.
 */
export function transcriptHasTranslation(segments: TranscriptSegment[]): boolean {
  return segments.some((s) => s.translation != null && s.translation !== '')
}
