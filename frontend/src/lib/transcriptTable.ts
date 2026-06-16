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
