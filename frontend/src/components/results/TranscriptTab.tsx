import { transcriptHasSpeaker, transcriptHasTranslation } from '../../lib/transcriptTable'
import { useT } from '../../i18n/LanguageContext'
import { useMediaPlayerStore } from '../../lib/mediaPlayerStore'
import { useFollowActiveRow } from '../../hooks/useFollowActiveRow'
import { TimeCell } from './TimeCell'
import { cn } from '../../lib/cn'
import type { TranscriptSegment } from '../../api/types'

interface TranscriptTabProps {
  jobId: string
  segments: TranscriptSegment[]
  fileName: string
  mediaUrl: string | null
}

/**
 * Displays the transcript as a time-stamped table with an optional Speaker
 * column (shown only when at least one segment carries a speaker label) and
 * an optional Translation column (shown only when at least one segment
 * carries a translation), so the original transcript and its translation can
 * be cross-referenced side by side. Its downloads sit in the result panel's
 * header (see `lib/resultDownloads.ts`).
 *
 * @param jobId - The job identifier; its recording is what the timestamps seek in.
 * @param segments - Transcript segments from the completed job result.
 * @param fileName - Original upload name, shown as the player's title.
 * @param mediaUrl - Capability URL for playback; `null` disables the
 *   timestamp controls (the upload is gone, so there is nothing to play).
 */
export function TranscriptTab({ jobId, segments, fileName, mediaUrl }: TranscriptTabProps) {
  const t = useT()
  const playingHere = useMediaPlayerStore((s) => s.session?.jobId === jobId)
  const currentTime = useMediaPlayerStore((s) => s.currentTime)
  const hasSpeaker = transcriptHasSpeaker(segments)
  const hasTranslation = transcriptHasTranslation(segments)

  /** The row the playhead sits in, or -1 when this job is not playing. */
  const activeIndex = playingHere
    ? segments.findIndex(
        (seg, i) =>
          seg.start_seconds !== null &&
          currentTime >= seg.start_seconds &&
          currentTime < (seg.end_seconds ?? segments[i + 1]?.start_seconds ?? Infinity),
      )
    : -1
  const activeRowRef = useFollowActiveRow(activeIndex)

  return (
    <div className="space-y-4">
      <div className="overflow-x-auto rounded-md border border-border">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border bg-muted text-muted-foreground">
              <th className="px-4 py-2 text-left font-medium">{t('results.col_start')}</th>
              <th className="px-4 py-2 text-left font-medium">{t('results.col_end')}</th>
              {hasSpeaker && <th className="px-4 py-2 text-left font-medium">{t('results.col_speaker')}</th>}
              <th className="px-4 py-2 text-left font-medium">
                {hasTranslation ? t('results.col_transcript') : t('results.col_text')}
              </th>
              {hasTranslation && <th className="px-4 py-2 text-left font-medium">{t('results.col_translation')}</th>}
            </tr>
          </thead>
          <tbody>
            {segments.map((seg, i) => (
              <tr
                // Transcript segments are an ordered sequence with no id, rendered from
                // an immutable result that never reorders, and the rows hold no state —
                // the index is the stable identity.
                // eslint-disable-next-line @eslint-react/no-array-index-key
                key={i}
                ref={i === activeIndex ? activeRowRef : undefined}
                aria-current={i === activeIndex ? 'true' : undefined}
                className={cn(
                  'border-b border-border last:border-0 hover:bg-accent/40',
                  i === activeIndex && 'bg-accent/60',
                )}
              >
                <TimeCell
                  label={seg.start ?? '—'}
                  seconds={seg.start_seconds}
                  jobId={jobId}
                  fileName={fileName}
                  mediaUrl={mediaUrl}
                />
                <td className="px-4 py-2 tabular-nums text-muted-foreground">{seg.end ?? '—'}</td>
                {hasSpeaker && (
                  <td className="px-4 py-2 text-muted-foreground">{seg.speaker ?? '—'}</td>
                )}
                <td className="px-4 py-2 text-foreground">{seg.text}</td>
                {hasTranslation && (
                  <td className="px-4 py-2 text-foreground">{seg.translation || '—'}</td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
