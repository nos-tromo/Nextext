import { DownloadButtons } from './DownloadButtons'
import { transcriptHasSpeaker, transcriptHasTranslation } from '../../lib/transcriptTable'
import { useT } from '../../i18n/LanguageContext'
import { useMediaPlayerStore } from '../../lib/mediaPlayerStore'
import { TimeCell } from './TimeCell'
import { cn } from '../../lib/cn'
import type { TranscriptSegment } from '../../api/types'

interface TranscriptTabProps {
  jobId: string
  segments: TranscriptSegment[]
  stem: string
  fileName: string
  mediaUrl: string | null
}

/**
 * Displays the transcript as a time-stamped table with an optional Speaker
 * column (shown only when at least one segment carries a speaker label) and
 * an optional Translation column (shown only when at least one segment
 * carries a translation), so the original transcript and its translation can
 * be cross-referenced side by side. Provides CSV, XLSX, TXT (or Transcript
 * TXT + Translation TXT if translation exists), and JSONL download buttons
 * with stem-prefixed filenames.
 *
 * @param jobId - The job identifier, forwarded to {@link DownloadButtons}.
 * @param segments - Transcript segments from the completed job result.
 * @param stem - Upload filename without extension; used to prefix download names.
 * @param fileName - Original upload name, shown as the player's title.
 * @param mediaUrl - Capability URL for playback; `null` disables the
 *   timestamp controls (the upload is gone, so there is nothing to play).
 */
export function TranscriptTab({ jobId, segments, stem, fileName, mediaUrl }: TranscriptTabProps) {
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

  const txtItems = hasTranslation
    ? [
        { name: 'transcript.txt', label: t('downloads.transcript_txt'), fileName: `${stem}_transcript.txt` },
        { name: 'translation.txt', label: t('downloads.translation_txt'), fileName: `${stem}_translation.txt` },
      ]
    : [{ name: 'transcript.txt', label: 'TXT', fileName: `${stem}_transcript.txt` }]

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
                key={i}
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
      <DownloadButtons
        jobId={jobId}
        items={[
          { name: 'transcript.csv', label: 'CSV', fileName: `${stem}_transcript.csv` },
          { name: 'transcript.xlsx', label: 'XLSX', fileName: `${stem}_transcript.xlsx` },
          ...txtItems,
          { name: 'docint.jsonl', label: 'JSONL', fileName: `${stem}_docint.jsonl` },
        ]}
      />
    </div>
  )
}
