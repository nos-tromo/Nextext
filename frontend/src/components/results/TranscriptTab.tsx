import { DownloadButtons } from './DownloadButtons'
import { transcriptHasSpeaker, transcriptHasTranslation } from '../../lib/transcriptTable'
import type { TranscriptSegment } from '../../api/types'

interface TranscriptTabProps {
  jobId: string
  segments: TranscriptSegment[]
  stem: string
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
 */
export function TranscriptTab({ jobId, segments, stem }: TranscriptTabProps) {
  const hasSpeaker = transcriptHasSpeaker(segments)
  const hasTranslation = transcriptHasTranslation(segments)

  const txtItems = hasTranslation
    ? [
        { name: 'transcript.txt', label: 'Transcript TXT', fileName: `${stem}_transcript.txt` },
        { name: 'translation.txt', label: 'Translation TXT', fileName: `${stem}_translation.txt` },
      ]
    : [{ name: 'transcript.txt', label: 'TXT', fileName: `${stem}_transcript.txt` }]

  return (
    <div className="space-y-4">
      <div className="overflow-x-auto rounded-md border border-border">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border bg-muted text-muted-foreground">
              <th className="px-4 py-2 text-left font-medium">Start</th>
              <th className="px-4 py-2 text-left font-medium">End</th>
              {hasSpeaker && <th className="px-4 py-2 text-left font-medium">Speaker</th>}
              <th className="px-4 py-2 text-left font-medium">{hasTranslation ? 'Transcript' : 'Text'}</th>
              {hasTranslation && <th className="px-4 py-2 text-left font-medium">Translation</th>}
            </tr>
          </thead>
          <tbody>
            {segments.map((seg, i) => (
              <tr key={i} className="border-b border-border last:border-0 hover:bg-accent/40">
                <td className="px-4 py-2 tabular-nums text-muted-foreground">{seg.start ?? '—'}</td>
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
