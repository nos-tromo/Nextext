import { DownloadButtons } from './DownloadButtons'
import { cn } from '../../lib/cn'
import { useT } from '../../i18n/LanguageContext'
import type { HateSpeechFinding, JobResult } from '../../api/types'

interface HateSpeechTabProps {
  jobId: string
  result: JobResult
  stem: string
}

const CONFIDENCE_CLASSES: Record<HateSpeechFinding['confidence'], string> = {
  high: 'bg-danger/20 text-danger',
  medium: 'bg-amber-500/20 text-amber-400',
  low: 'bg-muted text-muted-foreground',
}

/**
 * Displays hate-speech findings in a card list, with confidence badges and
 * CSV/XLSX download buttons. Transcript segments are counted against the
 * transcript; keyframe captions, judged for what a video shows, are counted
 * on their own and marked as video frames.
 * Renders a placeholder when no findings are present.
 *
 * @param jobId - The job identifier, forwarded to {@link DownloadButtons}.
 * @param result - The completed job result containing the hate-speech findings.
 * @param stem - Upload filename without extension; used to prefix download names.
 */
export function HateSpeechTab({ jobId, result, stem }: HateSpeechTabProps) {
  const t = useT()
  if (!result.hate_speech_findings || result.hate_speech_findings.length === 0) {
    return <p className="text-sm text-muted-foreground">{t('results.no_hate_speech')}</p>
  }

  const segmentFindings = result.hate_speech_findings.filter((f) => f.source !== 'frame')
  const frameFindings = result.hate_speech_findings.filter((f) => f.source === 'frame' && f.hate_speech)
  const flagged = segmentFindings.filter((f) => f.hate_speech)
  // Only flagged segments come back from the backend, so the denominator is the transcript.
  const total = result.transcript.length || segmentFindings.length

  return (
    <div className="space-y-4">
      <div className="space-y-1 text-sm text-muted-foreground">
        {total > 0 && (
          <p>
            {t(total === 1 ? 'results.flagged_summary_one' : 'results.flagged_summary_other', {
              flagged: flagged.length,
              total,
            })}
          </p>
        )}
        {frameFindings.length > 0 && (
          <p>
            {t(frameFindings.length === 1 ? 'results.flagged_frames_one' : 'results.flagged_frames_other', {
              count: frameFindings.length,
            })}
          </p>
        )}
      </div>
      <ul className="space-y-3">
        {result.hate_speech_findings.map((finding, i) => (
          // Findings carry no id, and the list renders from an immutable result that
          // never reorders; the rows hold no state, so the index is the identity.
          // eslint-disable-next-line @eslint-react/no-array-index-key
          <li key={i} className="rounded-md border border-border p-4">
            <div className="flex items-center gap-2">
              <span
                className={cn(
                  'rounded px-1.5 py-0.5 text-xs font-medium',
                  finding.hate_speech ? 'bg-danger/20 text-danger' : 'bg-muted text-muted-foreground',
                )}
              >
                {finding.hate_speech ? t('results.flagged') : t('results.clean')}
              </span>
              {finding.source === 'frame' && (
                <span className="rounded border border-border bg-muted px-1.5 py-px text-xs font-medium text-muted-foreground">
                  {t('results.source_frame')}
                </span>
              )}
              {finding.hate_speech && (
                <>
                  <span className="text-sm text-muted-foreground">{finding.category}</span>
                  <span
                    className={cn(
                      'ml-auto rounded px-1.5 py-0.5 text-xs font-medium',
                      CONFIDENCE_CLASSES[finding.confidence],
                    )}
                  >
                    {finding.confidence}
                  </span>
                </>
              )}
            </div>
            {(finding.start !== null || finding.speaker) && (
              <p className="mt-1 flex gap-2 text-xs text-muted-foreground">
                {finding.start !== null && <span>{finding.start}</span>}
                {finding.speaker && <span>{finding.speaker}</span>}
              </p>
            )}
            <p className="mt-2 text-sm text-foreground">{finding.text}</p>
            {finding.translation && (
              <p className="mt-1 text-sm text-muted-foreground">
                {t('results.col_translation')}: {finding.translation}
              </p>
            )}
            {finding.hate_speech && finding.reason && (
              <p className="mt-1 text-sm text-muted-foreground italic">{finding.reason}</p>
            )}
          </li>
        ))}
      </ul>
      <DownloadButtons
        jobId={jobId}
        items={[
          { name: 'hate_speech.csv', label: 'CSV', fileName: `${stem}_hate_speech.csv` },
          { name: 'hate_speech.xlsx', label: 'XLSX', fileName: `${stem}_hate_speech.xlsx` },
        ]}
      />
    </div>
  )
}
