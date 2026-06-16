import { DownloadButtons } from './DownloadButtons'
import { cn } from '../../lib/cn'
import type { HateSpeechFinding, JobResult } from '../../api/types'

interface HateSpeechTabProps {
  jobId: string
  result: JobResult
}

const CONFIDENCE_CLASSES: Record<HateSpeechFinding['confidence'], string> = {
  high: 'bg-danger/20 text-danger',
  medium: 'bg-amber-500/20 text-amber-400',
  low: 'bg-muted text-muted-foreground',
}

/**
 * Displays per-segment hate-speech detection findings in a card list, with
 * confidence badges and CSV/XLSX download buttons.
 * Renders a placeholder when no findings are present.
 *
 * @param jobId - The job identifier, forwarded to {@link DownloadButtons}.
 * @param result - The completed job result containing the hate-speech findings.
 */
export function HateSpeechTab({ jobId, result }: HateSpeechTabProps) {
  if (!result.hate_speech_findings || result.hate_speech_findings.length === 0) {
    return <p className="text-sm text-muted-foreground">No hate-speech findings for this job.</p>
  }

  const flagged = result.hate_speech_findings.filter((f) => f.hate_speech)

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        {flagged.length} of {result.hate_speech_findings.length} segment
        {result.hate_speech_findings.length === 1 ? '' : 's'} flagged.
      </p>
      <ul className="space-y-3">
        {result.hate_speech_findings.map((finding, i) => (
          <li key={i} className="rounded-md border border-border p-4">
            <div className="flex items-center gap-2">
              <span
                className={cn(
                  'rounded px-1.5 py-0.5 text-xs font-medium',
                  finding.hate_speech ? 'bg-danger/20 text-danger' : 'bg-muted text-muted-foreground',
                )}
              >
                {finding.hate_speech ? 'Flagged' : 'Clean'}
              </span>
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
            {finding.start !== null && (
              <p className="mt-1 text-xs text-muted-foreground">{finding.start}</p>
            )}
            <p className="mt-2 text-sm text-foreground">{finding.text}</p>
            {finding.hate_speech && finding.reason && (
              <p className="mt-1 text-sm text-muted-foreground italic">{finding.reason}</p>
            )}
          </li>
        ))}
      </ul>
      <DownloadButtons
        jobId={jobId}
        items={[
          { name: 'hate_speech.csv', label: 'CSV', fileName: 'hate_speech.csv' },
          { name: 'hate_speech.xlsx', label: 'XLSX', fileName: 'hate_speech.xlsx' },
        ]}
      />
    </div>
  )
}
