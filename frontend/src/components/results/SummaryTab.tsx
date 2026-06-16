import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { DownloadButtons } from './DownloadButtons'
import type { JobResult } from '../../api/types'

interface SummaryTabProps {
  jobId: string
  result: JobResult
}

/**
 * Displays the job summary rendered as Markdown, plus a download button
 * for `summary.txt`. Renders nothing when the job produced no summary.
 *
 * @param jobId - The job identifier, forwarded to {@link DownloadButtons}.
 * @param result - The completed job result containing the summary text.
 */
export function SummaryTab({ jobId, result }: SummaryTabProps) {
  if (!result.summary) {
    return <p className="text-sm text-muted-foreground">No summary produced for this job.</p>
  }

  return (
    <div className="space-y-4">
      <div className="prose prose-invert max-w-none text-sm">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{result.summary}</ReactMarkdown>
      </div>
      <DownloadButtons
        jobId={jobId}
        items={[{ name: 'summary.txt', label: 'TXT', fileName: 'summary.txt' }]}
      />
    </div>
  )
}
