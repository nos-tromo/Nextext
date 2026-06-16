import { DownloadButtons } from './DownloadButtons'
import type { JobResult } from '../../api/types'

interface WordCountsTabProps {
  jobId: string
  result: JobResult
  stem: string
}

/**
 * Displays a ranked word-frequency table for the job, plus CSV/XLSX
 * download buttons. Renders a placeholder when no word counts are present.
 *
 * @param jobId - The job identifier, forwarded to {@link DownloadButtons}.
 * @param result - The completed job result containing the word count list.
 * @param stem - Upload filename without extension; used to prefix download names.
 */
export function WordCountsTab({ jobId, result, stem }: WordCountsTabProps) {
  if (!result.word_counts || result.word_counts.length === 0) {
    return <p className="text-sm text-muted-foreground">No word counts available for this job.</p>
  }

  return (
    <div className="space-y-4">
      <div className="overflow-x-auto rounded-md border border-border">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border bg-muted text-muted-foreground">
              <th className="px-4 py-2 text-left font-medium">Word</th>
              <th className="px-4 py-2 text-right font-medium">Count</th>
            </tr>
          </thead>
          <tbody>
            {result.word_counts.map((wc, i) => (
              <tr key={i} className="border-b border-border last:border-0 hover:bg-accent/40">
                <td className="px-4 py-2 font-mono text-foreground">{wc.word}</td>
                <td className="px-4 py-2 text-right tabular-nums text-foreground">{wc.count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <DownloadButtons
        jobId={jobId}
        items={[
          { name: 'wordcounts.csv', label: 'CSV', fileName: `${stem}_wordcounts.csv` },
          { name: 'wordcounts.xlsx', label: 'XLSX', fileName: `${stem}_wordcounts.xlsx` },
        ]}
      />
    </div>
  )
}
