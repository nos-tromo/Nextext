import { DownloadButtons } from './DownloadButtons'
import { useT } from '../../i18n/LanguageContext'
import type { JobResult } from '../../api/types'

interface WordCountsTabProps {
  jobId: string
  result: JobResult
  stem: string
}

/**
 * Displays a ranked word-frequency histogram for the job — one row per word
 * with a bar scaled against the most frequent word, the exact count beside it,
 * and CSV/XLSX download buttons. Renders a placeholder when no word counts
 * are present.
 *
 * @param jobId - The job identifier, forwarded to {@link DownloadButtons}.
 * @param result - The completed job result containing the word count list.
 * @param stem - Upload filename without extension; used to prefix download names.
 */
export function WordCountsTab({ jobId, result, stem }: WordCountsTabProps) {
  const t = useT()
  if (!result.word_counts || result.word_counts.length === 0) {
    return <p className="text-sm text-muted-foreground">{t('results.no_word_counts')}</p>
  }

  // Counts are >= 1, so the non-empty guard above makes this a safe divisor.
  const max = Math.max(...result.word_counts.map((wc) => wc.count))

  return (
    <div className="space-y-4">
      <div className="overflow-x-auto rounded-md border border-border">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border bg-muted text-muted-foreground">
              <th className="px-4 py-2 text-left font-medium">{t('results.col_word')}</th>
              <th aria-hidden="true" className="w-full" />
              <th className="px-4 py-2 text-right font-medium">{t('results.col_count')}</th>
            </tr>
          </thead>
          <tbody>
            {result.word_counts.map((wc, i) => (
              <tr key={i} className="border-b border-border last:border-0 hover:bg-accent/40">
                <td className="whitespace-nowrap px-4 py-2 font-mono text-foreground">{wc.word}</td>
                <td aria-hidden="true" className="w-full py-2 pr-2">
                  <div className="h-2 w-full rounded-sm bg-muted">
                    <div
                      data-testid="word-count-bar"
                      className="h-2 rounded-r-sm bg-primary"
                      style={{ width: `${(wc.count / max) * 100}%` }}
                    />
                  </div>
                </td>
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
