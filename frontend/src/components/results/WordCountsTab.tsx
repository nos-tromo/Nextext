import { useT } from '../../i18n/LanguageContext'
import type { JobResult } from '../../api/types'

interface WordCountsTabProps {
  result: JobResult
}

/**
 * Displays a ranked word-frequency histogram for the job — one row per word
 * with a bar scaled against the most frequent word and the exact count beside
 * it. Renders a placeholder when no word counts are present.
 *
 * @param result - The completed job result containing the word count list.
 */
export function WordCountsTab({ result }: WordCountsTabProps) {
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
            {result.word_counts.map((wc) => (
              <tr key={wc.word} className="border-b border-border last:border-0 hover:bg-accent/40">
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
    </div>
  )
}
