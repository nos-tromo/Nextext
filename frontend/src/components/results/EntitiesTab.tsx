import { useT } from '../../i18n/LanguageContext'
import type { JobResult } from '../../api/types'

interface EntitiesTabProps {
  result: JobResult
}

/**
 * Displays named entities extracted from the job, grouped by category in a
 * table with frequency counts.
 * Renders a placeholder when no entities are present.
 *
 * @param result - The completed job result containing the named entity list.
 */
export function EntitiesTab({ result }: EntitiesTabProps) {
  const t = useT()
  if (!result.named_entities || result.named_entities.length === 0) {
    return <p className="text-sm text-muted-foreground">{t('results.no_entities')}</p>
  }

  return (
    <div className="space-y-4">
      <div className="overflow-x-auto rounded-md border border-border">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border bg-muted text-muted-foreground">
              <th className="px-4 py-2 text-left font-medium">{t('results.col_entity')}</th>
              <th className="px-4 py-2 text-left font-medium">{t('results.col_category')}</th>
              <th className="px-4 py-2 text-right font-medium">{t('results.col_frequency')}</th>
            </tr>
          </thead>
          <tbody>
            {result.named_entities.map((ne) => (
              <tr key={`${ne.entity}\u0000${ne.category}`} className="border-b border-border last:border-0 hover:bg-accent/40">
                <td className="px-4 py-2 text-foreground">{ne.entity}</td>
                <td className="px-4 py-2 text-muted-foreground">{ne.category}</td>
                <td className="px-4 py-2 text-right tabular-nums text-foreground">{ne.frequency}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
