import { DownloadButtons } from './DownloadButtons'
import type { JobResult } from '../../api/types'

interface EntitiesTabProps {
  jobId: string
  result: JobResult
}

/**
 * Displays named entities extracted from the job, grouped by category in a
 * table with frequency counts, plus CSV/XLSX download buttons.
 * Renders a placeholder when no entities are present.
 *
 * @param jobId - The job identifier, forwarded to {@link DownloadButtons}.
 * @param result - The completed job result containing the named entity list.
 */
export function EntitiesTab({ jobId, result }: EntitiesTabProps) {
  if (!result.named_entities || result.named_entities.length === 0) {
    return <p className="text-sm text-muted-foreground">No named entities found for this job.</p>
  }

  return (
    <div className="space-y-4">
      <div className="overflow-x-auto rounded-md border border-border">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border bg-muted text-muted-foreground">
              <th className="px-4 py-2 text-left font-medium">Entity</th>
              <th className="px-4 py-2 text-left font-medium">Category</th>
              <th className="px-4 py-2 text-right font-medium">Frequency</th>
            </tr>
          </thead>
          <tbody>
            {result.named_entities.map((ne, i) => (
              <tr key={i} className="border-b border-border last:border-0 hover:bg-accent/40">
                <td className="px-4 py-2 text-foreground">{ne.entity}</td>
                <td className="px-4 py-2 text-muted-foreground">{ne.category}</td>
                <td className="px-4 py-2 text-right tabular-nums text-foreground">{ne.frequency}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <DownloadButtons
        jobId={jobId}
        items={[
          { name: 'entities.csv', label: 'CSV', fileName: 'entities.csv' },
          { name: 'entities.xlsx', label: 'XLSX', fileName: 'entities.xlsx' },
        ]}
      />
    </div>
  )
}
