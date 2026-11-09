import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useT } from '../../i18n/LanguageContext'
import type { JobResult } from '../../api/types'

interface SummaryTabProps {
  result: JobResult
}

/**
 * Displays the job summary rendered as Markdown; its `summary.txt` download
 * sits in the result panel's header. Renders nothing when the job produced no
 * summary.
 *
 * When the job also sampled keyframes, the summary draws on their descriptions
 * as well — or on them alone, for a video whose audio held no speech. The
 * descriptions live in their own tab (see `VisualContextTab`), not here.
 *
 * @param result - The completed job result containing the summary text.
 */
export function SummaryTab({ result }: SummaryTabProps) {
  const t = useT()
  if (!result.summary) {
    return <p className="text-sm text-muted-foreground">{t('results.no_summary')}</p>
  }

  return (
    <div className="space-y-4">
      <div className="prose prose-invert max-w-none text-sm">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{result.summary}</ReactMarkdown>
      </div>
    </div>
  )
}
