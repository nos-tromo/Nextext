import { useState } from 'react'
import { useJobResult } from '../../hooks/useJobResult'
import { Spinner } from '../common/Spinner'
import { ErrorBanner } from '../common/ErrorBanner'
import { DownloadButtons } from './DownloadButtons'
import { TranscriptTab } from './TranscriptTab'
import { SummaryTab } from './SummaryTab'
import { WordCountsTab } from './WordCountsTab'
import { WordCloudTab } from './WordCloudTab'
import { EntitiesTab } from './EntitiesTab'
import { HateSpeechTab } from './HateSpeechTab'
import { cn } from '../../lib/cn'

interface ResultPanelProps {
  jobId: string
  /** The original upload filename (e.g. `clip.wav`), used to derive the download stem. */
  fileName: string
}

type TabId = 'transcript' | 'summary' | 'words' | 'wordcloud' | 'entities' | 'hate_speech'

interface TabSpec {
  id: TabId
  label: string
}

/**
 * Fetches and renders the result of a completed job in a tabbed panel.
 *
 * Calls {@link useJobResult} (always enabled — the parent is responsible for
 * only mounting ResultPanel on a terminal job). Handles loading, error, null
 * result, and skipped-job states. Renders only the tabs whose data is present
 * in the result, and provides a "Download all (.zip)" button in the header.
 *
 * @param jobId - The job whose result to fetch and display.
 * @param fileName - The original upload filename; used to derive the stem for
 *   prefixing download filenames (e.g. `clip.wav` → stem `clip`).
 */
export function ResultPanel({ jobId, fileName }: ResultPanelProps) {
  const query = useJobResult(jobId, true)
  const [activeTab, setActiveTab] = useState<TabId>('transcript')

  // Derive stem: strip everything from the last '.' onward (or use the full name).
  const dotIdx = fileName.lastIndexOf('.')
  const stem = dotIdx > 0 ? fileName.slice(0, dotIdx) : fileName

  if (query.isLoading) {
    return <Spinner label="Loading results…" />
  }

  if (query.isError) {
    const msg = query.error instanceof Error ? query.error.message : String(query.error)
    return <ErrorBanner message={`Failed to load results: ${msg}`} />
  }

  const result = query.data

  if (result === null || result === undefined) {
    return <p className="text-sm text-muted-foreground">No result data available.</p>
  }

  if (result.skipped) {
    return (
      <p className="text-sm text-muted-foreground">
        Job was skipped{result.skip_reason ? `: ${result.skip_reason}` : '.'}
      </p>
    )
  }

  // Build the list of available tabs based on what the result contains.
  const availableTabs: TabSpec[] = []

  if (result.transcript.length > 0) {
    availableTabs.push({ id: 'transcript', label: 'Transcript' })
  }
  if (result.summary) {
    availableTabs.push({ id: 'summary', label: 'Summary' })
  }
  if (result.word_counts && result.word_counts.length > 0) {
    availableTabs.push({ id: 'words', label: 'Words' })
    availableTabs.push({ id: 'wordcloud', label: 'Word Cloud' })
  }
  if (result.named_entities && result.named_entities.length > 0) {
    availableTabs.push({ id: 'entities', label: 'Entities' })
  }
  if (result.hate_speech_findings && result.hate_speech_findings.length > 0) {
    availableTabs.push({ id: 'hate_speech', label: 'Hate Speech' })
  }

  // If the current active tab is no longer available, fall back to the first one.
  const resolvedTab =
    availableTabs.find((t) => t.id === activeTab)?.id ?? availableTabs[0]?.id ?? 'transcript'

  return (
    <div className="space-y-4">
      {/* Header row: tab bar + archive download */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <nav className="flex flex-wrap gap-1" aria-label="Result tabs">
          {availableTabs.map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTab(tab.id)}
              className={cn(
                'rounded px-3 py-1 text-sm transition-colors',
                resolvedTab === tab.id
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:text-foreground',
              )}
            >
              {tab.label}
            </button>
          ))}
        </nav>
        <DownloadButtons
          jobId={jobId}
          items={[
            {
              name: 'archive.zip',
              label: 'Download all (.zip)',
              fileName: `${stem}_archive.zip`,
            },
          ]}
        />
      </div>

      {/* Active tab content */}
      <div>
        {resolvedTab === 'transcript' && (
          <TranscriptTab jobId={jobId} segments={result.transcript} stem={stem} />
        )}
        {resolvedTab === 'summary' && (
          <SummaryTab jobId={jobId} result={result} stem={stem} />
        )}
        {resolvedTab === 'words' && (
          <WordCountsTab jobId={jobId} result={result} stem={stem} />
        )}
        {resolvedTab === 'wordcloud' && (
          <WordCloudTab jobId={jobId} stem={stem} />
        )}
        {resolvedTab === 'entities' && (
          <EntitiesTab jobId={jobId} result={result} stem={stem} />
        )}
        {resolvedTab === 'hate_speech' && (
          <HateSpeechTab jobId={jobId} result={result} stem={stem} />
        )}
      </div>
    </div>
  )
}
