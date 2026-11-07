import { useState } from 'react'
import { useJobResult } from '../../hooks/useJobResult'
import { Banner, EmptyState, PlayButton, Spinner, WarningIcon } from '@infra/ui'
import { useT } from '../../i18n/LanguageContext'
import { describeError } from '../../api/errorMessage'
import { resultSkipMessageKey } from '../../lib/outcomeMessages'
import { DownloadButtons } from './DownloadButtons'
import { ResultDownloadMenu } from './ResultDownloadMenu'
import { TranscriptTab } from './TranscriptTab'
import { VisualContextTab } from './VisualContextTab'
import { FacesTab } from './FacesTab'
import { SummaryTab } from './SummaryTab'
import { WordCountsTab } from './WordCountsTab'
import { WordCloudTab } from './WordCloudTab'
import { EntitiesTab } from './EntitiesTab'
import { HateSpeechTab } from './HateSpeechTab'
import { cn } from '../../lib/cn'
import { useMediaPlayerStore } from '../../lib/mediaPlayerStore'
import { downloadsFor, type ResultTabId } from '../../lib/resultDownloads'
import { useAppConfig } from '../../hooks/useAppConfig'

interface ResultPanelProps {
  jobId: string
  /** The original upload filename (e.g. `clip.wav`), used to derive the download stem. */
  fileName: string
}

interface TabSpec {
  id: ResultTabId
  label: string
}

/**
 * Fetches and renders the result of a completed job in a tabbed panel.
 *
 * Calls {@link useJobResult} (always enabled — the parent is responsible for
 * only mounting ResultPanel on a terminal job). Handles loading, error, null
 * result, and skipped-job states. Renders only the tabs whose data is present
 * in the result. The header carries the active tab's download menu, the player
 * and "Download all (.zip)".
 *
 * @param jobId - The job whose result to fetch and display.
 * @param fileName - The original upload filename; used to derive the stem for
 *   prefixing download filenames (e.g. `clip.wav` → stem `clip`).
 */
export function ResultPanel({ jobId, fileName }: ResultPanelProps) {
  const t = useT()
  const query = useJobResult(jobId, true)
  const [activeTab, setActiveTab] = useState<ResultTabId>('transcript')
  const openPlayer = useMediaPlayerStore((s) => s.open)
  // Whether "not grouped" is news: only where the deployment groups faces.
  const groupCapable = useAppConfig().data?.faces === 'group'

  // Derive stem: strip everything from the last '.' onward (or use the full
  // name). Fall back to 'result' so an empty/extension-only name can't yield
  // an empty stem (which would produce filenames like "_archive.zip").
  const dotIdx = fileName.lastIndexOf('.')
  const stem = (dotIdx > 0 ? fileName.slice(0, dotIdx) : fileName) || 'result'

  if (query.isLoading) {
    return <div className="flex items-center gap-2 text-sm text-muted-foreground"><Spinner label={t('results.loading')} /><span aria-hidden>{t('results.loading')}</span></div>
  }

  if (query.isError) {
    const d = describeError(query.error)
    return (
      <Banner variant="danger">
        {t('results.load_failed')} {t(d.key, d.vars)}
      </Banner>
    )
  }

  const result = query.data

  if (result === null || result === undefined) {
    return <EmptyState>{t('results.no_data')}</EmptyState>
  }

  // A skipped job is not an error, but it IS the headline of the run — a muted
  // footnote let it read as an ordinary empty result. `skip_reason` is backend
  // English prose; the localized code is what gets rendered. The banner sits
  // above the tabs rather than replacing them: "skipped" means no transcript,
  // and a silent video can still have keyframes and a summary of them.
  const skippedBanner = result.skipped ? (
    <Banner variant="danger">
      <span className="flex items-start gap-2">
        {/* Repeat the size classes: our className replaces the icon default. */}
        <WarningIcon className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
        <span>
          <strong className="block font-medium">{t('results.skipped_title')}</strong>
          {t(resultSkipMessageKey(result.skip_reason_code))}
        </span>
      </span>
    </Banner>
  ) : null

  // Build the list of available tabs based on what the result contains.
  const availableTabs: TabSpec[] = []

  if (result.transcript.length > 0) {
    availableTabs.push({ id: 'transcript', label: t('results.tab_transcript') })
  }
  // Directly after the transcript: both are accounts of the source material
  // itself, ahead of the analyses derived from it. The frames alone earn the
  // tab — the operator may have captioning switched off.
  if ((result.frame_captions && result.frame_captions.length > 0) || result.keyframes_url) {
    availableTabs.push({ id: 'visual_context', label: t('results.tab_visual_context') })
  }
  // Who is on screen when: also an account of the source material. Present
  // whenever the stage ran on a video, so an empty or unavailable result
  // still says so instead of the tab silently missing.
  if (result.faces) {
    availableTabs.push({ id: 'faces', label: t('results.tab_faces') })
  }
  if (result.summary) {
    availableTabs.push({ id: 'summary', label: t('results.tab_summary') })
  }
  if (result.word_counts && result.word_counts.length > 0) {
    availableTabs.push({ id: 'words', label: t('results.tab_words') })
    availableTabs.push({ id: 'wordcloud', label: t('results.tab_wordcloud') })
  }
  if (result.named_entities && result.named_entities.length > 0) {
    availableTabs.push({ id: 'entities', label: t('results.tab_entities') })
  }
  if (result.hate_speech_findings && result.hate_speech_findings.length > 0) {
    availableTabs.push({ id: 'hate_speech', label: t('results.tab_hate_speech') })
  }

  // A skipped job that produced nothing else: the banner is the whole result.
  // (A completed job with an empty transcript keeps its own empty state below.)
  if (availableTabs.length === 0 && skippedBanner) {
    return skippedBanner
  }

  // If the current active tab is no longer available, fall back to the first one.
  const resolvedTab =
    availableTabs.find((t) => t.id === activeTab)?.id ?? availableTabs[0]?.id ?? 'transcript'

  return (
    <div className="space-y-4">
      {skippedBanner}
      {/* Header row: tabs on the left; the active tab's downloads, play and the
          archive on the right, which stay put when the tab changes. */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <nav className="flex flex-wrap gap-1" aria-label={t('results.tab_nav_label')}>
          {availableTabs.map((tab) => (
            <button
              key={tab.id}
              type="button"
              aria-pressed={resolvedTab === tab.id}
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
        <div className="ml-auto flex flex-wrap items-center justify-end gap-2">
          {/* Keyed by tab so a busy or failed download does not carry over. */}
          <ResultDownloadMenu
            key={resolvedTab}
            jobId={jobId}
            tabLabel={availableTabs.find((tab) => tab.id === resolvedTab)?.label ?? ''}
            items={downloadsFor(resolvedTab, result, stem, t)}
          />
          {/* Opens the recording without jumping anywhere; the timestamps in
              the tabs below are the seek controls. Hidden when the upload is
              gone, since the player would only 404. */}
          {result.media_url && (
            <PlayButton
              label={t('player.open')}
              onClick={() =>
                openPlayer({ jobId, fileName, mediaUrl: result.media_url as string })
              }
            />
          )}
          <DownloadButtons
            jobId={jobId}
            items={[
              {
                name: 'archive.zip',
                title: t('artifacts.download_archive'),
                fileName: `${stem}_archive.zip`,
              },
            ]}
          />
        </div>
      </div>

      {/* Active tab content */}
      <div>
        {resolvedTab === 'transcript' &&
          (result.transcript.length > 0 ? (
            <TranscriptTab
              jobId={jobId}
              segments={result.transcript}
              fileName={fileName}
              mediaUrl={result.media_url}
            />
          ) : (
            // No rows: render the same kind of empty state as the sibling tabs
            // rather than a header-only table.
            <p className="text-sm text-muted-foreground">{t('results.no_transcript')}</p>
          ))}
        {resolvedTab === 'visual_context' && (
          <VisualContextTab
            jobId={jobId}
            result={result}
            fileName={fileName}
            mediaUrl={result.media_url}
          />
        )}
        {resolvedTab === 'faces' && result.faces && (
          <FacesTab
            jobId={jobId}
            faces={result.faces}
            stem={stem}
            fileName={fileName}
            mediaUrl={result.media_url}
            groupCapable={groupCapable}
          />
        )}
        {resolvedTab === 'summary' && (
          <SummaryTab result={result} />
        )}
        {resolvedTab === 'words' && (
          <WordCountsTab result={result} />
        )}
        {resolvedTab === 'wordcloud' && (
          <WordCloudTab jobId={jobId} />
        )}
        {resolvedTab === 'entities' && (
          <EntitiesTab result={result} />
        )}
        {resolvedTab === 'hate_speech' && (
          <HateSpeechTab result={result} />
        )}
      </div>
    </div>
  )
}
