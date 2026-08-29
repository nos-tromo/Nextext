import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { DownloadButtons } from './DownloadButtons'
import { useT } from '../../i18n/LanguageContext'
import type { FrameCaption, JobResult } from '../../api/types'

interface SummaryTabProps {
  jobId: string
  result: JobResult
  stem: string
}

/**
 * Formats a caption timestamp as `mm:ss`, with minutes growing past 60.
 *
 * Mirrors the backend's `_timestamp` so the on-screen stamps match the
 * downloaded `visual_context.txt` exactly.
 *
 * @param seconds - Offset from the start of the clip.
 * @returns Zero-padded `mm:ss` stamp.
 */
function formatStamp(seconds: number): string {
  const total = Number.isFinite(seconds) && seconds > 0 ? Math.floor(seconds) : 0
  const minutes = Math.floor(total / 60)
  return `${String(minutes).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`
}

/**
 * Lists the video frames that fed the summary, collapsed by default.
 *
 * The summary itself is the answer; these are its visual sources, so they sit
 * behind a disclosure rather than competing with it.
 *
 * @param captions - Frame captions in time order.
 */
function VisualContext({ captions }: { captions: FrameCaption[] }) {
  const t = useT()
  return (
    <details className="rounded-md border border-border">
      <summary className="cursor-pointer px-3 py-2 text-sm font-medium">
        {t('results.visual_context')}
      </summary>
      <div className="space-y-2 px-3 pb-3">
        <p className="text-xs text-muted-foreground">{t('results.visual_context_hint')}</p>
        <ul className="space-y-1">
          {captions.map((caption) => (
            <li key={`${caption.time_sec}-${caption.caption}`} className="flex gap-2 text-sm">
              <span className="shrink-0 font-mono text-xs text-muted-foreground">
                {formatStamp(caption.time_sec)}
              </span>
              <span>{caption.caption}</span>
            </li>
          ))}
        </ul>
      </div>
    </details>
  )
}

/**
 * Displays the job summary rendered as Markdown, plus a download button
 * for `summary.txt`. Renders nothing when the job produced no summary.
 *
 * For video jobs the summary also draws on descriptions of sampled frames;
 * those are listed below it and downloadable as `visual_context.txt`, so a
 * reader can check what the model was shown as well as what it was told.
 *
 * @param jobId - The job identifier, forwarded to {@link DownloadButtons}.
 * @param result - The completed job result containing the summary text.
 * @param stem - Upload filename without extension; used to prefix download names.
 */
export function SummaryTab({ jobId, result, stem }: SummaryTabProps) {
  const t = useT()
  if (!result.summary) {
    return <p className="text-sm text-muted-foreground">{t('results.no_summary')}</p>
  }

  const captions = result.frame_captions ?? []

  return (
    <div className="space-y-4">
      <div className="prose prose-invert max-w-none text-sm">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{result.summary}</ReactMarkdown>
      </div>
      {captions.length > 0 && <VisualContext captions={captions} />}
      <DownloadButtons
        jobId={jobId}
        items={[
          { name: 'summary.txt', label: 'TXT', fileName: `${stem}_summary.txt` },
          ...(captions.length > 0
            ? [
                {
                  name: 'visual_context.txt',
                  label: 'TXT',
                  title: t('results.visual_context'),
                  fileName: `${stem}_visual_context.txt`,
                },
              ]
            : []),
        ]}
      />
    </div>
  )
}
