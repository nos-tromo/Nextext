import { DownloadButtons } from './DownloadButtons'
import { useT } from '../../i18n/LanguageContext'
import { useMediaPlayerStore } from '../../lib/mediaPlayerStore'
import { useFollowActiveRow } from '../../hooks/useFollowActiveRow'
import { TimeCell } from './TimeCell'
import { cn } from '../../lib/cn'
import type { JobResult } from '../../api/types'

interface VisualContextTabProps {
  jobId: string
  result: JobResult
  stem: string
  fileName: string
  mediaUrl: string | null
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
 * Displays what the video showed, moment by moment: one model-written
 * description per sampled keyframe, stamped with its time in the clip, plus a
 * `visual_context.txt` download.
 *
 * These descriptions are also what the summary drew on for a video job, so the
 * tab sits next to the transcript — both are accounts of the source material
 * itself rather than analyses derived from it.
 *
 * @param jobId - The job identifier, forwarded to {@link DownloadButtons}.
 * @param result - The completed job result containing the frame captions.
 * @param stem - Upload filename without extension; used to prefix download names.
 * @param fileName - Original upload name, shown as the player's title.
 * @param mediaUrl - Capability URL for playback; `null` disables the
 *   timestamp controls.
 */
export function VisualContextTab({ jobId, result, stem, fileName, mediaUrl }: VisualContextTabProps) {
  const t = useT()
  const playingHere = useMediaPlayerStore((s) => s.session?.jobId === jobId)
  const currentTime = useMediaPlayerStore((s) => s.currentTime)
  const captions = result.frame_captions ?? []
  // Frames carry no end time, so the active one is the latest already
  // reached by the playhead.
  let activeIndex = -1
  if (playingHere) {
    captions.forEach((caption, i) => {
      if (caption.time_sec <= currentTime) activeIndex = i
    })
  }
  const activeRowRef = useFollowActiveRow(activeIndex)

  if (captions.length === 0) {
    return <p className="text-sm text-muted-foreground">{t('results.no_visual_context')}</p>
  }

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">{t('results.visual_context_hint')}</p>
      <div className="overflow-x-auto rounded-md border border-border">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border bg-muted text-muted-foreground">
              <th className="px-4 py-2 text-left font-medium">{t('results.col_time')}</th>
              <th className="px-4 py-2 text-left font-medium">{t('results.col_description')}</th>
            </tr>
          </thead>
          <tbody>
            {captions.map((caption, i) => (
              <tr
                key={i}
                ref={i === activeIndex ? activeRowRef : undefined}
                aria-current={i === activeIndex ? 'true' : undefined}
                className={cn(
                  'border-b border-border last:border-0 hover:bg-accent/40',
                  i === activeIndex && 'bg-accent/60',
                )}
              >
                <TimeCell
                  label={formatStamp(caption.time_sec)}
                  seconds={caption.time_sec}
                  jobId={jobId}
                  fileName={fileName}
                  mediaUrl={mediaUrl}
                />
                <td className="px-4 py-2 text-foreground">{caption.caption}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <DownloadButtons
        jobId={jobId}
        items={[
          { name: 'visual_context.txt', label: 'TXT', fileName: `${stem}_visual_context.txt` },
        ]}
      />
    </div>
  )
}
