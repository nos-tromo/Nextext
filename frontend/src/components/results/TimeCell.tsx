import { useT } from '../../i18n/LanguageContext'
import { useMediaPlayerStore } from '../../lib/mediaPlayerStore'

interface TimeCellProps {
  /** Formatted timestamp shown in the cell, e.g. `0:00:05` or `01:15`. */
  label: string
  /** Offset to seek to; `null` when the timestamp could not be parsed. */
  seconds: number | null
  jobId: string
  /** Original upload name, shown as the player's title. */
  fileName: string
  /** Capability URL for playback; `null` when the upload is gone. */
  mediaUrl: string | null
}

/**
 * A transcript / visual-context timestamp, clickable when there is media to
 * play.
 *
 * Degrades to plain text rather than a dead control whenever playback is
 * impossible — the upload has been deleted, or this row's timestamp could not
 * be parsed into an offset.
 *
 * @param label - The timestamp text to display.
 * @param seconds - Offset to seek to, or `null` to render plain text.
 * @param jobId - Job whose recording this row belongs to.
 * @param fileName - Original upload name for the player title.
 * @param mediaUrl - Capability URL, or `null` to render plain text.
 */
export function TimeCell({ label, seconds, jobId, fileName, mediaUrl }: TimeCellProps) {
  const t = useT()
  const open = useMediaPlayerStore((s) => s.open)

  if (mediaUrl === null || seconds === null) {
    return <td className="whitespace-nowrap px-4 py-2 align-top tabular-nums text-muted-foreground">{label}</td>
  }

  return (
    <td className="whitespace-nowrap px-4 py-2 align-top tabular-nums">
      <button
        type="button"
        aria-label={t('player.play_from', { time: label })}
        onClick={() => open({ jobId, fileName, mediaUrl }, seconds)}
        className="rounded text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
      >
        {label}
      </button>
    </td>
  )
}
