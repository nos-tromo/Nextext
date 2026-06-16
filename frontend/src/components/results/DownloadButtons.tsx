import { useState } from 'react'
import { downloadArtifact } from '../../lib/download'
import { cn } from '../../lib/cn'

interface DownloadSpec {
  /** Artifact name on the backend, e.g. `transcript.csv`. */
  name: string
  /** Label shown on the button, e.g. `CSV`. */
  label: string
  /** Suggested download file name shown to the browser, e.g. `transcript.csv`. */
  fileName: string
}

interface DownloadButtonsProps {
  jobId: string
  items: DownloadSpec[]
}

/**
 * Renders a row of download buttons for a set of job artifacts.
 *
 * Each button triggers a blob-URL download via {@link downloadArtifact}.
 * An individual per-button loading state prevents double-clicks; errors
 * are surfaced inline as a short error message next to the buttons.
 *
 * @param jobId - The job whose artifacts to download.
 * @param items - The list of artifacts to expose as download buttons.
 */
export function DownloadButtons({ jobId, items }: DownloadButtonsProps) {
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function handleClick(item: DownloadSpec) {
    if (busy) return
    setError(null)
    setBusy(item.name)
    try {
      await downloadArtifact(jobId, item.name, item.fileName)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="flex flex-wrap items-center gap-2">
      {items.map((item) => (
        <button
          key={item.name}
          type="button"
          disabled={busy !== null}
          onClick={() => void handleClick(item)}
          className={cn(
            'rounded border border-border px-3 py-1 text-sm transition-colors',
            busy === item.name
              ? 'cursor-not-allowed text-muted-foreground'
              : 'text-foreground hover:border-primary hover:text-primary',
          )}
        >
          {busy === item.name ? '…' : item.label}
        </button>
      ))}
      {error && <span className="text-sm text-danger">{error}</span>}
    </div>
  )
}
