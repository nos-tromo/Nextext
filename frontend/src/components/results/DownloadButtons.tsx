import { useState } from 'react'
import { DownloadButton } from '@infra/ui'
import { downloadArtifact } from '../../lib/download'
import { useT } from '../../i18n/LanguageContext'
import { describeError } from '../../api/errorMessage'
import type { ErrorDescriptor } from '../../api/errorMessage'

interface DownloadSpec {
  /** Artifact name on the backend, e.g. `transcript.csv`. */
  name: string
  /**
   * Short format chip beside the icon, e.g. `CSV`. Set it where several
   * downloads sit side by side and the icon alone cannot tell them apart;
   * omit it for a lone download, which then stands as the icon by itself.
   */
  label?: string
  /** Accessible name. Defaults to "Download {label}". */
  title?: string
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
  const t = useT()
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<ErrorDescriptor | null>(null)

  async function handleClick(item: DownloadSpec) {
    if (busy) return
    setError(null)
    setBusy(item.name)
    try {
      await downloadArtifact(jobId, item.name, item.fileName)
    } catch (err) {
      setError(describeError(err))
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="flex flex-wrap items-center gap-2">
      {/* Each keeps its format beside the icon: these sit side by side, and a
          row of identical download icons would be a guessing game. The busy
          one shows a spinner rather than the `…` this used to type out. */}
      {items.map((item) => (
        <DownloadButton
          key={item.name}
          label={item.title ?? t('results.download_artifact', { label: item.label ?? '' })}
          busy={busy === item.name}
          disabled={busy !== null}
          onClick={() => void handleClick(item)}
        >
          {item.label}
        </DownloadButton>
      ))}
      {error && <span className="text-sm text-danger">{t(error.key, error.vars)}</span>}
    </div>
  )
}
