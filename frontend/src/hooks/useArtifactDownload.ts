import { useState } from 'react'
import { downloadArtifact } from '../lib/download'
import { describeError } from '../api/errorMessage'
import type { ErrorDescriptor } from '../api/errorMessage'
import type { DownloadSpec } from '../components/results/DownloadButtons'

/**
 * A job's artifact downloads, with one busy state and the last failure.
 *
 * Shared by every download control on the result screen, so each blocks a
 * second download the same way and leaves the caller to decide where the
 * failure is shown.
 *
 * @param jobId - The job whose artifacts to download.
 * @returns `busy` (the artifact being downloaded, else `null`), the last
 *   `error` (cleared when the next download starts) and `download`.
 */
export function useArtifactDownload(jobId: string) {
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<ErrorDescriptor | null>(null)

  async function download(item: Pick<DownloadSpec, 'name' | 'fileName'>) {
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

  return { busy, error, download }
}
