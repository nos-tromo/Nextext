import { useEffect, useState } from 'react'
import { fetchArtifactObjectUrl } from '../lib/download'

interface ArtifactImageState {
  url: string | null
  loading: boolean
  error: string | null
}

/**
 * Fetch a job artifact as a blob object URL for inline image display.
 *
 * Automatically revokes the previous object URL when the component unmounts
 * or when `jobId`/`artifactName` change, preventing memory leaks.
 *
 * @param jobId - The job identifier.
 * @param artifactName - The artifact file name on the backend (e.g. `wordcloud.png`).
 * @param enabled - Whether to fetch; pass false to defer until ready.
 * @returns State containing the blob URL, loading flag, and error string.
 */
export function useArtifactImage(
  jobId: string,
  artifactName: string,
  enabled: boolean,
): ArtifactImageState {
  const [state, setState] = useState<ArtifactImageState>({ url: null, loading: false, error: null })

  useEffect(() => {
    if (!enabled) return

    let revoked = false
    let objectUrl: string | null = null

    async function load(): Promise<void> {
      setState({ url: null, loading: true, error: null })
      try {
        const url = await fetchArtifactObjectUrl(jobId, artifactName)
        objectUrl = url
        if (!revoked) {
          setState({ url, loading: false, error: null })
        } else {
          URL.revokeObjectURL(url)
        }
      } catch (err: unknown) {
        if (!revoked) {
          setState({ url: null, loading: false, error: err instanceof Error ? err.message : String(err) })
        }
      }
    }

    void load()

    return () => {
      revoked = true
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl)
        objectUrl = null
      }
    }
  }, [jobId, artifactName, enabled])

  return state
}
