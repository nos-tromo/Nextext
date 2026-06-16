import { API_BASE, ApiError, OWNER_HEADER } from '../api/client'
import { resolveOwnerId } from '../identity/owner'

/** Artifact URL for a given job and artifact name. */
function artifactUrl(jobId: string, name: string): string {
  return `${API_BASE}/jobs/${jobId}/artifacts/${name}`
}

/**
 * Fetch a job artifact from the backend.
 *
 * Sends the owner identity header and throws {@link ApiError} on non-2xx.
 *
 * @param jobId - The job identifier.
 * @param name - The artifact name (e.g. `transcript.csv`).
 * @returns The raw `Response` so callers can consume the body as needed.
 */
export async function fetchArtifact(jobId: string, name: string): Promise<Response> {
  const res = await fetch(artifactUrl(jobId, name), {
    headers: { [OWNER_HEADER]: resolveOwnerId() },
  })
  if (!res.ok) {
    let detail: unknown
    const text = await res.text()
    try {
      const parsed = JSON.parse(text) as unknown
      detail =
        parsed && typeof parsed === 'object' && 'detail' in parsed
          ? (parsed as { detail: unknown }).detail
          : parsed
    } catch {
      detail = text
    }
    throw new ApiError(res.status, detail)
  }
  return res
}

/**
 * Fetch a job artifact and return a blob object URL for inline use.
 *
 * The caller is responsible for revoking the URL via `URL.revokeObjectURL`
 * when it is no longer needed.
 *
 * @param jobId - The job identifier.
 * @param name - The artifact name.
 * @returns A `blob:` URL pointing to the artifact bytes.
 */
export async function fetchArtifactObjectUrl(jobId: string, name: string): Promise<string> {
  const res = await fetchArtifact(jobId, name)
  const blob = await res.blob()
  return URL.createObjectURL(blob)
}

/**
 * Trigger a browser file download for a job artifact.
 *
 * Creates a hidden anchor, clicks it, then immediately revokes the blob URL
 * and removes the anchor from the DOM.
 *
 * @param jobId - The job identifier.
 * @param name - The artifact name on the backend.
 * @param fileName - The suggested download file name shown to the user.
 */
export async function downloadArtifact(jobId: string, name: string, fileName: string): Promise<void> {
  const url = await fetchArtifactObjectUrl(jobId, name)
  try {
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = fileName
    anchor.style.display = 'none'
    document.body.appendChild(anchor)
    anchor.click()
    document.body.removeChild(anchor)
  } finally {
    URL.revokeObjectURL(url)
  }
}
