import { apiFetch } from '../api/client'

// Every download is fetched and handed to the browser as a blob, never linked
// to directly: an `<a href>` to the backend could carry neither the identity
// header nor the stated purpose (`X-Audit-Purpose`) a disclosure needs, so the
// bytes come through the API client like any other request.

/**
 * Trigger a browser file download from a blob object URL.
 *
 * Creates a hidden anchor, clicks it, then revokes the blob URL and removes
 * the anchor from the DOM.
 *
 * @param url - A `blob:` URL pointing to the bytes to download.
 * @param fileName - The suggested download file name shown to the user.
 */
function triggerDownload(url: string, fileName: string): void {
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

/**
 * Fetch a job artifact from the backend.
 *
 * Goes through the API client, so it carries the identity and the stated
 * purpose; throws `ApiError` on non-2xx.
 *
 * @param jobId - The job identifier.
 * @param name - The artifact name (e.g. `transcript.csv`).
 * @returns The raw `Response` so callers can consume the body as needed.
 */
export async function fetchArtifact(jobId: string, name: string): Promise<Response> {
  return apiFetch(`/jobs/${jobId}/artifacts/${name}`)
}

/**
 * Fetch a batch artifact spanning all of the caller's completed jobs.
 *
 * @param name - The batch artifact name (`docint.jsonl` or `archive.zip`).
 * @returns The raw `Response` so callers can consume the body as needed.
 */
export async function fetchBatchArtifact(name: string): Promise<Response> {
  return apiFetch(`/jobs/batch/${name}`)
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
 * @param jobId - The job identifier.
 * @param name - The artifact name on the backend.
 * @param fileName - The suggested download file name shown to the user.
 */
export async function downloadArtifact(jobId: string, name: string, fileName: string): Promise<void> {
  const url = await fetchArtifactObjectUrl(jobId, name)
  triggerDownload(url, fileName)
}

/**
 * Trigger a browser file download for a batch artifact (combined across jobs).
 *
 * @param name - The batch artifact name (`docint.jsonl` or `archive.zip`).
 * @param fileName - The suggested download file name shown to the user.
 */
export async function downloadBatchArtifact(name: string, fileName: string): Promise<void> {
  const res = await fetchBatchArtifact(name)
  const blob = await res.blob()
  const url = URL.createObjectURL(blob)
  triggerDownload(url, fileName)
}
