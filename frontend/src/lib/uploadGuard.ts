/** Mirrors the backend NEXTEXT_MAX_UPLOAD_MB default (per-file hard cap). */
export const DEFAULT_MAX_FILE_MB = 8192

/**
 * Return an actionable message if any file exceeds the per-file cap, else null.
 * Advisory only — the backend enforces the real limit and streams to disk.
 */
export function checkUploadAcceptable(files: File[], maxFileMb: number = DEFAULT_MAX_FILE_MB): string | null {
  const maxBytes = maxFileMb * (1 << 20)
  const over = files.find((f) => f.size > maxBytes)
  if (!over) return null
  const gib = 1 << 30
  return (
    `"${over.name}" is ${(over.size / gib).toFixed(1)} GB, over the ` +
    `${(maxFileMb / 1024).toFixed(1)} GB per-file limit. Split it, or use ` +
    `\`nextext-cli\` for very large local files.`
  )
}
