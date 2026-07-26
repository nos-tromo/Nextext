/** Mirrors the backend NEXTEXT_MAX_UPLOAD_MB default (per-file hard cap). */
export const DEFAULT_MAX_FILE_MB = 8192

/** Interpolation vars for the oversized-file message; see {@link checkUploadAcceptable}. */
export interface OversizeFileVars {
  name: string
  sizeGb: string
  limitGb: string
}

/** Default (English, untranslated) rendering — used when no formatter is supplied. */
function defaultFormat({ name, sizeGb, limitGb }: OversizeFileVars): string {
  return (
    `"${name}" is ${sizeGb} GB, over the ${limitGb} GB per-file limit. ` +
    'Split it, or use `nextext-cli` for very large local files.'
  )
}

/**
 * Return an actionable message if any file exceeds the per-file cap, else null.
 * Advisory only — the backend enforces the real limit and streams to disk.
 *
 * @param formatMessage - Renders the final message from the oversize vars.
 *   This module has no i18n context of its own, so callers that need a
 *   localized message pass a formatter built from their own `t()`
 *   (e.g. `(vars) => t('upload.file_too_large', vars)`); the default renders
 *   the plain English template.
 */
export function checkUploadAcceptable(
  files: File[],
  maxFileMb: number = DEFAULT_MAX_FILE_MB,
  formatMessage: (vars: OversizeFileVars) => string = defaultFormat,
): string | null {
  const maxBytes = maxFileMb * (1 << 20)
  const over = files.find((f) => f.size > maxBytes)
  if (!over) return null
  const gib = 1 << 30
  return formatMessage({
    name: over.name,
    sizeGb: (over.size / gib).toFixed(1),
    limitGb: (maxFileMb / 1024).toFixed(1),
  })
}
