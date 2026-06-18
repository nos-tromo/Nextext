/**
 * Per-browser persistence for the chosen translation target language.
 *
 * The selection survives a page reload via localStorage — the same
 * reload-resilience model the rest of the stateless app uses (see the owner
 * id). A fresh browser (no stored preference) falls back to the backend's
 * configured default (`NEXTEXT_DEFAULT_TARGET_LANG`, English by default).
 */
export const TARGET_LANG_STORAGE_KEY = 'nextext-trg-lang'

/**
 * Read the persisted target language code, or `null` when none is stored or
 * localStorage is unavailable (e.g. private mode).
 */
export function readStoredTargetLang(): string | null {
  try {
    const value = localStorage.getItem(TARGET_LANG_STORAGE_KEY)
    return value && value.trim() ? value : null
  } catch {
    return null
  }
}

/** Persist the chosen target language code, ignoring storage errors. */
export function writeStoredTargetLang(code: string): void {
  try {
    localStorage.setItem(TARGET_LANG_STORAGE_KEY, code)
  } catch {
    // Storage may be unavailable (private mode / quota); the selection simply
    // won't survive a reload in that case.
  }
}
