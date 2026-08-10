import { useEffect, useRef, useState } from 'react'
import { ChevronDownIcon, DownloadButton } from '@infra/ui'
import { downloadBatchArtifact } from '../../lib/download'
import { useT } from '../../i18n/LanguageContext'
import { describeError } from '../../api/errorMessage'
import type { ErrorDescriptor } from '../../api/errorMessage'

interface BatchDownloadMenuProps {
  /** Number of completed jobs available to include in the batch. */
  completedCount: number
}

interface BatchItem {
  /** Backend batch artifact name. */
  name: string
  /** Menu item label catalog key. */
  labelKey: 'jobs.combined_jsonl' | 'jobs.full_batch_zip'
  /** Suggested download file name shown to the browser. */
  fileName: string
}

const ITEMS: BatchItem[] = [
  { name: 'docint.jsonl', labelKey: 'jobs.combined_jsonl', fileName: 'nextext_docint.jsonl' },
  { name: 'archive.zip', labelKey: 'jobs.full_batch_zip', fileName: 'nextext_batch.zip' },
]

/**
 * A "Download all jobs" dropdown that bundles every completed job into a
 * single download — either a combined docint JSONL or a ZIP of all outputs.
 *
 * Disabled until at least one job completes. A busy state prevents a second
 * concurrent download; failures surface inline next to the trigger. The menu
 * closes on outside click or Escape.
 *
 * @param completedCount - Number of completed jobs; enables the control.
 */
export function BatchDownloadMenu({ completedCount }: BatchDownloadMenuProps) {
  const t = useT()
  const [open, setOpen] = useState(false)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<ErrorDescriptor | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  const disabled = completedCount === 0 || busy !== null

  useEffect(() => {
    if (!open) return
    function onPointerDown(event: PointerEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false)
      }
    }
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') setOpen(false)
    }
    document.addEventListener('pointerdown', onPointerDown)
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('pointerdown', onPointerDown)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [open])

  async function handleSelect(item: BatchItem) {
    if (busy) return
    setError(null)
    setBusy(item.name)
    setOpen(false)
    try {
      await downloadBatchArtifact(item.name, item.fileName)
    } catch (err) {
      setError(describeError(err))
    } finally {
      setBusy(null)
    }
  }

  return (
    <div ref={containerRef} className="relative flex items-center gap-2">
      {error && <span className="text-sm text-danger">{t(error.key, error.vars)}</span>}
      {/* Only the trigger becomes an icon; the menu items below name the
          artifact they produce and keep their words. The caret is what says
          this opens a list rather than downloading on the spot. */}
      <DownloadButton
        label={t('jobs.download_all')}
        hint={completedCount === 0 ? t('jobs.no_completed_yet') : undefined}
        disabled={disabled}
        busy={busy !== null}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        className="gap-1 px-2"
      >
        <ChevronDownIcon className="h-3.5 w-3.5" />
      </DownloadButton>
      {open && (
        <div
          role="menu"
          className="absolute right-0 top-full z-10 mt-1 min-w-max rounded-md border border-border bg-muted py-1 shadow-lg"
        >
          {ITEMS.map((item) => (
            <button
              key={item.name}
              type="button"
              role="menuitem"
              onClick={() => void handleSelect(item)}
              className="block w-full px-3 py-1.5 text-left text-sm text-foreground hover:bg-accent hover:text-primary"
            >
              {t(item.labelKey)}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
