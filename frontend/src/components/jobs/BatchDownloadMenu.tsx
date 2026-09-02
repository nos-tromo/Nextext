import { useState } from 'react'
import { ChevronDownIcon, DownloadButton, Menu, MenuItem } from '@infra/ui'
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
 * concurrent download; failures surface inline next to the trigger. Dismissal
 * and the keyboard belong to `Menu`.
 *
 * @param completedCount - Number of completed jobs; enables the control.
 */
export function BatchDownloadMenu({ completedCount }: BatchDownloadMenuProps) {
  const t = useT()
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<ErrorDescriptor | null>(null)

  const disabled = completedCount === 0 || busy !== null

  async function handleSelect(item: BatchItem) {
    if (busy) return
    setError(null)
    setBusy(item.name)
    try {
      await downloadBatchArtifact(item.name, item.fileName)
    } catch (err) {
      setError(describeError(err))
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="flex items-center gap-2">
      {error && <span className="text-sm text-danger">{t(error.key, error.vars)}</span>}
      {/* Only the trigger becomes an icon; the menu items below name the
          artifact they produce and keep their words. The caret is what says
          this opens a list rather than downloading on the spot. */}
      <Menu
        align="end"
        trigger={(props) => (
          <DownloadButton
            {...props}
            label={t('jobs.download_all')}
            hint={completedCount === 0 ? t('jobs.no_completed_yet') : undefined}
            disabled={disabled}
            busy={busy !== null}
            className="gap-1 px-2"
          >
            <ChevronDownIcon className="h-3.5 w-3.5" />
          </DownloadButton>
        )}
      >
        {ITEMS.map((item) => (
          <MenuItem key={item.name} onSelect={() => void handleSelect(item)}>
            {t(item.labelKey)}
          </MenuItem>
        ))}
      </Menu>
    </div>
  )
}
