import { useState } from 'react'
import { ChevronDownIcon, DownloadButton, Menu, MenuItem } from '@infra/ui'
import { downloadArtifact } from '../../lib/download'
import { useT } from '../../i18n/LanguageContext'
import { describeError } from '../../api/errorMessage'
import type { ErrorDescriptor } from '../../api/errorMessage'
import type { DownloadSpec } from './DownloadButtons'

interface ResultDownloadMenuProps {
  jobId: string
  /** The active tab's name, which the trigger says it downloads. */
  tabLabel: string
  /** The tab's downloads, from `downloadsFor`. */
  items: DownloadSpec[]
}

/**
 * The active result tab's downloads, as one menu in the result panel's header.
 *
 * One quiet trigger instead of a row of format buttons, so the header keeps to
 * one line beside the tabs; the formats are the menu's items. Built like
 * `BatchDownloadMenu`: a busy state blocks a second download, and a failure
 * shows inline beside the trigger. Renders nothing for a tab without downloads.
 *
 * @param jobId - The job whose artifacts to download.
 * @param tabLabel - The active tab's name.
 * @param items - The tab's downloads.
 */
export function ResultDownloadMenu({ jobId, tabLabel, items }: ResultDownloadMenuProps) {
  const t = useT()
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<ErrorDescriptor | null>(null)

  if (items.length === 0) return null

  async function handleSelect(item: DownloadSpec) {
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
    <div className="flex items-center gap-2">
      {error && <span className="text-sm text-danger">{t(error.key, error.vars)}</span>}
      <Menu
        align="end"
        trigger={(props) => (
          <DownloadButton
            {...props}
            label={t('results.download_tab', { tab: tabLabel })}
            disabled={busy !== null}
            busy={busy !== null}
            className="gap-1 px-2"
          >
            <ChevronDownIcon className="h-3.5 w-3.5" />
          </DownloadButton>
        )}
      >
        {items.map((item) => (
          <MenuItem key={item.name} onSelect={() => void handleSelect(item)}>
            {item.label}
          </MenuItem>
        ))}
      </Menu>
    </div>
  )
}
