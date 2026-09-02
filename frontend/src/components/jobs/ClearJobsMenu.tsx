import { useState } from 'react'
import { Button, ChevronDownIcon, DeleteButton, Menu, MenuItem } from '@infra/ui'
import { isActive, useClearJobs } from '../../hooks/useJobs'
import { useT } from '../../i18n/LanguageContext'
import type { JobListItem } from '../../api/types'

interface ClearJobsMenuProps {
  /** The caller's current jobs; used to derive the clearable id sets. */
  jobs: JobListItem[]
}

/** Which set a pending confirmation targets. */
type ConfirmScope = 'finished' | 'all'

/**
 * A "Clear" dropdown that removes jobs from the list. Offers "Clear finished"
 * (terminal jobs only, leaving queued/running runs untouched) and "Clear all".
 * Both actions require an inline confirmation because deletion is irreversible
 * (jobs live only in memory). Mirrors {@link BatchDownloadMenu}: disables while
 * busy, and surfaces a partial-failure message inline next to the trigger.
 *
 * The confirmation replaces the items inside the same panel rather than opening
 * a dialog, so the question stays where the row it is about was.
 *
 * @param jobs - The caller's current jobs.
 */
export function ClearJobsMenu({ jobs }: ClearJobsMenuProps) {
  const t = useT()
  const [confirm, setConfirm] = useState<ConfirmScope | null>(null)
  const [error, setError] = useState<string | null>(null)
  const clear = useClearJobs()

  const allIds = jobs.map((job) => job.job_id)
  const finishedIds = jobs.filter((job) => !isActive(job)).map((job) => job.job_id)
  const disabled = allIds.length === 0 || clear.isPending
  const confirmIds = confirm === 'all' ? allIds : finishedIds

  async function runClear(close: () => void) {
    setError(null)
    const res = await clear.mutateAsync(confirmIds)
    setConfirm(null)
    close()
    if (res.failed > 0) {
      setError(
        t('jobs.clear_partial_failure', {
          cleared: res.cleared,
          total: res.cleared + res.failed,
          failed: res.failed,
        }),
      )
    }
  }

  return (
    <div className="flex items-center gap-2">
      {error && <span className="text-sm text-danger">{error}</span>}
      {/* Only the trigger becomes an icon. The menu items below carry counts
          and the confirmation asks a question — both need their words. */}
      <Menu
        align="end"
        // Closing is the answer "not now": a panel that reopens on yesterday's
        // question is one click from destroying the list.
        onOpenChange={(open) => !open && setConfirm(null)}
        trigger={(props) => (
          <DeleteButton
            {...props}
            label={t('jobs.clear')}
            hint={allIds.length === 0 ? t('jobs.no_jobs_to_clear') : undefined}
            disabled={disabled}
            busy={clear.isPending}
            className="gap-1 px-2"
          >
            <ChevronDownIcon className="h-3.5 w-3.5" />
          </DeleteButton>
        )}
      >
        {({ close }) =>
          confirm === null ? (
            <>
              <MenuItem
                closeOnSelect={false}
                disabled={finishedIds.length === 0}
                onSelect={() => setConfirm('finished')}
              >
                {t('jobs.clear_finished', { count: finishedIds.length })}
              </MenuItem>
              <MenuItem tone="danger" closeOnSelect={false} onSelect={() => setConfirm('all')}>
                {t('jobs.clear_all', { count: allIds.length })}
              </MenuItem>
            </>
          ) : (
            <div className="px-3 py-2">
              <p className="max-w-[16rem] text-sm text-foreground">
                {t(confirmIds.length === 1 ? 'jobs.clear_confirm_one' : 'jobs.clear_confirm_other', {
                  count: confirmIds.length,
                })}
              </p>
              <div className="mt-2 flex justify-end gap-2">
                <Button
                  size="sm"
                  variant="secondary"
                  disabled={clear.isPending}
                  onClick={() => setConfirm(null)}
                >
                  {t('common.cancel')}
                </Button>
                <Button
                  size="sm"
                  variant="danger"
                  disabled={clear.isPending}
                  onClick={() => void runClear(close)}
                >
                  {clear.isPending ? t('jobs.clearing') : t('jobs.clear_confirm_button')}
                </Button>
              </div>
            </div>
          )
        }
      </Menu>
    </div>
  )
}
