import { useEffect, useRef, useState } from 'react'
import { isActive, useClearJobs } from '../../hooks/useJobs'
import { cn } from '../../lib/cn'
import type { JobListItem } from '../../api/types'

interface ClearJobsMenuProps {
  /** The caller's current jobs; used to derive the clearable id sets. */
  jobs: JobListItem[]
}

/** Which set a pending confirmation targets. */
type ConfirmScope = 'finished' | 'all'

/**
 * A "Clear ▾" dropdown that removes jobs from the list. Offers "Clear finished"
 * (terminal jobs only, leaving queued/running runs untouched) and "Clear all".
 * Both actions require an inline confirmation because deletion is irreversible
 * (jobs live only in memory). Mirrors {@link BatchDownloadMenu}: closes on
 * outside click or Escape, disables while busy, and surfaces a partial-failure
 * message inline next to the trigger.
 *
 * @param jobs - The caller's current jobs.
 */
export function ClearJobsMenu({ jobs }: ClearJobsMenuProps) {
  const [open, setOpen] = useState(false)
  const [confirm, setConfirm] = useState<ConfirmScope | null>(null)
  const [error, setError] = useState<string | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const clear = useClearJobs()

  const allIds = jobs.map((job) => job.job_id)
  const finishedIds = jobs.filter((job) => !isActive(job)).map((job) => job.job_id)
  const disabled = allIds.length === 0 || clear.isPending
  const confirmIds = confirm === 'all' ? allIds : finishedIds

  useEffect(() => {
    if (!open) return
    function onPointerDown(event: PointerEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false)
        setConfirm(null)
      }
    }
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        setOpen(false)
        setConfirm(null)
      }
    }
    document.addEventListener('pointerdown', onPointerDown)
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('pointerdown', onPointerDown)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [open])

  async function runClear() {
    setError(null)
    const res = await clear.mutateAsync(confirmIds)
    setConfirm(null)
    setOpen(false)
    if (res.failed > 0) {
      setError(`Cleared ${res.cleared} of ${res.cleared + res.failed}; ${res.failed} failed`)
    }
  }

  return (
    <div ref={containerRef} className="relative flex items-center gap-2">
      {error && <span className="text-sm text-danger">{error}</span>}
      <button
        type="button"
        disabled={disabled}
        aria-haspopup="menu"
        aria-expanded={open}
        title={allIds.length === 0 ? 'No jobs to clear' : undefined}
        onClick={() => {
          setConfirm(null)
          setOpen((v) => !v)
        }}
        className={cn(
          'rounded border border-border px-3 py-1 text-sm transition-colors',
          disabled
            ? 'cursor-not-allowed text-muted-foreground'
            : 'text-foreground hover:border-primary hover:text-primary',
        )}
      >
        {clear.isPending ? 'Clearing…' : 'Clear ▾'}
      </button>
      {open && (
        <div
          role="menu"
          className="absolute right-0 top-full z-10 mt-1 min-w-max rounded-md border border-border bg-muted py-1 shadow-lg"
        >
          {confirm === null ? (
            <>
              <button
                type="button"
                role="menuitem"
                disabled={finishedIds.length === 0}
                onClick={() => setConfirm('finished')}
                className={cn(
                  'block w-full px-3 py-1.5 text-left text-sm',
                  finishedIds.length === 0
                    ? 'cursor-not-allowed text-muted-foreground'
                    : 'text-foreground hover:bg-accent hover:text-primary',
                )}
              >
                {`Clear finished (${finishedIds.length})`}
              </button>
              <button
                type="button"
                role="menuitem"
                onClick={() => setConfirm('all')}
                className="block w-full px-3 py-1.5 text-left text-sm text-foreground hover:bg-accent hover:text-primary"
              >
                {`Clear all (${allIds.length})`}
              </button>
            </>
          ) : (
            <div className="px-3 py-2">
              <p className="max-w-[16rem] text-sm text-foreground">
                {`Remove ${confirmIds.length} ${confirmIds.length === 1 ? 'job' : 'jobs'}? This can't be undone.`}
              </p>
              <div className="mt-2 flex justify-end gap-2">
                <button
                  type="button"
                  disabled={clear.isPending}
                  onClick={() => setConfirm(null)}
                  className="rounded border border-border px-2 py-1 text-sm text-foreground hover:border-primary hover:text-primary disabled:cursor-not-allowed"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  disabled={clear.isPending}
                  onClick={() => void runClear()}
                  className="rounded border border-danger px-2 py-1 text-sm font-medium text-danger transition-opacity hover:opacity-80 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {clear.isPending ? 'Clearing…' : 'Clear'}
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
