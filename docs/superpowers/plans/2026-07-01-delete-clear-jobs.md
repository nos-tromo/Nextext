# Delete Jobs & Clear the List — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-job delete and a bulk "Clear ▾" menu (Clear finished / Clear all) to the frontend job view so an operator can remove jobs and wipe the list between cases.

**Architecture:** Pure frontend wiring — the backend `DELETE /api/v1/jobs/{id}` and the `deleteJob` API client already exist. Two new react-query mutations (`useDeleteJob`, `useClearJobs`) drive a `Remove` button on each `JobCard` and a new `ClearJobsMenu` dropdown modeled on the existing `BatchDownloadMenu`. Both mutations invalidate the `['jobs']` query so the list, the header `StatusBar`, and the progress store self-update.

**Tech Stack:** React 19, TypeScript, `@tanstack/react-query` v5, Tailwind (via `@infra/ui` tokens), Vitest + React Testing Library (happy-dom).

## Global Constraints

- Every new/modified TS function and component gets a TSDoc doc comment matching the surrounding style (e.g. `BatchDownloadMenu`, `useJobs`). Copied verbatim from CLAUDE.md's docstring rule, adapted to TS.
- Follow the existing `BatchDownloadMenu` dropdown pattern: `role="menu"` / `role="menuitem"`, outside-click + Escape close, busy/disabled styling, inline error `<span>`.
- Tailwind: use theme tokens only — `text-foreground`, `text-muted-foreground`, `text-primary`, `text-danger`, `border-border`, `border-danger`, `bg-muted`, `bg-accent`, plus core utilities (`opacity-*`, `disabled:*`). No hard-coded colors, no un-configured opacity-on-token syntax.
- "Finished" (terminal) means `!isActive(job)`, where `isActive` is the existing helper in `hooks/useJobs.ts` (`status === 'queued' || 'running'`). Reuse it — do not re-enumerate statuses.
- Per-job delete is **immediate** (no confirm). Both bulk actions **confirm** inline. *Clear finished* leaves queued/running jobs untouched.
- A `404` from `deleteJob` counts as "already gone" (success), never a surfaced error.
- Tests mock at the `../api/jobs` (`deleteJob`) layer and wrap components/hooks in a fresh `QueryClient` with `retry: false`. Mirror `StatusBar.test.tsx` / `BatchDownloadMenu.test.tsx`.

## Toolchain (run once before the tasks)

`pnpm` is pinned to `9.12.0` and `node_modules` is not pre-installed. From any cwd:

```bash
corepack prepare pnpm@9.12.0 --activate
pnpm --dir "$(git rev-parse --show-toplevel)"/frontend install --frozen-lockfile
```

The absolute `--dir` is required (a relative `--dir frontend` breaks when cwd is already inside `frontend/`). The `install` is needed for `pnpm lint` (eslint flat config needs top-level devDeps); it leaves the lockfile untouched. Gates: `pnpm --dir <abs> test`, `pnpm --dir <abs> typecheck`, `pnpm --dir <abs> lint`.

## File Structure

- **Modify** `frontend/src/hooks/useJobs.ts` — add `useDeleteJob()` and `useClearJobs()` mutations + the `ClearJobsResult` type. (Task 1)
- **Create** `frontend/src/hooks/useJobs.test.tsx` — hook tests for the two mutations. (Task 1)
- **Create** `frontend/src/components/jobs/ClearJobsMenu.tsx` — the `Clear ▾` dropdown with inline confirm. (Task 2)
- **Create** `frontend/src/components/jobs/ClearJobsMenu.test.tsx` — menu/confirm/partial-failure tests. (Task 2)
- **Modify** `frontend/src/components/jobs/BatchProgress.tsx` — render `ClearJobsMenu` in the action row. (Task 3)
- **Create** `frontend/src/components/jobs/BatchProgress.test.tsx` — wiring test. (Task 3)
- **Modify** `frontend/src/components/jobs/JobCard.tsx` — add the `Remove` button + inline error. (Task 4)
- **Create** `frontend/src/components/jobs/JobCard.test.tsx` — remove-button tests. (Task 4)

---

### Task 1: Delete & clear mutations

**Files:**
- Modify: `frontend/src/hooks/useJobs.ts`
- Test: `frontend/src/hooks/useJobs.test.tsx` (create)

**Interfaces:**
- Consumes: `deleteJob(jobId: string): Promise<void>` from `../api/jobs`; `ApiError` (with `.status: number`) from `../api/client`.
- Produces:
  - `useDeleteJob(): UseMutationResult<void, Error, string>` — variable is the `job_id`.
  - `useClearJobs(): UseMutationResult<ClearJobsResult, Error, string[]>` — variable is the id list.
  - `interface ClearJobsResult { cleared: number; failed: number }`.
  - (Unchanged, reused by later tasks: `isActive(job: JobListItem): boolean`.)

- [ ] **Step 1: Write the failing tests**

Create `frontend/src/hooks/useJobs.test.tsx`:

```tsx
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'

vi.mock('../api/jobs', () => ({
  deleteJob: vi.fn(),
  listJobs: vi.fn(),
  submitJob: vi.fn(),
}))

import { deleteJob } from '../api/jobs'
import { ApiError } from '../api/client'
import { useClearJobs, useDeleteJob } from './useJobs'

const mockedDelete = vi.mocked(deleteJob)

function makeWrapper() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={qc}>{children}</QueryClientProvider>
  }
}

beforeEach(() => {
  mockedDelete.mockReset()
  mockedDelete.mockResolvedValue(undefined)
})
afterEach(() => vi.restoreAllMocks())

describe('useDeleteJob', () => {
  it('deletes the job', async () => {
    const { result } = renderHook(() => useDeleteJob(), { wrapper: makeWrapper() })
    result.current.mutate('j1')
    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(mockedDelete).toHaveBeenCalledWith('j1')
  })

  it('treats a 404 as already-gone (success)', async () => {
    mockedDelete.mockRejectedValueOnce(new ApiError(404, 'not found'))
    const { result } = renderHook(() => useDeleteJob(), { wrapper: makeWrapper() })
    result.current.mutate('gone')
    await waitFor(() => expect(result.current.isSuccess).toBe(true))
  })

  it('surfaces a non-404 failure', async () => {
    mockedDelete.mockRejectedValueOnce(new ApiError(500, 'boom'))
    const { result } = renderHook(() => useDeleteJob(), { wrapper: makeWrapper() })
    result.current.mutate('j1')
    await waitFor(() => expect(result.current.isError).toBe(true))
  })
})

describe('useClearJobs', () => {
  it('clears every id and reports counts, tolerating failures', async () => {
    mockedDelete.mockImplementation(async (id: string) => {
      if (id === 'bad') throw new ApiError(500, 'boom')
      if (id === 'gone') throw new ApiError(404, 'not found') // counts as cleared
      return undefined
    })
    const { result } = renderHook(() => useClearJobs(), { wrapper: makeWrapper() })
    result.current.mutate(['ok', 'gone', 'bad'])
    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(result.current.data).toEqual({ cleared: 2, failed: 1 })
    expect(mockedDelete).toHaveBeenCalledTimes(3)
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pnpm --dir "$(git rev-parse --show-toplevel)"/frontend test src/hooks/useJobs.test.tsx`
Expected: FAIL — `useDeleteJob`/`useClearJobs` are not exported from `./useJobs`.

- [ ] **Step 3: Implement the mutations**

Edit `frontend/src/hooks/useJobs.ts`. Change the import block at the top:

```tsx
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { deleteJob, listJobs, submitJob } from '../api/jobs'
import { ApiError } from '../api/client'
import type { JobListItem, JobOptions } from '../api/types'
```

Then append, after the existing `isActive` function:

```tsx
/**
 * Delete a single job, then refresh the jobs list. A 404 is treated as
 * already-gone (resolves quietly) so a double-click or a delete/refetch race
 * never surfaces a spurious error.
 */
export function useDeleteJob() {
  const qc = useQueryClient()
  return useMutation<void, Error, string>({
    mutationFn: async (jobId) => {
      try {
        await deleteJob(jobId)
      } catch (err) {
        if (err instanceof ApiError && err.status === 404) return // already gone
        throw err
      }
    },
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['jobs'] })
    },
  })
}

/** Outcome of a bulk clear: how many deletions succeeded vs. failed. */
export interface ClearJobsResult {
  cleared: number
  failed: number
}

/**
 * Delete many jobs concurrently, tolerating individual failures, then refresh
 * the jobs list once. A per-job 404 counts as cleared (already gone). The
 * mutation always resolves (never rejects) so the list refetch runs regardless
 * of partial failure; callers inspect {@link ClearJobsResult} to report it.
 *
 * @returns Counts of successfully cleared and failed deletions.
 */
export function useClearJobs() {
  const qc = useQueryClient()
  return useMutation<ClearJobsResult, Error, string[]>({
    mutationFn: async (jobIds) => {
      const settled = await Promise.allSettled(
        jobIds.map(async (jobId) => {
          try {
            await deleteJob(jobId)
          } catch (err) {
            if (err instanceof ApiError && err.status === 404) return // already gone -> cleared
            throw err
          }
        }),
      )
      const failed = settled.filter((r) => r.status === 'rejected').length
      return { cleared: settled.length - failed, failed }
    },
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['jobs'] })
    },
  })
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pnpm --dir "$(git rev-parse --show-toplevel)"/frontend test src/hooks/useJobs.test.tsx`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/hooks/useJobs.ts frontend/src/hooks/useJobs.test.tsx
git commit -m "feat(frontend): add useDeleteJob and useClearJobs mutations"
```

---

### Task 2: `ClearJobsMenu` dropdown

**Files:**
- Create: `frontend/src/components/jobs/ClearJobsMenu.tsx`
- Test: `frontend/src/components/jobs/ClearJobsMenu.test.tsx`

**Interfaces:**
- Consumes: `useClearJobs()` and `isActive()` from `../../hooks/useJobs`; `cn` from `../../lib/cn`; `JobListItem` from `../../api/types`.
- Produces: `ClearJobsMenu({ jobs }: { jobs: JobListItem[] }): JSX.Element`.

- [ ] **Step 1: Write the failing tests**

Create `frontend/src/components/jobs/ClearJobsMenu.test.tsx`:

```tsx
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactElement } from 'react'

vi.mock('../../api/jobs', () => ({
  deleteJob: vi.fn(),
  listJobs: vi.fn(),
  submitJob: vi.fn(),
}))

import { deleteJob } from '../../api/jobs'
import { ApiError } from '../../api/client'
import { ClearJobsMenu } from './ClearJobsMenu'
import type { JobListItem, JobStatus } from '../../api/types'

const mockedDelete = vi.mocked(deleteJob)

function mkJob(job_id: string, status: JobStatus): JobListItem {
  return {
    job_id,
    status,
    file_name: `${job_id}.wav`,
    stage: null,
    progress: 0,
    error: null,
    created_at: 't',
    started_at: null,
    finished_at: null,
    task: 'transcribe',
  }
}

function renderMenu(ui: ReactElement) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>)
}

beforeEach(() => {
  mockedDelete.mockReset()
  mockedDelete.mockResolvedValue(undefined)
})
afterEach(() => vi.restoreAllMocks())

describe('ClearJobsMenu', () => {
  it('disables the trigger when there are no jobs', () => {
    renderMenu(<ClearJobsMenu jobs={[]} />)
    expect(screen.getByRole('button', { name: /Clear/ })).toBeDisabled()
  })

  it('opens the menu with finished + all counts', () => {
    renderMenu(<ClearJobsMenu jobs={[mkJob('a', 'completed'), mkJob('b', 'running'), mkJob('c', 'failed')]} />)
    fireEvent.click(screen.getByRole('button', { name: /Clear ▾/ }))
    expect(screen.getByRole('menuitem', { name: 'Clear finished (2)' })).toBeInTheDocument()
    expect(screen.getByRole('menuitem', { name: 'Clear all (3)' })).toBeInTheDocument()
  })

  it('disables "Clear finished" when only active jobs exist', () => {
    renderMenu(<ClearJobsMenu jobs={[mkJob('a', 'running'), mkJob('b', 'queued')]} />)
    fireEvent.click(screen.getByRole('button', { name: /Clear ▾/ }))
    expect(screen.getByRole('menuitem', { name: 'Clear finished (0)' })).toBeDisabled()
    expect(screen.getByRole('menuitem', { name: 'Clear all (2)' })).toBeEnabled()
  })

  it('deletes every job on confirmed "Clear all"', async () => {
    renderMenu(<ClearJobsMenu jobs={[mkJob('a', 'completed'), mkJob('b', 'running')]} />)
    fireEvent.click(screen.getByRole('button', { name: /Clear ▾/ }))
    fireEvent.click(screen.getByRole('menuitem', { name: 'Clear all (2)' }))
    expect(screen.getByText(/Remove 2 jobs\?/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Clear' }))
    await waitFor(() => expect(mockedDelete).toHaveBeenCalledTimes(2))
    expect(mockedDelete).toHaveBeenCalledWith('a')
    expect(mockedDelete).toHaveBeenCalledWith('b')
  })

  it('deletes only finished jobs on confirmed "Clear finished"', async () => {
    renderMenu(<ClearJobsMenu jobs={[mkJob('done', 'completed'), mkJob('run', 'running')]} />)
    fireEvent.click(screen.getByRole('button', { name: /Clear ▾/ }))
    fireEvent.click(screen.getByRole('menuitem', { name: 'Clear finished (1)' }))
    fireEvent.click(screen.getByRole('button', { name: 'Clear' }))
    await waitFor(() => expect(mockedDelete).toHaveBeenCalledTimes(1))
    expect(mockedDelete).toHaveBeenCalledWith('done')
    expect(mockedDelete).not.toHaveBeenCalledWith('run')
  })

  it('cancels without deleting and returns to the menu', () => {
    renderMenu(<ClearJobsMenu jobs={[mkJob('a', 'completed')]} />)
    fireEvent.click(screen.getByRole('button', { name: /Clear ▾/ }))
    fireEvent.click(screen.getByRole('menuitem', { name: 'Clear all (1)' }))
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(mockedDelete).not.toHaveBeenCalled()
    expect(screen.getByRole('menuitem', { name: 'Clear all (1)' })).toBeInTheDocument()
  })

  it('reports a partial failure inline', async () => {
    mockedDelete.mockImplementation(async (id: string) => {
      if (id === 'bad') throw new ApiError(500, 'boom')
      return undefined
    })
    renderMenu(<ClearJobsMenu jobs={[mkJob('ok', 'completed'), mkJob('bad', 'failed')]} />)
    fireEvent.click(screen.getByRole('button', { name: /Clear ▾/ }))
    fireEvent.click(screen.getByRole('menuitem', { name: 'Clear all (2)' }))
    fireEvent.click(screen.getByRole('button', { name: 'Clear' }))
    await waitFor(() => expect(screen.getByText('Cleared 1 of 2; 1 failed')).toBeInTheDocument())
  })

  it('closes on Escape', () => {
    renderMenu(<ClearJobsMenu jobs={[mkJob('a', 'completed')]} />)
    fireEvent.click(screen.getByRole('button', { name: /Clear ▾/ }))
    expect(screen.getByRole('menu')).toBeInTheDocument()
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(screen.queryByRole('menu')).toBeNull()
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pnpm --dir "$(git rev-parse --show-toplevel)"/frontend test src/components/jobs/ClearJobsMenu.test.tsx`
Expected: FAIL — `./ClearJobsMenu` module does not exist.

- [ ] **Step 3: Implement the component**

Create `frontend/src/components/jobs/ClearJobsMenu.tsx`:

```tsx
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pnpm --dir "$(git rev-parse --show-toplevel)"/frontend test src/components/jobs/ClearJobsMenu.test.tsx`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/jobs/ClearJobsMenu.tsx frontend/src/components/jobs/ClearJobsMenu.test.tsx
git commit -m "feat(frontend): add ClearJobsMenu (clear finished / clear all) with inline confirm"
```

---

### Task 3: Wire `ClearJobsMenu` into `BatchProgress`

**Files:**
- Modify: `frontend/src/components/jobs/BatchProgress.tsx`
- Test: `frontend/src/components/jobs/BatchProgress.test.tsx` (create)

**Interfaces:**
- Consumes: `ClearJobsMenu({ jobs })` from `./ClearJobsMenu`; existing `useJobs()`, `BatchDownloadMenu`, `JobCard`.
- Produces: no new exports (renders `ClearJobsMenu` beside `BatchDownloadMenu`).

- [ ] **Step 1: Write the failing test**

Create `frontend/src/components/jobs/BatchProgress.test.tsx`. `JobCard` is mocked to avoid its SSE stream; `ResultPanel` is not reached:

```tsx
import { afterEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactElement } from 'react'
import type { JobListItem } from '../../api/types'

vi.mock('./JobCard', () => ({
  JobCard: ({ job }: { job: JobListItem }) => <div data-testid="jobcard">{job.file_name}</div>,
}))

import { BatchProgress } from './BatchProgress'

function renderWithClient(ui: ReactElement) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>)
}

function stubJobs(jobs: Partial<JobListItem>[]) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => new Response(JSON.stringify({ jobs }), { status: 200, headers: { 'content-type': 'application/json' } })),
  )
}

afterEach(() => vi.restoreAllMocks())

describe('BatchProgress', () => {
  it('renders the Clear control alongside the jobs when the list is non-empty', async () => {
    stubJobs([{ job_id: 'a', status: 'completed', file_name: 'a.wav' }])
    renderWithClient(<BatchProgress />)
    expect(await screen.findByRole('button', { name: /Clear ▾/ })).toBeInTheDocument()
    expect(screen.getByTestId('jobcard')).toHaveTextContent('a.wav')
  })

  it('shows the empty state and no Clear control when there are no jobs', async () => {
    stubJobs([])
    renderWithClient(<BatchProgress />)
    expect(await screen.findByText('No jobs yet.')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Clear/ })).toBeNull()
  })
})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pnpm --dir "$(git rev-parse --show-toplevel)"/frontend test src/components/jobs/BatchProgress.test.tsx`
Expected: FAIL — no `Clear ▾` button (`ClearJobsMenu` not wired yet).

- [ ] **Step 3: Wire the component**

Edit `frontend/src/components/jobs/BatchProgress.tsx`. Add the import after the `BatchDownloadMenu` import:

```tsx
import { ClearJobsMenu } from './ClearJobsMenu'
```

Replace the action-row `<div>` and its contents:

```tsx
      <div className="flex items-center justify-end gap-2">
        <BatchDownloadMenu completedCount={completedCount} />
        <ClearJobsMenu jobs={items} />
      </div>
```

(The surrounding `return (<div className="space-y-3"> … {items.map(...)} </div>)` is unchanged.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `pnpm --dir "$(git rev-parse --show-toplevel)"/frontend test src/components/jobs/BatchProgress.test.tsx`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/jobs/BatchProgress.tsx frontend/src/components/jobs/BatchProgress.test.tsx
git commit -m "feat(frontend): surface ClearJobsMenu in the jobs list action row"
```

---

### Task 4: Per-job `Remove` button on `JobCard`

**Files:**
- Modify: `frontend/src/components/jobs/JobCard.tsx`
- Test: `frontend/src/components/jobs/JobCard.test.tsx` (create)

**Interfaces:**
- Consumes: `useDeleteJob()` from `../../hooks/useJobs`.
- Produces: no new exports (adds a `Remove` button + inline error to `JobCard`).

- [ ] **Step 1: Write the failing tests**

Create `frontend/src/components/jobs/JobCard.test.tsx`. Mock the SSE hook (static completed progress), the `ResultPanel` (heavy, unreached), and `deleteJob`:

```tsx
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactElement } from 'react'
import type { JobListItem, JobStatus } from '../../api/types'

vi.mock('../../hooks/useJobStream', () => ({
  useJobStream: () => ({
    status: 'completed',
    stageIndex: 0,
    stageLabel: null,
    progress: 1,
    error: null,
    skipped: false,
    terminal: true,
  }),
}))
vi.mock('../results/ResultPanel', () => ({ ResultPanel: () => null }))
vi.mock('../../api/jobs', () => ({
  deleteJob: vi.fn(),
  listJobs: vi.fn(),
  submitJob: vi.fn(),
}))

import { deleteJob } from '../../api/jobs'
import { JobCard } from './JobCard'

const mockedDelete = vi.mocked(deleteJob)

function mkJob(job_id: string, status: JobStatus): JobListItem {
  return {
    job_id,
    status,
    file_name: `${job_id}.wav`,
    stage: null,
    progress: 0,
    error: null,
    created_at: 't',
    started_at: null,
    finished_at: null,
    task: 'transcribe',
  }
}

function renderCard(ui: ReactElement) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>)
}

beforeEach(() => {
  mockedDelete.mockReset()
  mockedDelete.mockResolvedValue(undefined)
})
afterEach(() => vi.restoreAllMocks())

describe('JobCard Remove', () => {
  it('deletes the job when Remove is clicked', async () => {
    renderCard(<JobCard job={mkJob('j1', 'completed')} />)
    fireEvent.click(screen.getByRole('button', { name: 'Remove' }))
    await waitFor(() => expect(mockedDelete).toHaveBeenCalledWith('j1'))
  })

  it('shows an inline error when removal fails', async () => {
    mockedDelete.mockRejectedValue(new Error('nope'))
    renderCard(<JobCard job={mkJob('j1', 'completed')} />)
    fireEvent.click(screen.getByRole('button', { name: 'Remove' }))
    await waitFor(() => expect(screen.getByText(/Could not remove job/)).toBeInTheDocument())
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pnpm --dir "$(git rev-parse --show-toplevel)"/frontend test src/components/jobs/JobCard.test.tsx`
Expected: FAIL — no button named `Remove`.

- [ ] **Step 3: Implement the button**

Edit `frontend/src/components/jobs/JobCard.tsx`.

(a) Add the import after the `useJobStream` import:

```tsx
import { useDeleteJob } from '../../hooks/useJobs'
```

(b) Inside the component, add the mutation next to the other hooks (right after `const [showResults, setShowResults] = useState(false)`):

```tsx
  const del = useDeleteJob()
```

(c) In the header action row, add a `Remove` button after the status `<span>`. Replace this block:

```tsx
          <span className="text-sm text-muted-foreground">{LABEL[p.status]}</span>
        </div>
```

with:

```tsx
          <span className="text-sm text-muted-foreground">{LABEL[p.status]}</span>
          <button
            type="button"
            disabled={del.isPending}
            onClick={() => del.mutate(job.job_id)}
            className="text-sm text-muted-foreground transition-colors hover:text-danger disabled:cursor-not-allowed disabled:opacity-60"
          >
            {del.isPending ? 'Removing…' : 'Remove'}
          </button>
        </div>
```

(d) Add an inline error line immediately after the closing `</p>` of the status message (before the `{p.status === 'completed' && showResults && (` block):

```tsx
      {del.isError && (
        <p className="mt-1 text-sm text-danger">{`Could not remove job: ${del.error?.message ?? 'unknown error'}`}</p>
      )}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pnpm --dir "$(git rev-parse --show-toplevel)"/frontend test src/components/jobs/JobCard.test.tsx`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/jobs/JobCard.tsx frontend/src/components/jobs/JobCard.test.tsx
git commit -m "feat(frontend): add per-job Remove button to JobCard"
```

---

### Task 5: Full gates & final verification

**Files:** none (verification only).

- [ ] **Step 1: Run the whole frontend test suite**

Run: `pnpm --dir "$(git rev-parse --show-toplevel)"/frontend test`
Expected: PASS — all suites green, including the four new/updated files.

- [ ] **Step 2: Typecheck**

Run: `pnpm --dir "$(git rev-parse --show-toplevel)"/frontend typecheck`
Expected: no errors.

- [ ] **Step 3: Lint**

Run: `pnpm --dir "$(git rev-parse --show-toplevel)"/frontend lint`
Expected: no errors. (If it fails with `ERR_MODULE_NOT_FOUND: 'globals'`, run the `install --frozen-lockfile` from the Toolchain section, then re-run.)

- [ ] **Step 4: Manual smoke (optional, no browser required)**

A live browser check isn't possible here (Playwright/Chrome unavailable). The component/integration tests above are the behavioral gate. If a reviewer wants a visual pass, they can run `pnpm --dir <abs> dev` locally against a stub backend.

- [ ] **Step 5: Final commit (only if Steps 1–3 required fixups)**

```bash
git add -A
git commit -m "chore(frontend): satisfy typecheck/lint for delete-jobs feature"
```

---

## Self-Review

**Spec coverage:**
- Per-job delete → Task 4 (`Remove` button + `useDeleteJob`). ✓
- Bulk `Clear ▾` menu with *Clear finished* / *Clear all* → Task 2 (`ClearJobsMenu`), Task 3 (wiring). ✓
- Inline confirm on both bulk actions → Task 2 (confirm view + tests). ✓
- Per-job delete immediate (no confirm) → Task 4 (direct `mutate` on click). ✓
- *Clear finished* leaves queued/running untouched → Task 2 (`!isActive` filter + "only finished" test). ✓
- 404 tolerated as already-gone → Task 1 (both mutations + tests). ✓
- Partial-failure reporting via `allSettled` → Task 1 (`useClearJobs`) + Task 2 (inline message test). ✓
- Reactivity/cleanup (list invalidation → StatusBar recompute, progress-store self-clean) → Task 1 (`invalidateQueries(['jobs'])`); existing `useJobStream` unmount cleanup is unchanged and needs no new code. ✓
- Error handling (per-job inline, bulk inline) → Task 4 + Task 2. ✓
- Out of scope (server-side cancellation, endpoint changes, undo) → not implemented, per spec. ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to Task N" — every step has literal code and exact commands. ✓

**Type consistency:** `useDeleteJob(): UseMutationResult<void, Error, string>`, `useClearJobs(): UseMutationResult<ClearJobsResult, Error, string[]>`, `ClearJobsResult { cleared; failed }`, `ClearJobsMenu({ jobs: JobListItem[] })` — names and signatures match across Tasks 1–4. `isActive` reused unchanged. `del.error?.message` guards the nullable error. Menu labels (`Clear finished (N)`, `Clear all (M)`, `Remove`, `Cancel`, `Clear`) match between component and tests. ✓
