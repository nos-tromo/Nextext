# Delete jobs & clear the list — design

Date: 2026-07-01
Status: Approved (pending spec review)

## Problem

The frontend job view (`BatchProgress` → one `JobCard` per job) offers no way to
remove a job or clear the list. An operator who processes files for one case and
then another has both cases' jobs stacked in the same list, with no way to wipe
the slate. Because artifacts (transcripts, ZIPs, combined JSONL) are downloaded
from that list — including the "Download all jobs" batch bundle, which spans
*every* completed job — a lingering job from a previous case risks mixing
outputs across cases. The operator needs to delete individual jobs and clear the
list between cases.

## Current state (what already exists)

- **Backend:** `DELETE /api/v1/jobs/{id}` is implemented and owner-scoped
  (`nextext/api/routes/jobs.py`, `JobManager.delete`). It removes the job from
  the in-memory dict, signals SSE subscribers to close, and deletes the per-job
  tempdir. It returns `204` on success, `404` on a missing/cross-owner job.
  There is **no true cancellation**: deleting a *running* job leaves its worker
  thread to run to completion, but the result is discarded because the state is
  no longer tracked.
- **Frontend API client:** `deleteJob(jobId)` already exists in
  `frontend/src/api/jobs.ts` (`DELETE /jobs/{id}`, tolerates the 204 empty
  body). It is currently never called from any UI.
- **List rendering:** `BatchProgress.tsx` renders a `JobCard` per job and an
  action row that today holds only `BatchDownloadMenu`. `completedCount` is
  derived from the authoritative `['jobs']` query.
- **Aggregate header:** `StatusBar` reads `summarizeJobs(jobs, live)`, which
  counts only jobs present in the authoritative `['jobs']` list and ignores
  `live` progress-store entries for jobs absent from that list. So once the list
  is invalidated after a delete, header counts recompute correctly.
- **Progress store:** each mounted `useJobStream` publishes into a shared
  zustand store and, on unmount, calls `removeJob` to drop its entry. A card
  that disappears from a refetched list therefore self-cleans.
- **Shared UI:** `@infra/ui` exports `Button` (and `Badge`, `Banner`, `Card`,
  `Spinner`, `cn`). There is **no** dialog/modal primitive.
- **Status helper:** `isActive(job)` (`hooks/useJobs.ts`) is
  `status === 'queued' || status === 'running'`. Terminal ("finished") jobs are
  its complement.

## Chosen approach

Reuse the existing dropdown-menu pattern (Approach A). A new `ClearJobsMenu`
component is modeled directly on `BatchDownloadMenu` — same outside-click and
Escape handling, same busy/disabled styling, same inline error span. Selecting a
bulk action flips the menu panel into an inline confirmation before it deletes.
This keeps the UI styled and consistent, is fully testable with React Testing
Library, and adds no new dependency (no `window.confirm`, no hand-rolled modal).

Rejected alternatives:
- **Native `window.confirm`** — least code, but an unstyled OS dialog
  inconsistent with the app.
- **New reusable modal primitive** — `@infra/ui` has no modal, so it would mean
  hand-rolling focus-trap/overlay for a single feature. Over-engineered (YAGNI).

## Detailed design

### 1. Mutations — `frontend/src/hooks/useJobs.ts`

- `useDeleteJob()` — a `useMutation` calling `deleteJob(id)`. `onSuccess`
  invalidates `['jobs']`. A `404` is treated as already-gone (resolve quietly,
  still invalidate) rather than surfaced as an error.
- `useClearJobs()` — a `useMutation` whose variables are `job_ids: string[]`.
  It deletes them concurrently via `Promise.allSettled`, invalidates `['jobs']`
  once at the end, and returns `{ cleared: number; failed: number }` so the UI
  can report partial failure. A per-id `404` counts as `cleared` (already gone).

### 2. Per-job delete — `frontend/src/components/jobs/JobCard.tsx`

Add a **Remove** button to the card header action row (next to the status label
and the existing "Show results" toggle). Behavior:

- Immediate delete on click — no confirmation (a single, explicitly labeled
  item). Approved.
- While the mutation is pending: disabled, labeled "Removing…".
- On non-404 failure: an inline error message on the card; the card stays.
- On success: the refetched `['jobs']` list no longer includes the job, the
  card unmounts, and `useJobStream` cleanup drops the progress-store entry.

### 3. Bulk clear — `frontend/src/components/jobs/ClearJobsMenu.tsx` (new)

A `Clear ▾` trigger and dropdown mirroring `BatchDownloadMenu`.

- **Props:** the job list (`JobListItem[]`). The component derives both id sets
  from it — `allIds` = every job; `finishedIds` = jobs where `!isActive(job)`.
- **Menu items:**
  - `Clear finished (N)` — deletes only terminal jobs; disabled when `N === 0`.
    Leaves queued/running jobs in place (approved) so an active run is never
    interrupted.
  - `Clear all (M)` — deletes every listed job; disabled when the list is empty.
- **Confirmation:** selecting either item flips the menu panel into an inline
  confirm — `Remove N jobs? This can't be undone.` with `[Cancel]` and a
  `[Clear]` button — before calling `useClearJobs`. Both bulk actions confirm
  (both irreversibly destroy in-memory-only results).
- **Busy/error:** a busy state disables the trigger during deletion; a
  partial-failure message (`Cleared X of Y; Z failed`) surfaces inline next to
  the trigger, mirroring `BatchDownloadMenu`'s error span. The menu closes on
  outside click or Escape.

### 4. Wiring — `frontend/src/components/jobs/BatchProgress.tsx`

Render `ClearJobsMenu` in the existing action row alongside `BatchDownloadMenu`,
passing the current `items`.

### 5. Reactivity & cleanup

- After any delete, `['jobs']` invalidation triggers a `BatchProgress` refetch.
  An emptied list renders the existing "No jobs yet." message.
- `StatusBar` recomputes from `summarizeJobs`; a 0-total set renders nothing.
- Progress-store entries for removed jobs are dropped by `useJobStream`'s unmount
  cleanup — no extra bookkeeping.

## Error handling

- **Per-job:** inline error text on the card for non-404 failures; 404 is
  swallowed (the job is gone either way).
- **Bulk:** `Promise.allSettled` so one failure never aborts the sweep; report
  `cleared`/`failed` counts inline. 404s count as cleared.

## Testing

- `ClearJobsMenu.test.tsx` (new; mirrors `BatchDownloadMenu.test.tsx`): open and
  close (outside click + Escape), disabled states (no finished / empty list),
  the inline-confirm flow, one `deleteJob` call per id, and the partial-failure
  message. Mocks `../../api/jobs`.
- Hook coverage for `useDeleteJob` and `useClearJobs`: assert the API calls and
  `['jobs']` invalidation; mock `../../api/jobs`.
- `JobCard` Remove-button test: mocks the SSE stream and `deleteJob`, asserts the
  click triggers deletion.
- Full `pnpm test`, `pnpm lint`, and `pnpm typecheck` pass (corepack
  `pnpm@9.12.0`, absolute `--dir`).

## Out of scope

- True server-side cancellation of a running job's worker thread (a separate
  backend concern; current behavior is documented above).
- Any change to the `DELETE` endpoint or `JobManager` — the backend already
  supports owner-scoped deletion.
- Undo / soft-delete — jobs are in-memory only with no recovery path by design.
