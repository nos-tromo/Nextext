# Nextext — AppShell Adoption Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adopt `@infra/ui` v0.9.0 in the Nextext SPA — `AppShell` (fixed
chrome, no sidebar), `PageHeader`, sign-out menu — plus a full-width
two-column home (upload card beside the jobs list), the always-reserved
StatusBar strip removed, and the home-grown button/select/spinner/card
replaced by the shared primitives.

**Architecture:** Plan 3 of the federation rollout (design:
`infra-ui/docs/2026-08-04-app-shell-federation-design.md` in the infra-ui
repo). Frontend-only. `Shell.tsx` keeps its responsibilities (owner SSE
stream, version, whoami) but delegates chrome to `AppShell`; the StatusBar
moves into the jobs column where it only occupies space when jobs exist.
Results stay inline in the job card (the separate results-view redesign is
deferred — it needs its own design pass; record this in the PR).

**Tech Stack:** React 19 + Vite + Tailwind v4 + `@infra/ui` (tag-pinned
pnpm git dep) + vitest.

## Global Constraints

- All frontend commands run inside `frontend/` with pnpm.
- Functionality preserved: jobs SSE stream (mounted ONCE in the shell),
  reload re-discovery, upload guard, per-file errors, i18n en/de parity
  (`i18n.test.ts`), theme toggle, user display; sign-out is the one
  addition. Accent CSS already follows the federation pattern — do not
  touch `globals.css`.
- Semantic tokens only; panels use the `Card` primitive (opaque `bg-muted`
  tile in v0.9.0), no shadows.
- Known accepted limitation: `AppShell` v0.9.0 forwards no `menuLabel`, so
  the user-menu aria-label prefix stays "Account" in both locales.
- Tests stay behavior-based; when a moved element makes an assertion
  location-dependent, move the assertion with the element, never delete
  the behavior check.
- Confidentiality: synthetic data only; no local machine paths committed.
- Working branch: `feature/app-shell` off current `main` (the controller
  creates it and commits this plan; implementers work on it).

---

### Task 1: Bump the `@infra/ui` pin to the v0.9.1 candidate

> The pin targets commit `58ae43a…` — the v0.9.1 candidate from infra-ui
> PR #36 (PageHeader banner-role fix). Repin to the `v0.9.1` tag in a
> follow-up commit once that PR merges.

**Files:**
- Modify: `frontend/package.json:21`
- Modify: `frontend/pnpm-lock.yaml` (via install)

**Interfaces:**
- Produces: `@infra/ui` v0.9.0 in `node_modules` — adds `AppShell`,
  `PageHeader`, `UserMenu`, `Select`, tile-style `Card`; `AppHeader` still
  exported (used until Task 3). NOTE v0.9.0's `Card` fill change
  (`bg-muted/30` → `bg-muted`) — Nextext imports no `Card` today, so
  nothing restyles at this step.

- [ ] **Step 1: Bump the pin** — in `frontend/package.json` change

```json
"@infra/ui": "github:nos-tromo/infra-ui#v0.8.1",
```

to

```json
"@infra/ui": "github:nos-tromo/infra-ui#58ae43a498cffc1058e040b0d2b29e0d07f1d941",
```

- [ ] **Step 2: Install and run the existing gates**

```bash
cd frontend && pnpm install && pnpm lint && pnpm typecheck && pnpm test && pnpm build
```

Expected: all green (no removed export is imported here).

- [ ] **Step 3: Commit**

```bash
git add frontend/package.json frontend/pnpm-lock.yaml
git commit -m "chore(frontend): bump @infra/ui to v0.9.0"
```

---

### Task 2: i18n keys for the shell

**Files:**
- Modify: `frontend/src/i18n/en.ts`
- Modify: `frontend/src/i18n/de.ts`

**Interfaces:**
- Produces keys Tasks 3-4 consume: `header.sign_out`, `page.title`,
  `page.caption`. (Nextext's header keys use the `header.` prefix, not
  `appheader.` — match the existing `header.home` / `header.theme_*`.)

- [ ] **Step 1: Add to `frontend/src/i18n/en.ts`** next to the existing
  `header.*` keys, plus a new page block:

```ts
  'header.sign_out': 'Sign out',

  // page
  'page.title': 'Transcribe',
  'page.caption': 'Audio and video transcription, translation, and analysis',
```

- [ ] **Step 2: Add to `frontend/src/i18n/de.ts`** in the same positions:

```ts
  'header.sign_out': 'Abmelden',

  // page
  'page.title': 'Transkribieren',
  'page.caption': 'Audio- und Video-Transkription, Übersetzung und Analyse',
```

- [ ] **Step 3: Run the parity test**

Run: `cd frontend && pnpm test src/i18n/i18n.test.ts`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/i18n/en.ts frontend/src/i18n/de.ts
git commit -m "feat(frontend): i18n keys for AppShell page header and sign-out"
```

---

### Task 3: Shell swap — `AppShell` in `Shell.tsx`, StatusBar strip removed

**Files:**
- Modify: `frontend/src/components/layout/Shell.tsx`
- Modify: `frontend/src/components/layout/Shell.test.tsx`

**Interfaces:**
- Consumes: `AppShell { title, version?, user?, homeLabel?, themeLabels?,
  signOutLabel?, children }`, `PageHeader` NOT here (it goes in Home,
  Task 4); i18n keys from Task 2.
- Produces: canvas children wrapped in `<div className="flex h-full
  min-h-0 flex-col p-8">` — Task 4's Home relies on this wrapper; the
  StatusBar is no longer rendered by the shell (Task 4 relocates it; both
  land before the branch ships, and the `StatusBar` component itself is
  untouched).

- [ ] **Step 1: Update `Shell.test.tsx` first.** Keep the mocks, `stubJobs`,
  and `mountShell` helpers exactly as they are. Replace the `describe`
  block's tests:
  - "renders exactly one header row…" — unchanged (still valid).
  - "renders the fetched version…" — unchanged.
  - "opens exactly one owner-multiplexed job stream" — unchanged.
  - DELETE "surfaces the job status bar below the header" (the shell no
    longer renders StatusBar; Task 4 adds the equivalent assertion to the
    Home suite).
  - REPLACE "pins the single header…" with a fixed-chrome assertion:

```tsx
  it('renders the chrome header and the scrolling canvas main', () => {
    stubJobs([])
    mountShell()
    expect(screen.getByRole('banner')).toBeInTheDocument()
    expect(screen.getByRole('main')).toHaveTextContent('page-body')
  })
```

  - REPLACE the two identity tests' assertions — the user renders as the
    menu button now:

```tsx
  it('shows the display name in the header when whoami resolves', async () => {
    stubJobs([], { username: 'alice', display_name: 'Alice Example' })
    mountShell()
    expect(await screen.findByRole('button', { name: /Alice Example/ })).toBeInTheDocument()
  })

  it('falls back to the username when no display name is set', async () => {
    stubJobs([], { username: 'alice', display_name: null })
    mountShell()
    expect(await screen.findByRole('button', { name: /alice/ })).toBeInTheDocument()
  })
```

  - "omits the identity slot when whoami fails" — unchanged.

- [ ] **Step 2: Run to verify the changed tests fail against the old shell**

Run: `cd frontend && pnpm test src/components/layout/Shell.test.tsx`
Expected: FAIL on the two identity-as-button tests (old shell renders
plain text).

- [ ] **Step 3: Rewrite `Shell.tsx`** — same hooks and data flow, new
  chrome. Full replacement of the component body (imports: drop
  `AppHeader` and `StatusBar`, add `AppShell`):

```tsx
export function Shell({ children }: { children: ReactNode }) {
  useOwnerJobStream()
  const t = useT()
  const { data } = useQuery({
    queryKey: ['version'],
    queryFn: getVersion,
    staleTime: Infinity,
  })
  const { data: whoami } = useWhoami()
  return (
    <AppShell
      title="Nextext"
      version={data?.version ? `v${data.version}` : undefined}
      user={whoami?.display_name ?? whoami?.username}
      homeLabel={t('header.home')}
      themeLabels={{
        system: t('header.theme_system'),
        light: t('header.theme_light'),
        dark: t('header.theme_dark'),
      }}
      signOutLabel={t('header.sign_out')}
    >
      <div className="flex h-full min-h-0 flex-col p-8">{children}</div>
    </AppShell>
  )
}
```

Update the component docstring: the StatusBar note is obsolete — the jobs
column (Home) now hosts it; the SSE-stream paragraph stays accurate.

- [ ] **Step 4: Run the suite for this file, then the full suite**

Run: `cd frontend && pnpm test src/components/layout/Shell.test.tsx && pnpm test`
Expected: Shell suite PASS. Full suite: the StatusBar integration test
(`StatusBar.integration.test.tsx`) may fail if it mounts `Shell` expecting
the strip — if so, note it and fix it in Task 4 where StatusBar gets its
new home (do not delete the integration test).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/layout/Shell.tsx frontend/src/components/layout/Shell.test.tsx
git commit -m "feat(frontend): adopt AppShell chrome; retire the header StatusBar strip"
```

---

### Task 4: Two-column Home — upload card beside the jobs list

**Files:**
- Modify: `frontend/src/routes/Home.tsx`
- Modify: `frontend/src/components/layout/StatusBar.integration.test.tsx`
  (re-point at Home if it asserted the shell strip)
- Test: extend the Home coverage inside the existing route/StatusBar test
  files — do not create a new test file if an existing one already mounts
  `Home`.

**Interfaces:**
- Consumes: the shell canvas wrapper from Task 3; `PageHeader`, `Card`
  from `@infra/ui`; i18n keys from Task 2; existing `StatusBar`,
  `UploadForm`, `BatchProgress` components unchanged.
- Produces: no API changes.

- [ ] **Step 1: Rewrite the `Home` returned JSX** (hooks/error derivation
  above it stay identical):

```tsx
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <PageHeader title={t('page.title')} caption={t('page.caption')} />
      <div className="grid min-h-0 flex-1 items-start gap-6 lg:grid-cols-[minmax(20rem,26rem)_1fr]">
        <section className="space-y-4">
          <Card title={t('home.new_job')}>
            <UploadForm
              pending={submit.isPending}
              onRun={(files, options) => submit.mutate({ files, options })}
            />
          </Card>
          {submitErrorDescriptor && (
            <Banner variant="danger">
              {t('errors.upload_failed')} {t(submitErrorDescriptor.key, submitErrorDescriptor.vars)}
            </Banner>
          )}
          {fileErrors.length > 0 && <Banner variant="danger">{fileErrors.join('\n')}</Banner>}
        </section>
        <section className="min-w-0 space-y-3">
          <div className="flex items-center justify-between gap-4">
            <h2 className="text-base font-semibold">{t('home.jobs_heading')}</h2>
            <StatusBar />
          </div>
          <BatchProgress />
        </section>
      </div>
    </div>
  )
```

Imports change accordingly: add `PageHeader, Card` to the `@infra/ui`
import, add `StatusBar` from `../components/layout/StatusBar`. The
`h2` headings for "New job" disappear (the Card title carries it); the
jobs heading stays as the column header with the StatusBar beside it
(StatusBar renders `null` with no jobs — no reserved empty strip
anywhere anymore).

- [ ] **Step 2: Re-point displaced tests.** If
  `StatusBar.integration.test.tsx` mounted `Shell` to find the status
  bar, mount `Home` (inside the same providers/mocks) instead and keep
  every behavioral assertion (counts, progress fraction). Add one Home
  assertion if no test covers it yet: with one completed job stubbed,
  `await screen.findByText('1 finished')` resolves.

- [ ] **Step 3: Run the affected suites, then everything**

Run: `cd frontend && pnpm test src/components/layout/ src/routes/ 2>/dev/null || cd frontend && pnpm test`
Expected: all green.

- [ ] **Step 4: Full gates, commit**

Run: `cd frontend && pnpm lint && pnpm typecheck && pnpm test && pnpm build`

```bash
git add frontend/src/routes/Home.tsx frontend/src/components/layout/StatusBar.integration.test.tsx
git commit -m "feat(frontend): two-column home — upload card beside jobs, StatusBar in the jobs column"
```

---

### Task 5: Primitive adoption — Select/Button/Spinner/Card replace home-grown widgets

**Files:**
- Modify: `frontend/src/components/upload/UploadForm.tsx:80-121`
- Modify: `frontend/src/components/jobs/JobCard.tsx:60-114`
- Delete: `frontend/src/components/common/Spinner.tsx` (after re-pointing
  its importers to `@infra/ui`'s `Spinner` — find them with
  `grep -rn "common/Spinner" frontend/src`)
- Test: existing suites (`UploadForm.test.tsx`, `JobCard.test.tsx`, and
  any suite that rendered the local spinner) are the spec — they must
  pass; adjust only assertions that encoded the old DOM (e.g. a class),
  never behavior.

**Interfaces:**
- Consumes: `Select`, `Button`, `Spinner`, `Card` from `@infra/ui`
  (`Button` variants `primary|secondary|ghost|danger`, sizes `sm|md`;
  `Spinner` takes an optional `label`).
- Produces: no API changes — `UploadFormProps` and `JobCard` props stay
  identical.

- [ ] **Step 1: UploadForm — primitives for the three selects and the run
  button.** Add `Select, Button` to the `@infra/ui` import. Replace each
  raw `<select className="w-full rounded border border-border bg-muted px-2 py-1" …>`
  with `<Select className="w-full" …>` (children unchanged), e.g.:

```tsx
        <label className="space-y-1">
          <span className="text-sm text-muted-foreground">{t('options.task')}</span>
          <Select className="w-full" value={task} onChange={(e) => setTask(e.target.value as Task)}>
            <option value="transcribe">{t('options.task_transcribe')}</option>
            <option value="translate">{t('options.task_translate')}</option>
          </Select>
        </label>
```

(same pattern for the source-language and target-language selects), and
replace the raw run button with:

```tsx
      <Button type="button" disabled={!canRun} onClick={run}>
        {pending ? t('upload.submitting') : t('upload.run')}
      </Button>
```

- [ ] **Step 2: JobCard — Card surface and Button actions.** Import
  `Button, Card` from `@infra/ui`. The outer
  `<div className="rounded-lg border border-border p-4">` becomes
  `<Card>` (closing tag too — v0.9.0 Card is the tile: opaque fill, same
  radius/border/padding). The show/hide-results and remove text-buttons
  become ghost buttons, keeping their exact labels and handlers:

```tsx
          {p.status === 'completed' && (
            <Button variant="ghost" size="sm" type="button" onClick={() => setShowResults((v) => !v)}>
              {showResults ? t('jobs.hide_results') : t('jobs.show_results')}
            </Button>
          )}
          <span className="text-sm text-muted-foreground">{t(LABEL_KEY[p.status])}</span>
          <Button
            variant="ghost"
            size="sm"
            type="button"
            disabled={del.isPending}
            onClick={() => del.mutate(job.job_id)}
          >
            {del.isPending ? t('jobs.removing') : t('common.remove')}
          </Button>
```

The progress bar's inner track `bg-muted` sits on the Card's own
`bg-muted` fill now — change the track div's class from
`rounded bg-muted` to `rounded bg-background` (both occurrences of a
muted track inside the card, i.e. `JobCard.tsx:85`) so the bar stays
visible on the tile.

- [ ] **Step 3: Replace the local Spinner.** For every file
  `grep -rn "common/Spinner" frontend/src` reports, change the import to
  `import { Spinner } from '@infra/ui'` (call sites keep their props —
  the shared Spinner accepts the same optional `label`). Then
  `git rm frontend/src/components/common/Spinner.tsx`.

- [ ] **Step 4: Run the affected suites, then full gates**

Run: `cd frontend && pnpm test src/components/upload/ src/components/jobs/ && pnpm lint && pnpm typecheck && pnpm test && pnpm build`
Expected: all green; fix only DOM-encoding assertions (a `getByRole`
('button') query now matching a real `<button>` still passes — most
suites should be untouched).

- [ ] **Step 5: Commit**

```bash
git add -A frontend/src/components
git commit -m "feat(frontend): adopt shared Select/Button/Spinner/Card primitives"
```

---

### Task 6: Release bump + verify

**Files:**
- Modify: `pyproject.toml:7` (`version = "1.2.0"` → `"1.3.0"`)

- [ ] **Step 1: Bump the declared version** — `pyproject.toml`
  `[project].version` → `1.3.0` (tag auto-mints on merge).

- [ ] **Step 2: The full pre-push gate**

```bash
make verify
```

Expected: green (pre-commit ruff+pyrefly on the untouched backend +
frontend `pnpm lint`/`pnpm build`). Also `cd frontend && pnpm test` once
more — all green.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: v1.3.0"
```

- [ ] **Step 4: STOP — do not push.** The controller opens the PR.
