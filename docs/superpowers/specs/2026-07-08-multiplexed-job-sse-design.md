# Multiplexed owner-scoped job SSE stream

**Date:** 2026-07-08
**Status:** Approved (design)

## Problem

The frontend opens one long-lived SSE connection per `JobCard`
(`useJobStream`). A batch of *N* files renders *N* cards, each opening a
stream. The SPA is served over HTTP/1.1 (`listen 80;`, no `http2`), and
browsers cap concurrent connections at **6 per host**. With a large batch
(observed: 42 files) all 6 connection slots are permanently consumed by SSE
streams — and by the *newest* cards (top of the newest-first list), which are
queued and idle. Every other request starves: the remaining streams never
connect, and `GET /jobs` never refetches. The UI freezes on the post-submit
snapshot, so the header progress bar reads **0%** for the whole run even though
the backend processes every file normally.

Empirical confirmation: exactly 6 ESTABLISHED browser→nginx connections and 6
nginx→backend SSE upstreams; the browser issued no request for ~16 minutes
while the backend completed 24/43 jobs. SSE itself works — an isolated `curl`
replays and closes cleanly.

## Goal

Use **one** SSE connection per browser (owner) that multiplexes events for all
of the caller's jobs, so batch size never approaches the connection cap. Also
make the header progress bar reflect **overall batch completion** rather than
one job's intra-progress.

## Backend

### New endpoint `GET /jobs/events` (owner-scoped)

Declared **before** `/{job_id}` in `routes/jobs.py` so the literal `events`
segment matches ahead of the job-id pattern (same precedent as
`/jobs/batch/{name}`). Returns a `text/event-stream` carrying every event for
the caller's jobs, including jobs created *after* the stream opens. Same
response headers as the per-job endpoint (`Cache-Control: no-cache`,
`X-Accel-Buffering: no`).

### `JobManager` owner-level fan-out

Add owner subscribers alongside the existing per-job ones:

- `self._owner_subscribers: dict[str, list[asyncio.Queue[bytes | None]]]`.
- `subscribe_owner(owner_id) -> AsyncIterator[bytes]`: under `self._lock`,
  atomically (a) replay every owned job's `event_history` into a fresh queue
  via `put_nowait` (no `await` between snapshot and registration, so no frame
  can slip past), then (b) register the queue. Live-tail with the existing
  15 s ping keepalive. On generator close (client disconnect) remove the queue
  and drop the owner key when empty.
- `_dispatch_event` also forwards each frame to
  `self._owner_subscribers[state.owner_id]`. It forwards the terminal **frame**
  (so the client sees `job_completed`/`job_failed`) but **not** the `None`
  sentinel — one job finishing must not close the owner stream. The stream
  closes only on disconnect.

The existing per-job `subscribe` / `state.subscribers` path is unchanged, so
`GET /jobs/{id}/events` keeps working (retained; unused by the batch UI).

### Event contract (additive, backward-compatible)

Every event becomes self-identifying with a `job_id` so a multiplexed consumer
can route by `data.job_id`. **Implementation note:** rather than editing each
`stage_started` / `stage_completed` call site (and the test stub), `job_id` is
injected once in the worker's `_push` closure (which already has `state` in
scope), so every frame — stage and terminal — is tagged at a single choke
point. `job_completed`/`job_failed` already carried it; re-adding is idempotent.
Update `StageEvent` (schemas.py) and the mirrored TS `StageStartedEvent` /
`StageCompletedEvent`. Old per-job consumers ignore the new field.

## Frontend

- **`useOwnerJobStream()`** — new hook mounted **once** in the app `Shell`
  (layout root), opening exactly one `streamSse('/jobs/events')` for the
  session. It folds each frame (routed by `data.job_id`) through the existing
  `reduceJobEvent` into per-job `JobProgress`, keeping a local
  `Map<jobId, JobProgress>` for prior state, and publishes each into the
  existing `useJobProgressStore` (keyed by `job_id`). Reconnect/replay logic
  mirrors the current `useJobStream` (reducer is idempotent). When a job first
  reaches terminal, invalidate `['jobs']` so the list (completed count, batch
  download) refreshes.
- **`JobCard`** becomes presentational: reads
  `useJobProgressStore(s => s.byId[jobId])` with a fallback to
  `initialJobProgress(seedOf(job))` for queued jobs that have no events yet. It
  no longer owns a stream. The now-dead `useJobStream` hook is removed.
- **`StatusBar` / `summarizeJobs`** — change the header bar to show **batch
  progress = terminal jobs / total** (completed + failed) / total, so
  1/10 done → 10% and the bar reaches 100% when nothing is left running. The
  existing per-bucket badges (processing / queued / finished / failed) stay.

Net effect: *N* jobs → 1 connection. The connection cap is never approached;
`GET /jobs` refetches and artifact downloads are no longer starved.

## Testing

**Backend** (asyncio, no network):
- `subscribe_owner` multiplexes events for 2+ jobs, each tagged with `job_id`,
  and stays open across one job's completion.
- Replays history for jobs that exist at connect time.
- Owner-scoped: another owner's job events never appear.
- A job created *after* connect streams its events to the open owner queue.
- `/jobs/events` routes to the owner stream (not `/{job_id}` with id `events`).
- `stage_started`/`stage_completed` frames include `job_id`.

> **Test-harness note.** The owner stream never closes on its own, which
> deadlocks *both* in-process test transports — the sync `TestClient` portal and
> `httpx.AsyncClient` + `ASGITransport` (which buffers the whole body). So the
> fan-out behaviour is verified by driving `subscribe_owner` directly as an async
> generator (bounded reads), and the *route* is verified with a bounded stub
> `subscribe_owner` so the response closes. Never assert on a real owner-stream
> body through the sync `TestClient`.

**Frontend** (vitest):
- `useOwnerJobStream` routes multiplexed frames to the store by `job_id`,
  reduces correctly, and invalidates `['jobs']` on terminal.
- `JobCard` renders from the store with a list-status fallback.
- Banner shows terminal/total percentage.
- Only one stream is opened regardless of job count.

## Out of scope

- Enabling HTTP/2 (infra change; multiplexing removes the need).
- A per-job "job_deleted" event (the list refetch already handles removal).
- Removing the per-job `/jobs/{id}/events` backend endpoint.
