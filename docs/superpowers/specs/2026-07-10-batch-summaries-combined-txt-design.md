# Combined summaries file in the batch ZIP

**Date:** 2026-07-10
**Status:** Approved — implementing

## Goal

When the operator downloads the **full batch** as a ZIP (`nextext_batch.zip`,
built by `build_batch_archive()` in `nextext/api/artifacts.py`), add a single
top-level file `batch_summaries.txt` that lists the summary of every file in
the batch — each entry carrying the file name and its summary text.

This is **additive**: the existing per-job `{stem}/{stem}_summary.txt` files
stay exactly as they are (confirmed with the user — keep both). The combined
file is a convenience index at the archive root so a reviewer can read every
summary in one place without opening each job folder.

## Scope

- **Batch ZIP only** — the `archive.zip` batch artifact
  (`GET /api/v1/jobs/batch/archive.zip`, downloaded as `nextext_batch.zip`).
- **Not** the per-job `archive.zip`, the batch `docint.jsonl`, the frontend
  (the "Full batch (ZIP)" menu item already downloads this archive), or any
  env/config.

## Background

`build_batch_archive(states)` bundles every completed job's rendered members
(`_render_archive_members`) under a collision-free per-job folder. A job that
produced a summary already contributes `{stem}_summary.txt` inside its folder
(`_render_archive_members`: `result["summary"]` → `{stem}_summary.txt` when the
summary is a non-empty string). The combined file reuses that same
`result["summary"]` value.

## Design

### 1. Combined-summaries helper (`nextext/api/artifacts.py`)

New pure helper:

```python
def _build_batch_summaries_txt(states: list[JobState]) -> bytes:
    ...
```

Behavior:

- Iterate `states` in order. For each job whose `result["summary"]` is a
  non-empty string (`isinstance(summary, str) and summary.strip()`), emit one
  banner-delimited block; skip every other job silently.
- Block format (banner headers — the layout the user chose), header is the
  **full upload file name with extension** (`state.file_name`):

  ```
  ========================================
  interview_2024.mp4
  ========================================

  <full summary text for this file>
  ```

  The rule is 40 `=` characters. Blocks are separated by one blank line.
- Returns UTF-8 bytes, or `b""` when **no** job in the batch has a summary.

### 2. Wire into `build_batch_archive`

Prepend the combined file as the first top-level member when the helper
returns non-empty:

```python
summaries = _build_batch_summaries_txt(states)
if summaries:
    named.append(("batch_summaries.txt", summaries))
# ... then the existing per-job folder members ...
```

- Placed at the archive **root** (no folder prefix), before the per-job
  folders, so it reads as a manifest.
- The name constant lives beside the function
  (`_BATCH_SUMMARIES_NAME = "batch_summaries.txt"`).

## Scope rule ("if summary is activated")

Evaluated **per file**, from `result["summary"]`:

- ≥1 job has a non-empty summary → `batch_summaries.txt` is added, listing only
  those jobs (in ZIP order), each under its own banner.
- **No** job has a summary → the file is **not** added; the ZIP is byte-for-byte
  identical to today.

Because a non-empty `result["summary"]` also makes `_render_archive_members`
non-empty (it adds `{stem}_summary.txt`), a summary-bearing job always
contributes a folder too — the helper and the per-job render never disagree.

## Data flow

```
states ─► _build_batch_summaries_txt(states) ─► banner-delimited UTF-8 bytes (or b"")
                                                   │
build_batch_archive: prepend ("batch_summaries.txt", bytes) ─► _zip_members ─► nextext_batch.zip
```

## Ordering

The route passes `list_for_owner(owner)`-ordered completed jobs (newest first),
the same order used for the per-job folders. The combined file lists entries in
that same order, so the manifest matches the folder order in the ZIP.

## Error handling / edge cases

- **Duplicate upload names:** each job still gets its own banner entry keyed by
  `state.file_name`; identical headers are acceptable and mirror the two
  (disambiguated `stem` / `stem_2`) folders below.
- **Job with a summary but otherwise empty:** impossible to reach an
  inconsistent state — the same non-empty summary guarantees a non-empty
  `_render_archive_members`, so the job's folder exists.
- **Empty batch / all jobs summary-less:** helper returns `b""`, member is not
  added, and the existing `if not named: return b""` guard is unchanged.

## Testing (`tests/test_archive_cache.py`)

- Extend `_completed_job` (or add a summary-bearing variant) so a job can carry
  `result["summary"]`.
- **Present:** a 2-job batch where both jobs have summaries → the batch ZIP has
  a **root** `batch_summaries.txt` containing both file names and both summary
  texts, in job order.
- **Absent:** a batch whose jobs have no summary → no `batch_summaries.txt`
  member (asserts the additive/no-op contract).
- **Partial:** a batch mixing a summary-bearing job with a summary-less one →
  only the summary-bearing file appears in `batch_summaries.txt`, and the
  summary-less job still gets its folder.
- `test_batch_member_bytes_match_per_job` is unaffected: its `_completed_job`
  has `summarization: False` and no `result["summary"]`, so no combined file is
  added for that single job.

## Out of scope

- No change to the per-job `archive.zip`, the per-job `{stem}_summary.txt`, the
  batch `docint.jsonl`, the frontend, or the CLI.
- No new environment variables or configuration.
