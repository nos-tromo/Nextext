# Tab-delimited `.txt` transcript export

**Date:** 2026-07-09
**Status:** Implemented — see Revision below

## Revision (2026-07-09): human-readable block layout supersedes TSV

After delivery, the customer-facing `.txt` format was changed from a
tab-delimited table (TSV, described in the sections below) to a readable
per-segment block layout. Only the `.txt` output changed; CSV/XLSX stay
tabular, the two-file transcript/translation split is unchanged, and the
backend/CLI/ZIP/frontend surfaces are the same. Each segment renders as:

```
{start} - {end} (SPEAKER_00)
<segment text>

```

- The `(SPEAKER_00)` tag is appended **only when the row carries a speaker
  label** (a diarized job); real pyannote labels pass through verbatim. An
  undiarized transcript (no `speaker` column) has **no** `(...)` tag at all.
- Segments are separated by a blank line; an empty transcript renders to an
  empty file. Rendering lives in `nextext.pipeline._render_transcript_block`,
  consumed by the same `transcript_txt_exports` seam.

The TSV details in the sections below are retained as the original design record.

## Revision (2026-07-10): banner-fenced segment headers

The timestamp/speaker header is now fenced above and below by a 40-char `=`
rule and bracketed, so it stays distinguishable from the text body. Without
this, a blank line *inside* a segment (a paragraph break in the transcribed
text) reads like the blank line that separates segments. One segment renders
as:

```
========================================
[{start} - {end}]  {speaker}
========================================
{text, which may span multiple paragraphs}
```

- `  {speaker}` (two spaces, no parentheses) is appended only when the row
  carries a speaker label; an undiarized transcript has no speaker suffix (the
  header is just `[{start} - {end}]`).
- Segments are still separated by a single blank line; the block now ends with
  one trailing newline (the prior block layout ended with a blank line).
- Rendering still lives in `nextext.pipeline._render_transcript_block`, the sole
  seam behind `transcript_txt_exports`, so the backend artifacts
  (`transcript.txt`/`translation.txt`, per-job and batch ZIP) and the CLI
  on-disk output all change together. CSV/XLSX/JSONL are unaffected.

## Goal

Expand the file output for transcription and translation tasks with a
tab-delimited `.txt` export of the transcript, alongside the existing
CSV/XLSX. For **translate** tasks the txt is split into two files — one for
the transcript (source text) and one for the translation — because a single
wide side-by-side table is hard for a customer to read.

This is **additive**: CSV and XLSX outputs are unchanged. It applies to the
transcript output only (the artifact produced by transcription/translation),
not to the word-count, entity, summary, or hate-speech exports.

## Scope

Both output surfaces gain the export (confirmed with the user):

- **Backend HTTP artifacts** — what the frontend/customer downloads, including
  the ZIP bundle.
- **Local `nextext-cli` on-disk output** — `FileProcessor.write_*`.

The React frontend gains matching TXT download button(s).

## Background: the transcript DataFrame

The transcript is a `pd.DataFrame` with columns:

- `start`, `end` — always present.
- `speaker` — optional; present only when diarization ran (`max speakers > 1`).
- `text` — always present; the original transcribed (source-language) text.
- `translation` — present only after a `translate` task
  (`translation_pipeline`), holding the target-language text.

`effective_text_column(df)` already returns `"translation"` when that column
exists, else `"text"`. The txt split maps directly onto the `text` vs
`translation` columns.

## Design

### 1. Shared split helper (single source of truth)

To avoid duplicating the split logic across the backend and the CLI, add one
helper to `nextext/pipeline.py`:

```python
def transcript_txt_frames(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    ...
```

Behavior:

- Timing/speaker columns carried into every returned frame: `start`, `end`,
  and `speaker` **iff** present.
- **Transcribe-only** frame (no `translation` column) → returns a single pair
  `[("transcript", <timing cols + text>)]`.
- **Translated** frame (`translation` column present) → returns two pairs:
  - `("transcript", <timing cols + text>)`
  - `("translation", <timing cols + translation>)`

The label strings (`"transcript"`, `"translation"`) are semantic; each surface
applies its own file-naming. An empty input frame yields header-only frames
(consistent with how the CSV export already handles the no-speech case).

Serialization is plain TSV with a header row, reusing pandas:

```python
frame.to_csv(sep="\t", index=False)
```

The header row is kept deliberately — "the content that's written into
CSV/XLSX" includes the header, and `start / end / [speaker] / text` reads
cleanly as a TSV.

### 2. Backend artifacts (`nextext/api/artifacts.py`)

- New artifact **`transcript.txt`** — available whenever the transcript
  DataFrame is non-empty. Content = the `("transcript", …)` frame only (source
  `text`, never the wide combined view).
- New artifact **`translation.txt`** — available **only** when a `translation`
  column is present; returns `None` (→ 404) otherwise.
- `_render_archive_members` (ZIP bundle) gains `{stem}_transcript.txt` always,
  and `{stem}_translation.txt` when translated. These flow through the existing
  per-job and batch ZIP builders and the archive-members cache unchanged.
- Content-type: `text/plain; charset=utf-8` (the existing `_TEXT_PLAIN`,
  matching `summary.txt`).
- Both names added to `SUPPORTED_ARTIFACTS`.

**CSV/XLSX are unchanged.** For a translate task they keep `text` and
`translation` side by side in one file; only the txt is split.

### 3. CLI (`nextext/core/processing.py` + `nextext/cli.py`)

- Add a focused method `FileProcessor.write_transcript_output(df)` that writes:
  - the combined `{filename}_transcript.csv` + `{filename}_transcript.xlsx`
    (today's DataFrame behavior), and
  - the split `.txt` file(s) via `transcript_txt_frames`:
    `{filename}_transcript.txt` and, when translated,
    `{filename}_translation.txt`.
- The two transcript write sites in `cli.py` switch to this method:
  - the no-speech guard (`_run_main`, currently `write_file_output(transcript_df, "transcript")`), and
  - the final transcript write near the end of `_run_main`.
- The generic `write_file_output` is left untouched, so word-count, entity,
  summary, and hate-speech exports keep their current behavior. This honors the
  "transcription and translation tasks" scope.

### 4. Frontend (`frontend/src/components/results/TranscriptTab.tsx`)

The download button set (`DownloadButtons` items) becomes conditional on
`transcriptHasTranslation(segments)`:

- No translation:
  `CSV · XLSX · TXT · JSONL`
  (`{ name: 'transcript.txt', label: 'TXT', fileName: '${stem}_transcript.txt' }`)
- With translation:
  `CSV · XLSX · Transcript TXT · Translation TXT · JSONL`
  (adds `{ name: 'translation.txt', label: 'Translation TXT', fileName: '${stem}_translation.txt' }`
  and relabels the transcript button to `Transcript TXT`).

No changes to the on-screen transcript table itself.

## Data flow

```
transcript DataFrame  ──►  transcript_txt_frames(df)  ──►  [("transcript", …), ("translation", …)?]
                                                             │
   backend:  render_artifact("transcript.txt" | "translation.txt")  ─► TSV bytes ─► HTTP / ZIP
   cli:      FileProcessor.write_transcript_output(df)               ─► TSV bytes ─► disk
```

## Error handling / edge cases

- **Empty / no-speech transcript:** `transcript_txt_frames` returns header-only
  frames. Backend `transcript.txt` follows the existing `_missing_dataframe`
  guard (empty DataFrame → 404), matching `transcript.csv`. The CLI writes a
  header-only txt, matching its header-only csv today.
- **`translation.txt` requested for a transcribe job:** no `translation`
  column → artifact returns `None` → 404. Frontend never shows the button in
  that case.
- **Speaker column:** included in both split files whenever present; absent
  otherwise. No special-casing beyond the column check.

## Testing

- **Backend** (`tests/test_api/test_artifacts.py`): `transcript.txt` content is
  tab-delimited with a header and the expected columns; `translation.txt`
  present for a translate job and 404 for a transcribe job; both appear in the
  per-job ZIP (`{stem}_transcript.txt`, `{stem}_translation.txt`).
- **CLI** (new `tests/test_processing.py`): `write_transcript_output` writes the
  combined csv/xlsx plus the split txt file(s); the translate case produces two
  txt files with the right per-file columns; the transcribe case produces one.
- **Frontend**: extend the TranscriptTab/download tests to assert the TXT
  button(s) render (single vs. split) based on presence of a translation.

## Out of scope

- No change to CSV/XLSX layout or to the docint JSONL.
- No change to word-count / entity / summary / hate-speech exports.
- No new environment variables or configuration.
