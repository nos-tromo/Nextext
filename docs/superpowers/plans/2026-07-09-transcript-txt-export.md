# Tab-delimited `.txt` Transcript Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tab-delimited `.txt` export of the transcript across backend artifacts, the ZIP bundle, the CLI, and the frontend download buttons — split into two files (transcript + translation) for translate tasks.

**Architecture:** One shared helper in `nextext/pipeline.py` (`transcript_txt_exports`) owns the split-and-serialize logic and is consumed by both the backend artifact renderer and the CLI file writer, so the tab delimiter and column split live in exactly one place. The frontend gains conditional TXT download button(s). CSV/XLSX outputs are untouched — this is purely additive.

**Tech Stack:** Python 3.12, pandas, FastAPI (backend); React + TypeScript + Vitest + Testing Library (frontend); pytest (backend tests).

## Global Constraints

- Python 3.12 target; all new/modified Python functions need Google-style docstrings (project rule).
- Prefer explicit types and distinct variable names across branches (pyrefly).
- The TXT delimiter is a literal tab (`\t`); TXT files include a header row.
- No new environment variables, no new dependencies, no changes to CSV/XLSX layout or docint JSONL.
- Backend artifact content-type for `.txt` is `text/plain; charset=utf-8` (reuse existing `_TEXT_PLAIN`).
- Frontend package manager is pinned `pnpm@9.12.0`; run frontend commands as `corepack pnpm --dir "$(git rev-parse --show-toplevel)"/frontend <script>` (pnpm is not on PATH).
- Run the full backend suite (`uv run pytest`) and `pre-commit run --all-files` before declaring done (project rule).

## File Structure

- `nextext/pipeline.py` — **Modify.** Add `transcript_txt_exports(df)`; the single source of truth for the split + tab serialization.
- `nextext/api/artifacts.py` — **Modify.** Render `transcript.txt` / `translation.txt`; add both to `SUPPORTED_ARTIFACTS`; add txt members to the ZIP bundle.
- `nextext/core/processing.py` — **Modify.** Add `FileProcessor.write_transcript_output(df)`.
- `nextext/cli.py` — **Modify.** Route the two transcript write sites through `write_transcript_output`.
- `frontend/src/components/results/TranscriptTab.tsx` — **Modify.** Conditional TXT download button(s).
- `tests/test_pipeline.py` — **Modify.** Unit tests for `transcript_txt_exports`.
- `tests/test_api/test_artifacts.py` — **Modify.** Backend artifact + ZIP tests.
- `tests/test_processing.py` — **Create.** CLI writer tests.
- `frontend/src/components/results/TranscriptTab.test.tsx` — **Create.** Download-button tests.

---

### Task 1: Shared split-and-serialize helper

**Files:**
- Modify: `nextext/pipeline.py` (add function near `effective_text_column`, ~line 165)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `transcript_txt_exports(df: pd.DataFrame) -> list[tuple[str, str]]` — returns `[(label, tsv_text), ...]`. Labels are `"transcript"` (always) and `"translation"` (only when a `translation` column is present). Each `tsv_text` is a tab-delimited table with a header row: columns `start`, `end`, `speaker` (only if present), and exactly one of `text` / `translation`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_pipeline.py` (keep existing imports; add `import pandas as pd` and the helper import if not already present):

```python
from nextext.pipeline import transcript_txt_exports


def test_transcript_txt_exports_transcribe_returns_single_file() -> None:
    """A transcribe-only transcript yields one tab-delimited 'transcript' export."""
    df = pd.DataFrame(
        {
            "start": ["00:00:00"],
            "end": ["00:00:02"],
            "speaker": ["S1"],
            "text": ["Hello world."],
        }
    )
    exports = transcript_txt_exports(df)
    assert [label for label, _ in exports] == ["transcript"]
    _, tsv = exports[0]
    lines = tsv.splitlines()
    assert lines[0] == "start\tend\tspeaker\ttext"
    assert lines[1] == "00:00:00\t00:00:02\tS1\tHello world."


def test_transcript_txt_exports_translate_splits_into_two_files() -> None:
    """A translated transcript yields separate 'transcript' and 'translation' exports."""
    df = pd.DataFrame(
        {
            "start": ["00:00:00"],
            "end": ["00:00:02"],
            "speaker": ["S1"],
            "text": ["Hello world."],
            "translation": ["Hallo Welt."],
        }
    )
    exports = dict(transcript_txt_exports(df))
    assert set(exports) == {"transcript", "translation"}
    assert exports["transcript"].splitlines()[0] == "start\tend\tspeaker\ttext"
    assert "Hello world." in exports["transcript"]
    assert "Hallo Welt." not in exports["transcript"]
    assert exports["translation"].splitlines()[0] == "start\tend\tspeaker\ttranslation"
    assert "Hallo Welt." in exports["translation"]
    assert "Hello world." not in exports["translation"]


def test_transcript_txt_exports_omits_speaker_when_absent() -> None:
    """The speaker column is dropped from the header when no speaker data exists."""
    df = pd.DataFrame({"start": ["00:00:00"], "end": ["00:00:02"], "text": ["Hi."]})
    ((_, tsv),) = transcript_txt_exports(df)
    assert tsv.splitlines()[0] == "start\tend\ttext"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py -k transcript_txt_exports -v`
Expected: FAIL with `ImportError: cannot import name 'transcript_txt_exports'`.

- [ ] **Step 3: Implement the helper**

Add to `nextext/pipeline.py` (e.g. immediately after `effective_text_column`):

```python
def transcript_txt_exports(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Split a transcript DataFrame into tab-delimited TXT exports.

    The transcript keeps the original text in ``text`` and, after
    :func:`translation_pipeline`, the translated text in a separate
    ``translation`` column. A single wide table pairing both is hard to read,
    so this returns one export per text column, each carrying the timing (and
    speaker, when present) columns plus exactly one text column:

    - Transcribe-only frame (no ``translation`` column): a single
      ``("transcript", <tsv>)`` pair.
    - Translated frame: two pairs — ``("transcript", <tsv>)`` and
      ``("translation", <tsv>)``.

    Each ``tsv`` value is tab-delimited with a header row.

    Args:
        df (pd.DataFrame): Transcript DataFrame with ``start``/``end``/``text``
            columns, an optional ``speaker`` column, and an optional
            ``translation`` column.

    Returns:
        list[tuple[str, str]]: ``(label, tsv_text)`` pairs, ``"transcript"``
            first and ``"translation"`` second when present.
    """
    base_columns = [column for column in ("start", "end", "speaker") if column in df.columns]
    exports: list[tuple[str, str]] = [
        ("transcript", df[[*base_columns, "text"]].to_csv(sep="\t", index=False))
    ]
    if "translation" in df.columns:
        exports.append(
            ("translation", df[[*base_columns, "translation"]].to_csv(sep="\t", index=False))
        )
    return exports
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_pipeline.py -k transcript_txt_exports -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add nextext/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): add transcript_txt_exports tab-delimited split helper"
```

---

### Task 2: Backend `.txt` artifacts + ZIP members

**Files:**
- Modify: `nextext/api/artifacts.py` (import at ~line 22; `_render_archive_members` ~lines 112-115; `render_artifact` ~line 294; `SUPPORTED_ARTIFACTS` ~line 357)
- Test: `tests/test_api/test_artifacts.py`

**Interfaces:**
- Consumes: `transcript_txt_exports` (Task 1).
- Produces: artifacts `"transcript.txt"` (always available when the transcript is non-empty) and `"translation.txt"` (only when a `translation` column exists, else 404). ZIP members `{stem}_transcript.txt` and, when translated, `{stem}_translation.txt`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_api/test_artifacts.py`:

```python
def test_transcript_txt_artifact_is_tab_delimited(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """The transcript TXT artifact is a tab-delimited table with a header row."""
    client, _ = stub_app_client
    job_id = _submit_and_wait(
        client,
        {
            "task": "transcribe",
            "trg_lang": "de",
            "speakers": 1,
            "words": False,
            "summarization": False,
            "hate_speech": False,
        },
    )
    response = client.get(f"/api/v1/jobs/{job_id}/artifacts/transcript.txt")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.content.decode("utf-8").splitlines()[0] == "start\tend\tspeaker\ttext"
    df = pd.read_csv(io.BytesIO(response.content), sep="\t")
    assert list(df.columns) == ["start", "end", "speaker", "text"]
    assert len(df) == 2


def test_translation_txt_artifact_404_for_transcribe_task(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """translation.txt is absent (404) when the job did not translate."""
    client, _ = stub_app_client
    job_id = _submit_and_wait(
        client,
        {
            "task": "transcribe",
            "trg_lang": "de",
            "speakers": 1,
            "words": False,
            "summarization": False,
            "hate_speech": False,
        },
    )
    response = client.get(f"/api/v1/jobs/{job_id}/artifacts/translation.txt")
    assert response.status_code == 404


def test_translate_task_splits_transcript_and_translation_txt(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """A translate job exposes transcript.txt (source) and translation.txt (target) separately."""
    client, _ = stub_app_client
    job_id = _submit_and_wait(
        client,
        {
            "task": "translate",
            "trg_lang": "de",
            "speakers": 1,
            "words": False,
            "summarization": False,
            "hate_speech": False,
        },
    )
    transcript = client.get(f"/api/v1/jobs/{job_id}/artifacts/transcript.txt")
    assert transcript.status_code == 200
    transcript_df = pd.read_csv(io.BytesIO(transcript.content), sep="\t")
    assert list(transcript_df.columns) == ["start", "end", "speaker", "text"]
    assert list(transcript_df["text"]) == ["Hello world.", "Second segment."]

    translation = client.get(f"/api/v1/jobs/{job_id}/artifacts/translation.txt")
    assert translation.status_code == 200
    translation_df = pd.read_csv(io.BytesIO(translation.content), sep="\t")
    assert list(translation_df.columns) == ["start", "end", "speaker", "translation"]
    assert list(translation_df["translation"]) == ["Hallo Welt.", "Zweites Segment."]


def test_archive_zip_contains_txt_exports(
    stub_app_client: tuple[TestClient, JobManager],
) -> None:
    """The per-job ZIP bundles both split TXT files for a translate task."""
    client, _ = stub_app_client
    job_id = _submit_and_wait(
        client,
        {
            "task": "translate",
            "trg_lang": "de",
            "speakers": 1,
            "words": False,
            "summarization": False,
            "hate_speech": False,
        },
    )
    response = client.get(f"/api/v1/jobs/{job_id}/artifacts/archive.zip")
    assert response.status_code == 200
    names = set(zipfile.ZipFile(io.BytesIO(response.content)).namelist())
    assert any(name.endswith("_transcript.txt") for name in names)
    assert any(name.endswith("_translation.txt") for name in names)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_api/test_artifacts.py -k "txt" -v`
Expected: FAIL — `transcript.txt`/`translation.txt` return 404 (unsupported), ZIP lacks `_transcript.txt`.

- [ ] **Step 3: Implement**

3a. Extend the pipeline import (currently `from nextext.pipeline import normalize_language_code`):

```python
from nextext.pipeline import normalize_language_code, transcript_txt_exports
```

3b. In `_render_archive_members`, extend the transcript block (after the `.xlsx` line):

```python
    transcript = result.get("transcript")
    if isinstance(transcript, pd.DataFrame) and not transcript.empty:
        members[f"{stem}_transcript.csv"] = transcript.to_csv(index=False).encode("utf-8")
        members[f"{stem}_transcript.xlsx"] = _df_to_xlsx(transcript)
        for label, tsv in transcript_txt_exports(transcript):
            members[f"{stem}_{label}.txt"] = tsv.encode("utf-8")
```

3c. In `render_artifact`, add two branches (place them right after the `transcript.xlsx` branch):

```python
    if name == "transcript.txt":
        if _missing_dataframe(state, "transcript"):
            return None
        exports = dict(transcript_txt_exports(result["transcript"]))
        return exports["transcript"].encode("utf-8"), _TEXT_PLAIN
    if name == "translation.txt":
        if _missing_dataframe(state, "transcript"):
            return None
        exports = dict(transcript_txt_exports(result["transcript"]))
        translation_tsv = exports.get("translation")
        if translation_tsv is None:
            return None
        return translation_tsv.encode("utf-8"), _TEXT_PLAIN
```

3d. Add both names to `SUPPORTED_ARTIFACTS` (after `"transcript.xlsx"`):

```python
        "transcript.txt",
        "translation.txt",
```

3e. Update the artifact enumeration in `CLAUDE.md` so the docs don't go stale. Find the `GET /jobs/{id}/artifacts/{name}` line under **HTTP API** and change the leading artifact list from:

```
binary download (transcript.csv/xlsx, summary.txt, wordcounts.csv/xlsx,
```

to:

```
binary download (transcript.csv/xlsx/txt, translation.txt, summary.txt, wordcounts.csv/xlsx,
```

(The "Pipeline flow" bullet already says the backend renders `.txt`, so no change is needed there.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_api/test_artifacts.py -v`
Expected: PASS (existing + 4 new).

- [ ] **Step 5: Commit**

```bash
git add nextext/api/artifacts.py tests/test_api/test_artifacts.py CLAUDE.md
git commit -m "feat(api): serve transcript.txt/translation.txt artifacts and bundle them in the ZIP"
```

---

### Task 3: CLI transcript writer

**Files:**
- Modify: `nextext/core/processing.py` (add method to `FileProcessor`; add import)
- Modify: `nextext/cli.py` (two call sites: the no-speech guard and the final write)
- Test: `tests/test_processing.py` (Create)

**Interfaces:**
- Consumes: `transcript_txt_exports` (Task 1); the existing `FileProcessor.write_file_output`.
- Produces: `FileProcessor.write_transcript_output(df: pd.DataFrame) -> None` — writes the combined `{filename}_transcript.csv` + `.xlsx` plus the split `{filename}_transcript.txt` and (when translated) `{filename}_translation.txt`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_processing.py`:

```python
"""Tests for the CLI-side FileProcessor transcript output."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nextext.core.processing import FileProcessor


def test_write_transcript_output_transcribe_writes_single_txt(tmp_path: Path) -> None:
    """A transcribe transcript writes combined csv/xlsx plus one transcript.txt."""
    processor = FileProcessor(file_path=Path("clip.wav"), output_dir=tmp_path)
    df = pd.DataFrame(
        {
            "start": ["00:00:00"],
            "end": ["00:00:02"],
            "speaker": ["S1"],
            "text": ["Hello world."],
        }
    )
    processor.write_transcript_output(df)
    out = tmp_path / "clip"
    assert (out / "clip_transcript.csv").exists()
    assert (out / "clip_transcript.xlsx").exists()
    txt = out / "clip_transcript.txt"
    assert txt.exists()
    assert txt.read_text(encoding="utf-8").splitlines()[0] == "start\tend\tspeaker\ttext"
    assert not (out / "clip_translation.txt").exists()


def test_write_transcript_output_translate_writes_two_txt(tmp_path: Path) -> None:
    """A translated transcript writes separate transcript.txt and translation.txt."""
    processor = FileProcessor(file_path=Path("clip.wav"), output_dir=tmp_path)
    df = pd.DataFrame(
        {
            "start": ["00:00:00"],
            "end": ["00:00:02"],
            "speaker": ["S1"],
            "text": ["Hello world."],
            "translation": ["Hallo Welt."],
        }
    )
    processor.write_transcript_output(df)
    out = tmp_path / "clip"
    transcript_txt = (out / "clip_transcript.txt").read_text(encoding="utf-8")
    translation_txt = (out / "clip_translation.txt").read_text(encoding="utf-8")
    assert transcript_txt.splitlines()[0] == "start\tend\tspeaker\ttext"
    assert "Hallo Welt." not in transcript_txt
    assert translation_txt.splitlines()[0] == "start\tend\tspeaker\ttranslation"
    assert "Hallo Welt." in translation_txt
    # The combined CSV still carries both columns side by side.
    combined = pd.read_csv(out / "clip_transcript.csv")
    assert list(combined.columns) == ["start", "end", "speaker", "text", "translation"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_processing.py -v`
Expected: FAIL with `AttributeError: 'FileProcessor' object has no attribute 'write_transcript_output'`.

- [ ] **Step 3: Implement**

3a. Add the import at the top of `nextext/core/processing.py` (after the existing imports):

```python
from nextext.pipeline import transcript_txt_exports
```

3b. Add the method to `FileProcessor` (after `write_file_output`):

```python
    def write_transcript_output(self, data: pd.DataFrame) -> None:
        """Write the transcript as combined CSV/XLSX plus split tab-delimited TXT.

        The combined CSV/XLSX keep the original and translated text side by side
        (unchanged behavior). The TXT export is split per text column via
        :func:`nextext.pipeline.transcript_txt_exports` so a reader gets a clean
        ``{filename}_transcript.txt`` and, for a translate task, a separate
        ``{filename}_translation.txt`` instead of one wide table.

        Args:
            data (pd.DataFrame): The transcript DataFrame, optionally translated.
        """
        self.write_file_output(data, "transcript")
        for label, tsv in transcript_txt_exports(data):
            txt_path = (self.output_path / f"{self.filename}_{label}").with_suffix(".txt")
            txt_path.write_text(tsv, encoding="utf-8")
            logger.info("Saved output: {}", txt_path)
```

3c. In `nextext/cli.py`, change the no-speech guard write (currently `file_processor.write_file_output(transcript_df, "transcript")` inside the empty-transcript branch):

```python
            file_processor.write_transcript_output(transcript_df)
            return
```

3d. In `nextext/cli.py`, change the final transcript write (currently `file_processor.write_file_output(transcript_df, "transcript")` near the end of `_run_main`, above the docint block):

```python
    # Save final transcript
    file_processor.write_transcript_output(transcript_df)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_processing.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add nextext/core/processing.py nextext/cli.py tests/test_processing.py
git commit -m "feat(cli): write split tab-delimited transcript/translation txt files"
```

---

### Task 4: Frontend TXT download buttons

**Files:**
- Modify: `frontend/src/components/results/TranscriptTab.tsx` (`DownloadButtons` items, ~lines 57-64)
- Test: `frontend/src/components/results/TranscriptTab.test.tsx` (Create)

**Interfaces:**
- Consumes: existing `transcriptHasTranslation(segments)` from `../../lib/transcriptTable`, `DownloadButtons`.
- Produces: TranscriptTab renders a `TXT` button for transcribe-only transcripts, and `Transcript TXT` + `Translation TXT` buttons when any segment carries a translation.

- [ ] **Step 1: Write the failing tests**

Create `frontend/src/components/results/TranscriptTab.test.tsx`:

```tsx
import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import { TranscriptTab } from './TranscriptTab'
import type { TranscriptSegment } from '../../api/types'

const transcribeSegments: TranscriptSegment[] = [
  { start: '0.00', end: '2.00', speaker: null, text: 'Hello world', translation: null },
]

const translateSegments: TranscriptSegment[] = [
  { start: '0.00', end: '2.00', speaker: null, text: 'Hello world', translation: 'Hallo Welt' },
]

describe('TranscriptTab download buttons', () => {
  it('shows a single TXT button for a transcribe-only transcript', () => {
    render(<TranscriptTab jobId="j1" segments={transcribeSegments} stem="clip" />)
    expect(screen.getByRole('button', { name: 'TXT' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Translation TXT' })).not.toBeInTheDocument()
  })

  it('splits into Transcript TXT and Translation TXT when a translation exists', () => {
    render(<TranscriptTab jobId="j1" segments={translateSegments} stem="clip" />)
    expect(screen.getByRole('button', { name: 'Transcript TXT' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Translation TXT' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'TXT' })).not.toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `corepack pnpm --dir "$(git rev-parse --show-toplevel)"/frontend test -- TranscriptTab`
Expected: FAIL — no `TXT` / `Transcript TXT` / `Translation TXT` buttons exist yet.

- [ ] **Step 3: Implement**

Replace the `DownloadButtons` `items` array in `TranscriptTab.tsx` (the `return`'s `<DownloadButtons .../>`) with a computed list. Build it above the `return`:

```tsx
  const txtItems = hasTranslation
    ? [
        { name: 'transcript.txt', label: 'Transcript TXT', fileName: `${stem}_transcript.txt` },
        { name: 'translation.txt', label: 'Translation TXT', fileName: `${stem}_translation.txt` },
      ]
    : [{ name: 'transcript.txt', label: 'TXT', fileName: `${stem}_transcript.txt` }]
```

Then set the `DownloadButtons` items to interleave the TXT buttons between XLSX and JSONL:

```tsx
      <DownloadButtons
        jobId={jobId}
        items={[
          { name: 'transcript.csv', label: 'CSV', fileName: `${stem}_transcript.csv` },
          { name: 'transcript.xlsx', label: 'XLSX', fileName: `${stem}_transcript.xlsx` },
          ...txtItems,
          { name: 'docint.jsonl', label: 'JSONL', fileName: `${stem}_docint.jsonl` },
        ]}
      />
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `corepack pnpm --dir "$(git rev-parse --show-toplevel)"/frontend test -- TranscriptTab`
Expected: PASS (2 tests).

- [ ] **Step 5: Typecheck + lint the frontend**

Run: `corepack pnpm --dir "$(git rev-parse --show-toplevel)"/frontend typecheck && corepack pnpm --dir "$(git rev-parse --show-toplevel)"/frontend lint`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/results/TranscriptTab.tsx frontend/src/components/results/TranscriptTab.test.tsx
git commit -m "feat(frontend): add TXT / Transcript+Translation TXT download buttons"
```

---

### Task 5: Full verification

**Files:** none (verification only).

- [ ] **Step 1: Run the full backend suite**

Run: `uv run pytest`
Expected: all pass (note: `test_sse` stage-pair test rarely flakes on a cold run — re-run once before treating as a regression).

- [ ] **Step 2: Run the full frontend suite**

Run: `corepack pnpm --dir "$(git rev-parse --show-toplevel)"/frontend test`
Expected: all pass.

- [ ] **Step 3: Run pre-commit**

Run: `pre-commit run --all-files`
Expected: pyrefly + ruff + docstring hooks pass.

- [ ] **Step 4: Fix any failures at the root cause, then re-run until green.**

---

## Notes for the implementer

- **Why the helper returns strings, not DataFrames:** centralizing `to_csv(sep="\t")` in one place keeps the tab delimiter (a spec requirement) from drifting between the backend and CLI consumers.
- **CSV/XLSX unchanged:** for a translate task the combined `transcript.csv`/`.xlsx` keep `text` and `translation` side by side. Only the TXT export splits. Do not alter the CSV/XLSX branches.
- **Empty transcript:** the backend guards with `_missing_dataframe` (empty → 404), matching `transcript.csv`. The CLI writes a header-only txt for the no-speech case, matching its header-only csv.
- **`text` column always present:** `ExternalWhisperTranscriber` always emits a `text` column (even for an empty transcript), so `transcript_txt_exports` can index it unconditionally.
