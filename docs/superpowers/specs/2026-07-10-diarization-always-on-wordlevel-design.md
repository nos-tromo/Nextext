# Diarization: always-on auto-detect, canonical labels, word-level precision — design

**Date:** 2026-07-10
**Status:** Approved
**Branch:** `feature/diarization-always-on-wordlevel` (Nextext) + a follow-on branch in `vllm-service`

## Problem

Three client-reported diarization defects:

1. **Imprecise in dialogue.** A single Whisper segment can span a speaker
   change (fast back-and-forth with no clear pause). The current alignment
   assigns **one** speaker per Whisper segment by maximum temporal overlap
   (`nextext/core/diarization.py::assign_speakers_by_overlap`), so the whole
   segment — both voices — gets one label.
2. **Arbitrary, non-contiguous labels.** The `/diarize` service faithfully
   returns pyannote's raw labels (`SPEAKER_00`, `SPEAKER_02`, …), which are
   unordered and gap-y: pyannote may hand `SPEAKER_02` to the first voice and
   `SPEAKER_00` to the second. Operators expect `Speaker 1` = first voice heard,
   contiguous.
3. **Speaker count is unknown up front.** Diarization is gated behind a per-job
   "Max speakers" control (`speakers: int`, 1–10, default `1` = off) that
   forces the operator to guess a count. They want it to "just run" and detect
   the count itself.

## Approach (decided)

- **Always-on, auto-detecting, bypassable.** Diarization defaults **on** and
  passes pyannote **no** speaker bounds, so pyannote estimates the count. The
  per-job "Max speakers" number input is replaced by a single **`diarize`
  boolean** (default `true`); turning it off is the only knob (the fast path for
  a known single-speaker file). There is no manual speaker-count override.
- **Canonical labels, client-side.** Nextext renumbers pyannote's raw labels to
  `Speaker N` by **first-appearance time** before alignment. The `/diarize`
  server stays a faithful pyannote proxy — canonicalization is a consumer
  concern and lives here.
- **Word-level, segment-anchored precision.** Nextext requests word timestamps
  from Whisper and assigns speakers per **word**; a Whisper segment whose words
  carry two speakers is split at the exact word the voice changes. This needs a
  small, backward-compatible change to the self-hosted ASR server, which today
  emits no word timestamps. Nextext **degrades to today's segment-level
  behavior** when an endpoint returns no words, so the Nextext change is safe to
  ship independently.
- **Hide the trivial column.** When diarization resolves to ≤1 distinct speaker
  (one voice, or a fail-soft empty result), the transcript carries **no** speaker
  column.
- **Two repos, two PRs.** `vllm-service` (ASR word timestamps) lands first or
  alongside; Nextext carries the rest. Consolidate the duplicated overlap logic
  as part of the Nextext change.

## Component: `vllm-service/src/asr_server.py` (separate repo/PR)

Backward-compatible addition — segment-only requests are byte-for-byte
unchanged.

- Accept a repeated `timestamp_granularities[]` form field on
  `POST /v1/audio/transcriptions` (OpenAI's field name).
- When it contains `"word"`, pass `word_timestamps=True` to
  `_model.transcribe(...)` (openai-whisper supports this natively) and add a
  **top-level** `words` array to the `verbose_json` body, matching OpenAI's
  shape:

  ```json
  {
    "task": "transcribe",
    "language": "en",
    "duration": 12.3,
    "text": "...",
    "segments": [ ... ],
    "words": [ {"word": "Hello", "start": 0.00, "end": 0.42}, ... ]
  }
  ```

  Each word's `start`/`end` are absolute seconds. Absent / segment-only requests
  omit `words` exactly as today.
- The pinned openai-whisper aggregates per-word timings under each segment's
  `words`; flatten them (in chronological order) into the top-level array.

Docs (`vllm-service/CLAUDE.md`, `README.md`, `.env.example` if relevant): note
that `verbose_json` now returns `words` when `timestamp_granularities=["word"]`.

## Component: `nextext/core/transcription.py`

- **Request word timestamps.** In `transcription()`, change the transcribe call
  to `timestamp_granularities=["segment", "word"]` (both — segments still carry
  the clean text and `no_speech_prob`; words drive speaker boundaries). Capture
  `response.words` (list of `{word, start, end}`) into
  `transcription_result["words"]` (empty list when the endpoint returns none).
  The `translate` endpoint path is unchanged (no diarization on that path today;
  it stays segment-only).
- **Remove the duplicated diarization code.** Delete
  `ExternalWhisperTranscriber.diarization()` and the module-level
  `_assign_speakers` — they duplicate `diarization.py::assign_speakers_by_overlap`
  and are exercised only by tests; the production pipeline never calls them.
  Speaker assignment consolidates in `nextext/core/diarization.py` (below).
- **`transcript_output()`** keeps building the DataFrame and running
  `_merge_transcriptions_by_sentence`, but the "drop speaker column" rule moves
  from `n_speakers <= 1` to "≤1 distinct speaker actually present" (see
  pipeline). `_merge_transcriptions_by_sentence` is unchanged — it already
  breaks a merged row on speaker change, which is exactly what we want once
  sub-segments are speaker-correct.

## Component: `nextext/core/diarization.py` (single home for speaker logic)

- **`diarize_file(path, *, num_speakers=None, min_speakers=None, max_speakers=None)`**
  — unchanged signature (keeps the bound params for API completeness and the
  server's mutual-exclusion contract), but the pipeline now calls it with **no
  bounds** → pyannote auto-detects. Fail-soft behavior (unset endpoint / HTTP /
  transport / non-dict payload → `[]`) is unchanged.

- **`canonicalize_speaker_labels(turns) -> list[dict]`** (new) — renumber raw
  pyannote labels to `Speaker N` by first-appearance time:
  1. Sort turns by `start` (stable).
  2. Walk them; the first time a raw label is seen, map it to the next integer
     (`Speaker 1`, `Speaker 2`, …).
  3. Return turns with rewritten `speaker` values (original order preserved;
     only the label string changes).

  The label prefix is a module constant `SPEAKER_LABEL_PREFIX = "Speaker"`.
  (Optional, out of scope for the first cut: localize to `Sprecher` via
  `NEXTEXT_RESPONSE_LANGUAGE`.)

- **`build_speaker_segments(segments, words, turns) -> list[dict]`** (new) —
  segment-anchored word-level assignment. Returns a **new** segment list
  (splitting changes length, so no in-place mutation):
  - **Word path** (when `words` is non-empty): assign each word a speaker =
    argmax cumulative overlap with `turns` (words overlapping no turn → `None`).
    For each Whisper segment, collect its words (word whose midpoint ∈
    `[seg.start, seg.end]`). If the segment has no words or all its words share
    one speaker → emit the segment unchanged with that speaker (**exact Whisper
    text preserved** — matters for space-less scripts). If its words carry ≥2
    distinct speakers → split at each speaker change: each sub-segment's
    `start`/`end` = its words' span, `text` = its words joined, `speaker` = that
    speaker.
  - **Fallback path** (when `words` is empty): apply the existing
    `assign_speakers_by_overlap` in place — today's segment-level behavior, so an
    endpoint without word timestamps is no worse than today (and still gets
    canonical labels).

- **`assign_speakers_by_overlap`** is retained as the fallback aligner (still
  covered by tests). It becomes an internal helper of `build_speaker_segments`
  plus a public fallback.

## Component: `nextext/pipeline.py`

`transcription_pipeline(file_path, src_lang, diarize: bool)` replaces the
`n_speakers: int` parameter.

```python
config = load_whisper_env()
transcriber = ExternalWhisperTranscriber(file_path=file_path, src_lang=src_lang, model_id=config.model)
transcriber.transcription()
segments = transcriber.transcription_result["segments"] if transcriber.transcription_result else []
words = transcriber.transcription_result.get("words", []) if transcriber.transcription_result else []

if diarize and segments:
    turns = canonicalize_speaker_labels(diarize_file(file_path))  # no bounds → auto-detect
    if turns:
        labeled = build_speaker_segments(segments, words, turns)
        transcriber.transcription_result["segments"] = labeled

df = transcriber.transcript_output()
```

- Diarization is skipped when `diarize` is `False` or the transcript is empty
  (VAD/no-speech short-circuit) — no wasted `/diarize` call.
- **Hide-if-≤1:** `transcript_output()` drops the speaker column when the number
  of distinct non-null speakers across the segments is ≤1 (replaces the
  `n_speakers <= 1` rule). Diarization that finds one voice, or fails soft,
  yields a clean transcript.
- `ExternalWhisperTranscriber` no longer needs `n_speakers`; drop the field and
  its `diarization()` method (see transcription component).

## Component: API — `schemas.py`, `jobs.py`

- `JobOptions.speakers: int = Field(default=1, ge=1, le=10)` →
  **`diarize: bool = True`**. `model_config = ConfigDict(extra="forbid")` is
  retained, so a client that still sends `speakers` gets `422` — an intentional,
  documented contract change (the React SPA is updated in lockstep; jobs are
  in-memory with no other consumers).
- `jobs.py`: `file_opts["speakers"]` → `file_opts["diarize"]`; the
  `transcription_pipeline(...)` call passes `diarize=file_opts["diarize"]`.

## Component: CLI — `nextext/cli.py`

- Replace `--speakers N` (default 1) with a boolean flag, default **on**:
  `parser.add_argument("--diarize", action=argparse.BooleanOptionalAction, default=True)`
  → `--diarize` / `--no-diarize`.
- The `transcription_pipeline(...)` call passes `diarize=args.diarize`.

## Component: Frontend — `frontend/src`

- `components/upload/UploadForm.tsx`: replace the "Max speakers" number input
  (`speakers`, 1–10) with a **"Detect speakers"** checkbox bound to a
  `diarize: boolean` state, default `true`. Submit `diarize` in the options.
- `api/types.ts`: `speakers: number` → `diarize: boolean` on the job-options
  type.
- Update the affected component/unit tests and fixtures (`speakers: 1` →
  `diarize: true`).

## Wire contract (unchanged on `/diarize`; extended on ASR)

- `/diarize` request/response is unchanged; Nextext simply stops sending speaker
  bounds. pyannote auto-detects when none are given.
- `/v1/audio/transcriptions` gains the optional `timestamp_granularities[]`
  request field and the optional top-level `words` response array (above).

## Error handling & edge cases

| Situation | Behavior |
|---|---|
| Endpoint returns no `words` (old ASR / segment-only) | Fall back to segment-level overlap = today's behavior, + canonical labels |
| `/diarize` unset or unreachable | Fail-soft: no turns → unlabelled transcript → no speaker column |
| Empty transcript (VAD/no-speech) | Diarization skipped entirely |
| Diarization finds 1 speaker | Labels applied, then column dropped (≤1 distinct) |
| `diarize = false` | Diarization skipped; fast path |
| CJK / space-less script, **mixed** segment | Sub-segment text is word-join with spaces — imperfect but rare; single-speaker segments keep exact Whisper text. Known limitation. |

## Tests

- **`tests/test_diarization.py`** — new: `canonicalize_speaker_labels` (first
  voice → `Speaker 1`; `SPEAKER_02`-before-`SPEAKER_00` renumbers to appearance
  order; contiguity). `build_speaker_segments`: single-speaker segment kept
  verbatim; a mixed segment splits at the right word with correct times/text;
  empty `words` → segment-level fallback; word overlapping no turn → unlabelled.
  Update `diarize_file` tests to the no-bounds call (drop `max_speakers=` from
  the pipeline-path assertions; keep a bounds test for the still-supported
  params).
- **`tests/test_transcription.py`** — request asserts
  `timestamp_granularities=["segment", "word"]` and that `words` is captured;
  **remove** the `diarization()` / `_assign_speakers` tests (that code is
  deleted) or repoint them at `diarization.build_speaker_segments`.
- **`tests/test_pipeline.py`** — `diarize=True` sends no bounds and calls the
  word-level builder; `diarize=False` skips; empty transcript skips; hide-if-≤1
  drops the column; ≥2 speakers keeps it.
- **`tests/test_schemas.py` / API tests** — `diarize` default `true`; a payload
  with `speakers` now `422`s (extra forbidden).
- **CLI test** — `--no-diarize` sets `diarize=False`; default is `True`.
- **Frontend** — `UploadForm` renders the checkbox (default checked) and submits
  `diarize`; update `jobs`/`ResultPanel` fixtures.
- **vllm-service** — an asr test that `timestamp_granularities=["word"]` returns
  a non-empty `words` array with monotonic times (separate repo).
- Gate: full `uv run pytest`, `cd frontend && pnpm test`, `pre-commit run
  --all-files` (ruff + pyrefly + docstrings), `make verify`.

## Docs

- **Nextext `CLAUDE.md`** — diarization is always-on + auto-detecting + bypassable;
  drop "max speakers > 1 triggers diarization"; document the `diarize` option, the
  canonical `Speaker N` labels, the word-level alignment and its segment-level
  fallback, and hide-if-≤1.
- **`.env.example`** — no new backend env (diarization is a per-job boolean, not
  env-gated); leave `DIARIZE_API_BASE` / `DIARIZE_TIMEOUT` as-is.
- **`vllm-service`** — ASR `words` response note (above).

## Out of scope

- Server-side pyannote parameter tuning (segmentation thresholds, clustering) —
  revisit only if precision is still lacking after word-level alignment.
- Overlapping / simultaneous-speech attribution — pyannote 3.1 assigns one
  speaker per turn; a word in an overlap region maps to the max-overlap turn.
- Localized speaker labels beyond an optional `Sprecher N` follow-up.
- Durable storage of diarization output — jobs remain in-memory.
- A manual speaker-count override — explicitly removed per the always-on choice.
