# Sentence restoration for low-punctuation transcripts — design

**Date:** 2026-07-12
**Status:** Approved
**Branch:** `Nextext-ara`

## Problem

Whisper transcribes Arabic (and other scripts that omit terminal punctuation)
with essentially **no sentence-ending marks**. Nextext's
`_merge_transcriptions_by_sentence` (`nextext/core/transcription.py`) starts from
Whisper's naturally granular segments and re-merges them into one row per
sentence, flushing only when the running text `_ends_with_punctuation`
(`. ! ? ؟ ۔`) **or** the speaker changes. With no punctuation the first signal
never fires, so a row grows until the speaker changes — one 34 s / 60 s blob per
speaker turn (undiarized: the **entire file** collapses to a single row).

The merge is **not** cosmetic: `translation_pipeline` translates strictly
per-row (`df["text"].apply(translate)`, `pipeline.py:156`), so each row is a
standalone translation unit. The current output therefore sits on the
*under-split* failure mode (a whole turn = many sentences fused into one
overstuffed translation unit). The opposite fix — a length/duration cap — sits
on the *over-split* mode: arbitrary mid-sentence fragments, which translate to
proven nonsense. The correct target is between them: **exactly one sentence per
row**. The goal is real sentence boundaries, not size tuning.

Arabic sentence-boundary detection on unpunctuated text genuinely needs a model;
the offline NLP libs (camel-tools' heavier features are excluded by the no-torch
invariant; a rule-based tokenizer has nothing to work with without punctuation).
This fits Nextext's architecture — all inference is external HTTP, and
`TEXT_MODEL` is already wired via `InferencePipeline`.

## Approach (decided)

A new stateless agent recovers sentence boundaries from the **word timestamps
Nextext already fetches** (`timestamp_granularities=["segment","word"]`) and
rebuilds the segment list so each segment is one sentence, then feeds the
existing merge/translate path unchanged.

- **Runs always for any low-punctuation transcript** (gated on punctuation
  density, language-agnostic — *not* on the translate flag), so readability, the
  sentence-level docint JSONL, and per-row hate-speech localization all benefit
  even without translation. A no-op for already-punctuated English/German.
- **Hallucination-proof contract:** the LLM returns only **sentence-ending word
  indices plus a 1-of-3 type code** (`S`/`Q`/`E`) — never text. Text and
  timestamps always come from Whisper's words, so the model can never reword the
  transcript or corrupt a timestamp; a bad index or code is sanitized.
- **Insert approach (decided):** each restored sentence gets its
  **model-classified terminal mark** appended when it lacks one — statement `.`,
  question `؟`, exclamation `!` — so the existing
  `_merge_transcriptions_by_sentence` fires naturally with zero special-casing,
  readability improves, and the translator receives well-formed sentences **with
  the correct mood** (a question closed as a statement can mistranslate — flipped
  particle/word order in the target). The type rides alongside each boundary as a
  1-of-3 code the LLM returns; it keeps the words/timestamps model-untouched, and
  an unclassified or forced boundary defaults to `.`.
- **Fail-soft** throughout (matching NER / diarization / summary): any failure
  degrades to today's behavior, never crashes a job.

## Components

### `nextext/core/sentence_segmentation.py` (new agent)

`terminal_punctuation_ratio(text: str) -> float`
Count of terminal marks (`. ! ? ؟ ۔ …`) ÷ `max(word_count, 1)`. The gate
primitive; unit-testable in isolation.

`restore_sentence_segments(words, turns, inference_pipeline, *, default_mark=".") -> list[dict]`
1. `not words` → return `[]` (caller keeps existing segments — cannot re-time
   without word offsets).
2. Label each word with a speaker via `_speaker_by_overlap` (reused from
   `diarization.py`) when `turns` is given, else `None`.
3. Partition words into **contiguous same-speaker runs** (one run when
   undiarized) — a sentence is never allowed to cross a speaker change.
4. Per run, segment its words into sentences (see `_segment_run`, which returns
   `(end_index, mark)` per sentence), then emit one segment per sentence:
   `start` = first word's `start`, `end` = last word's `end`, `text` = words
   joined by single spaces (same convention as `_word_run_segment`; correct for
   space-separated Arabic) with the sentence's **`mark`** appended when the text
   lacks terminal punctuation; `speaker` set only when the run has one.
   `default_mark` is the fallback used for any boundary the model left
   unclassified.

`_TYPE_TO_MARK` maps the model's type codes to terminal marks:
`{"S": ".", "Q": "؟", "E": "!"}` (Arabic question mark — recognized by
`_ends_with_punctuation`); any other/absent code → `default_mark` (`.`).

`_segment_run(run_words, inference_pipeline) -> list[tuple[int, str]]` (private)
- Chunk `run_words` into windows of `_SEGMENT_WORD_BUDGET` (400) — a chunk edge
  forces a sentence boundary (mark `.`; accepted: ≤1 possible over-split per 400
  words).
- Per window: build a numbered token list, call
  `inference_pipeline.call_model(prompt, include_system_prompt=False,
  temperature=0.0, num_predict=_SEGMENT_MAX_TOKENS)` (`_SEGMENT_MAX_TOKENS = 256`
  — the reply is only a short `index:code` list) with the `sentence_segment`
  prompt, parse the comma-separated `index:code` pairs, **sanitize** (keep pairs
  whose index is in range; ascending, deduped by index; unknown/missing code →
  `.` via `_TYPE_TO_MARK`), offset each index by the window start.
- On an empty/unparseable result or a raised call, the window degrades to a
  single sentence (boundary at its end, mark `.`) — fail-soft per run.
- Returns ascending `(end_index, mark)` pairs, always terminating at the last
  word.

### `nextext/utils/prompts/en/sentence_segment.txt` (new prompt)

Instructs: given a numbered list of transcript word tokens with no punctuation,
output **only** a comma-separated, ascending list of `index:code` pairs — one per
sentence — where `index` is the sentence's last token and `code` is its type:
`S` (statement), `Q` (question), or `E` (exclamation). Example: `4:S, 9:Q, 15:S`.
Do not output words; do not add/remove/reorder tokens. Placeholder `{tokens}`.
English-only (instruction is content-agnostic; `load_prompt`'s English fallback
covers other `NEXTEXT_RESPONSE_LANGUAGE` values). Only the sentence-**terminal**
mark is restored — internal punctuation is not (out of scope).

### `nextext/pipeline.py`

In `transcription_pipeline`, compute `turns` (diarize → VAD-gate → canonicalize)
as today, then branch **before** segment assignment so restoration supersedes
`build_speaker_segments` on the low-punctuation path:

```python
restore_cfg = load_sentence_restore_env()
low_punct = bool(segments) and terminal_punctuation_ratio(_segments_text(segments)) < restore_cfg.min_punct_ratio
if restore_cfg.enabled and words and low_punct and transcriber.transcription_result is not None:
    restored = restore_sentence_segments(words, turns or None, InferencePipeline())
    if restored:
        transcriber.transcription_result["segments"] = restored
elif diarize and turns and transcriber.transcription_result is not None:
    transcriber.transcription_result["segments"] = build_speaker_segments(segments, words, turns)
```

- The gate is measured on the **raw** Whisper segment text (before any speaker
  split); `_segments_text(segments)` is a one-line join of the segments' `text`
  fields. `InferencePipeline()` is constructed **lazily, only when restoration
  runs**, so pure-transcription jobs never require `TEXT_MODEL`. When restoration
  runs diarized, it re-derives speakers from the same `turns`, so labeling stays
  consistent; the earlier `build_speaker_segments` is simply not called on that
  path.
- Restored segments already carry terminal punctuation, so
  `transcript_output()` → `_merge_transcriptions_by_sentence` emits one row per
  sentence and still applies the `≥2 distinct speakers` column rule.

### `nextext/utils/env_cfg.py`

`load_sentence_restore_env() -> SentenceRestoreConfig(enabled: bool, min_punct_ratio: float)`:
- `NEXTEXT_SENTENCE_RESTORE` — tri-state via `_parse_tristate_bool`, **default on**.
- `SENTENCE_RESTORE_MIN_PUNCT_RATIO` — float, **default `0.01`** (≈ <1 terminator
  per 100 words); values outside `(0, 1)` or non-numeric warn → default.
  Punctuated EN/DE sit at ~0.05–0.07 (skip); unpunctuated Arabic ~0 (restore).

## Error handling

| Situation | Behavior |
|---|---|
| `NEXTEXT_SENTENCE_RESTORE` off | Skip; segments unchanged (diarize path as today). |
| Well-punctuated (ratio ≥ threshold) | Skip; segments unchanged. |
| No word timestamps (`words` empty) | Skip restore (cannot re-time); keep existing segments. |
| `TEXT_MODEL` unset / endpoint unreachable / call raises | Per-run fail-soft → run emitted as one segment; job continues (no worse than today's blob). |
| Empty / garbage / non-monotonic indices | Sanitized; empty → single segment for the run. |
| Unknown / missing / malformed type code | Default to `.` (statement) via `_TYPE_TO_MARK`. |
| Empty transcript (no segments) | Skip (nothing to segment), as today. |

## Tests (TDD)

- `tests/test_sentence_segmentation.py` — `terminal_punctuation_ratio` (high for
  punctuated EN, ~0 for unpunctuated AR); `restore_sentence_segments` with a
  mocked `call_model` returning `index:code` pairs → correct sentence segments,
  word-derived `start`/`end`, per-sentence mark (`S`→`.`, `Q`→`؟`, `E`→`!`),
  speaker inherited from turns; unknown/missing code → `.`; multi-speaker → runs
  split at the speaker change; no words → `[]`; `call_model` raises → fail-soft
  single segment per run (mark `.`); garbage indices sanitized; run > budget →
  multiple calls with offset indices (chunk-edge boundary → `.`).
- `tests/test_pipeline.py` — restore runs when enabled + low-punct + words;
  skipped when disabled / well-punctuated / no words; supersedes
  `build_speaker_segments` on the diarized low-punct path; undiarized low-punct →
  single run, no speaker column; `InferencePipeline` not constructed when skipped.
- `tests/test_env_cfg.py` — `load_sentence_restore_env` defaults + tri-state
  enable + ratio parsing/fallbacks.
- `tests/test_transcription.py` — unchanged; add a case that inserted-punctuation
  sentence segments merge to one row each.
- `tests/test_no_torch.py` — unaffected (agent is pure HTTP + stdlib).
- Docs: `.env.example`, `CLAUDE.md`, `AGENTS.md` (new agent I/O contract).

## Out of scope

- Internal punctuation (commas, colons, quotes) — only the sentence-**terminal**
  mark is restored (statement `.` / question `؟` / exclamation `!`). Per-sentence
  translation makes internal marks far lower-value and higher-risk.
- Re-punctuating already-punctuated languages (the density gate excludes them).
- Whisper-side punctuation (`initial_prompt` / model choice) — an orthogonal,
  cheaper experiment an operator can try; not part of this design.
- A per-job `JobOptions` toggle — restoration is operator-env-gated to match
  "always for any low-punctuation transcript"; no request field is added.
- Pause/gap-based splitting on word timestamps — considered, but pauses only
  correlate with sentence ends (Arabic speakers pause mid-clause and run
  sentences together), so it cannot guarantee the whole-sentence invariant the
  translator needs.
