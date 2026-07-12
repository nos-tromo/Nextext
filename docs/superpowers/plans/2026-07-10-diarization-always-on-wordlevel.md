# Diarization: always-on auto-detect, canonical labels, word-level precision — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make speaker diarization always-on and auto-detecting (bypassable), relabel speakers to contiguous `Speaker N` by first appearance, and split mixed-speaker Whisper segments at the word level for accurate dialogue.

**Architecture:** Nextext requests word timestamps from the Whisper endpoint and assigns speakers per word, splitting a Whisper segment only when its words carry two speakers (exact segment text is preserved otherwise). Raw pyannote labels are renumbered client-side by first-appearance time. Diarization becomes a per-job boolean (`diarize`, default on) that passes pyannote no bounds so it estimates the count. A small, backward-compatible change to the self-hosted ASR server (`vllm-service`) makes it emit the `words` array; Nextext falls back to today's segment-level behavior when no words are returned.

**Tech Stack:** Python 3.12 (FastAPI, pandas, httpx, openai SDK), pytest + respx, ruff + pyrefly + pre-commit; React 19 + Vite + Tailwind v4 + vitest frontend; `vllm-service` FastAPI + openai-whisper (no pytest suite).

## Global Constraints

- **Python 3.12** target; every new/modified Python function gets a **Google-style docstring** and explicit type annotations (pyrefly is enforced).
- **Dependency management via `uv`** — no change to `pyproject.toml`/`uv.lock` is expected in this plan (no new deps).
- **No-torch invariant** in the Nextext backend — add no model runtimes; all inference stays HTTP.
- **Diarization is fail-soft** — an unset/unreachable `/diarize` endpoint yields no turns and an unlabelled transcript; never crash a job.
- **Skip diarization** when the transcript is empty (VAD/no-speech short-circuit) or `diarize` is false.
- **Frontend** consumes `@infra/ui`; keep business logic server-side; tests are vitest + `@testing-library/react`.
- **`vllm-service` has no pytest suite** — verify with `pre-commit run --all-files` (ruff + pyrefly) plus a manual `curl` smoke; keep changes backward-compatible (segment-only requests unchanged).
- **Commits:** small, topical, conventional prefix (`feat:`/`fix:`/`refactor:`/`test:`/`docs:`); end every commit message body with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- **Gate before done:** `uv run pytest`, `cd frontend && pnpm test`, `uv run pre-commit run --all-files`, `make verify`.
- **Label format:** the module constant `SPEAKER_LABEL_PREFIX = "Speaker"` → labels render as `Speaker 1`, `Speaker 2`, … (English-only for this cut).

---

### Task 1: `vllm-service` ASR — emit word timestamps (separate repo/PR)

**Repo:** `../vllm-service` — execute in a **separate branch** (`feature/asr-word-timestamps`) and its own PR. Independently mergeable; Nextext degrades gracefully without it, so the remaining tasks do not block on this one (their tests inject word data directly).

**Files:**
- Modify: `../vllm-service/src/asr_server.py` (`_transcribe` at lines 145-220; `transcriptions` endpoint at lines 223-248)

**Interfaces:**
- Produces: `POST /v1/audio/transcriptions` with `response_format=verbose_json` and `timestamp_granularities[]=word` returns a top-level `"words": [{"word": str, "start": float, "end": float}, ...]` array (absolute seconds, chronological). Segment-only requests are unchanged and omit `words`.

- [ ] **Step 1: Add the `timestamp_granularities` parameter and word flattening to `_transcribe`**

Change the signature and the `verbose_json` branch of `_transcribe` (lines 145-220):

```python
def _transcribe(
    file: UploadFile,
    task: str,
    language: str | None,
    prompt: str | None,
    temperature: float | None,
    response_format: str,
    timestamp_granularities: list[str] | None = None,
) -> Any:
    """Decode an upload and run Whisper, shaping the OpenAI response.

    Declared on the ``def`` (threadpool) endpoints below so a minutes-long
    decode never starves ``/health`` on the event loop; the module-level
    lock serializes the non-reentrant model.

    Args:
        file: Uploaded audio in any container ffmpeg can decode.
        task: ``transcribe`` (verbatim) or ``translate`` (to English).
        language: Source language hint; ``None`` lets Whisper auto-detect.
        prompt: Optional decoding prompt (``initial_prompt``).
        temperature: Optional sampling temperature; ``None`` uses Whisper's
            built-in fallback schedule.
        response_format: ``json`` (default), ``verbose_json``, or ``text``.
        timestamp_granularities: Optional OpenAI-style granularity list; when
            it contains ``"word"`` and ``response_format`` is ``verbose_json``,
            per-word timings are computed and returned as a top-level ``words``
            array.

    Returns:
        A dict (serialized as JSON) for ``json`` / ``verbose_json``, or a
        ``PlainTextResponse`` for ``text``.
    """
    audio_bytes = file.file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty audio payload")
    try:
        audio = _decode_audio(audio_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"failed to decode audio: {exc}") from exc

    want_words = bool(timestamp_granularities) and "word" in timestamp_granularities
    options: dict[str, Any] = {"task": task, "fp16": DEVICE != "cpu"}
    if want_words:
        options["word_timestamps"] = True
    if language:
        options["language"] = language
    if prompt:
        options["initial_prompt"] = prompt
    if temperature is not None:
        options["temperature"] = temperature
    try:
        with _model_lock:
            result = _model.transcribe(audio, **options)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"transcription failed: {exc}") from exc

    text = str(result.get("text", "")).strip()
    fmt = (response_format or "json").lower()
    if fmt == "text":
        return PlainTextResponse(text)
    if fmt == "verbose_json":
        segments = [
            {
                "id": int(seg.get("id", i)),
                "seek": int(seg.get("seek", 0)),
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": str(seg.get("text", "")),
                "tokens": list(seg.get("tokens", [])),
                "temperature": float(seg.get("temperature", 0.0)),
                "avg_logprob": float(seg.get("avg_logprob", 0.0)),
                "compression_ratio": float(seg.get("compression_ratio", 0.0)),
                "no_speech_prob": float(seg.get("no_speech_prob", 0.0)),
            }
            for i, seg in enumerate(result.get("segments", []))
        ]
        body: dict[str, Any] = {
            "task": task,
            "language": result.get("language"),
            "duration": len(audio) / SAMPLE_RATE,
            "text": text,
            "segments": segments,
        }
        if want_words:
            body["words"] = [
                {
                    "word": str(word.get("word", "")),
                    "start": float(word.get("start", 0.0)),
                    "end": float(word.get("end", 0.0)),
                }
                for seg in result.get("segments", [])
                for word in (seg.get("words") or [])
            ]
        return body
    return {"text": text}
```

- [ ] **Step 2: Accept the OpenAI-bracketed form field on the endpoint**

The OpenAI SDK sends the granularity list as the bracketed field name `timestamp_granularities[]`. Add it (aliased) to the `transcriptions` endpoint (lines 223-248):

```python
@app.post("/v1/audio/transcriptions")
def transcriptions(
    file: UploadFile = File(...),  # noqa: B008 — FastAPI dependency marker
    model: str | None = Form(default=None),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    temperature: float | None = Form(default=None),
    response_format: str = Form(default="json"),
    timestamp_granularities: list[str] | None = Form(default=None, alias="timestamp_granularities[]"),
) -> Any:
    """Transcribe an uploaded media file (Whisper ``task=transcribe``).

    ``model`` is accepted for OpenAI-client compatibility but ignored — the
    server always uses the model it loaded at boot.

    Args:
        file: Uploaded audio in any container ffmpeg can decode.
        model: Ignored; present for OpenAI-client compatibility.
        language: Source language hint; ``None`` auto-detects.
        prompt: Optional decoding prompt.
        temperature: Optional sampling temperature.
        response_format: ``json`` (default), ``verbose_json``, or ``text``.
        timestamp_granularities: OpenAI-style granularity list (bracketed form
            field ``timestamp_granularities[]``); ``["word"]`` adds per-word
            timings to a ``verbose_json`` response.

    Returns:
        The transcription in the requested ``response_format``.
    """
    return _transcribe(
        file, "transcribe", language, prompt, temperature, response_format, timestamp_granularities
    )
```

Leave the `translations` endpoint (lines 251-274) unchanged — it stays segment-only.

- [ ] **Step 3: Lint**

Run: `cd ../vllm-service && uv run pre-commit run --all-files`
Expected: ruff + pyrefly PASS (no type errors on the new `list[str] | None` param).

- [ ] **Step 4: Manual smoke (documented; run if a server is reachable)**

With a diarize/ASR dev server up, confirm the bracketed field elicits words:

Run:
```bash
curl -s -F 'file=@sample.wav' -F 'response_format=verbose_json' \
  -F 'timestamp_granularities[]=segment' -F 'timestamp_granularities[]=word' \
  http://localhost:8000/v1/audio/transcriptions | python -c 'import sys,json; d=json.load(sys.stdin); print("words:", len(d.get("words",[])), d.get("words",[])[:2])'
```
Expected: a non-zero `words:` count with `{word,start,end}` entries in chronological order. (If the count is 0, the SDK/field-name assumption is wrong — check whether the client sends `timestamp_granularities` without brackets and adjust the alias.)

- [ ] **Step 5: Commit (in `../vllm-service`)**

```bash
cd ../vllm-service
git add src/asr_server.py
git commit -m "$(cat <<'EOF'
feat: return word timestamps from /v1/audio/transcriptions

Accept timestamp_granularities[]=word (OpenAI-bracketed form field) and,
for verbose_json, run openai-whisper with word_timestamps=True and flatten
the per-segment word timings into a top-level words array. Segment-only
requests are unchanged. Enables word-level speaker assignment downstream.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Canonical speaker labels (`canonicalize_speaker_labels`)

**Files:**
- Modify: `nextext/core/diarization.py` (add constant + function; extend `__all__` at line 24)
- Test: `tests/test_diarization.py`

**Interfaces:**
- Produces: `SPEAKER_LABEL_PREFIX: str` and `canonicalize_speaker_labels(turns: list[dict[str, Any]]) -> list[dict[str, Any]]` — returns a new turn list (original order preserved) whose `speaker` values are remapped to `f"{SPEAKER_LABEL_PREFIX} {n}"`, numbered by first-appearance time (earliest `start` = `Speaker 1`).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_diarization.py`:

```python
from nextext.core.diarization import canonicalize_speaker_labels


def test_canonicalize_numbers_by_first_appearance() -> None:
    """Raw pyannote labels renumber to contiguous Speaker N by earliest start."""
    turns = [
        {"start": 5.0, "end": 6.0, "speaker": "SPEAKER_02"},
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_02"},
    ]

    result = canonicalize_speaker_labels(turns)

    # First voice heard (start=0.0, SPEAKER_00) -> Speaker 1; next new voice -> Speaker 2.
    assert [t["speaker"] for t in result] == ["Speaker 2", "Speaker 1", "Speaker 2"]
    # Original order and timings are preserved; only the label string changes.
    assert [t["start"] for t in result] == [5.0, 0.0, 1.0]


def test_canonicalize_empty_is_empty() -> None:
    """No turns canonicalizes to no turns."""
    assert canonicalize_speaker_labels([]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_diarization.py::test_canonicalize_numbers_by_first_appearance -v`
Expected: FAIL with `ImportError: cannot import name 'canonicalize_speaker_labels'`.

- [ ] **Step 3: Write minimal implementation**

In `nextext/core/diarization.py`, update `__all__` (line 24) and add the constant + function:

```python
__all__ = [
    "SPEAKER_LABEL_PREFIX",
    "assign_speakers_by_overlap",
    "build_speaker_segments",
    "canonicalize_speaker_labels",
    "diarize_file",
]

SPEAKER_LABEL_PREFIX = "Speaker"


def canonicalize_speaker_labels(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Renumber raw diarization labels to contiguous ``Speaker N`` by first appearance.

    pyannote's ``SPEAKER_00``/``SPEAKER_02`` labels are arbitrary and gap-y.
    This maps them to ``Speaker 1``, ``Speaker 2``, … in the order each label
    is first heard (earliest turn ``start``), so the first voice is always
    ``Speaker 1``. The input order is preserved in the output; only the
    ``speaker`` string changes.

    Args:
        turns (list[dict[str, Any]]): Speaker turns with ``start`` / ``end`` /
            ``speaker`` keys, as returned by :func:`diarize_file`.

    Returns:
        list[dict[str, Any]]: New turn dicts (same order) with canonical labels.
    """
    mapping: dict[str, str] = {}
    for turn in sorted(turns, key=lambda t: float(t["start"])):
        raw = str(turn["speaker"])
        if raw not in mapping:
            mapping[raw] = f"{SPEAKER_LABEL_PREFIX} {len(mapping) + 1}"
    return [{**turn, "speaker": mapping[str(turn["speaker"])]} for turn in turns]
```

(Constant and function go near the top of the module, after the imports/`__all__`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_diarization.py -v`
Expected: PASS (both new tests + the existing ones).

- [ ] **Step 5: Commit**

```bash
git add nextext/core/diarization.py tests/test_diarization.py
git commit -m "$(cat <<'EOF'
feat: canonicalize diarization labels to contiguous Speaker N

Renumber pyannote's arbitrary SPEAKER_0x labels to Speaker 1, Speaker 2, …
in first-appearance order so the first voice heard is always Speaker 1.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Word-level segment builder (`build_speaker_segments`)

**Files:**
- Modify: `nextext/core/diarization.py` (add `_speaker_by_overlap`; refactor `assign_speakers_by_overlap` to use it; add `build_speaker_segments`)
- Test: `tests/test_diarization.py`

**Interfaces:**
- Consumes: `canonicalize_speaker_labels` output (turns with `Speaker N` labels).
- Produces:
  - `build_speaker_segments(segments: list[dict[str, Any]], words: list[dict[str, Any]], turns: list[dict[str, Any]]) -> list[dict[str, Any]]` — returns a **new** segment list. Word path (when `words` non-empty): a segment whose words carry ≥2 speakers is split at each speaker change (`text` = its words joined by spaces, `start`/`end` = its words' span); single-speaker segments are emitted verbatim with a `speaker` key. Fallback path (empty `words`): segment-level max-overlap assignment on copies.
  - `_speaker_by_overlap(start: float, end: float, turns: list[dict[str, Any]]) -> str | None` — the max cumulative-overlap speaker for a `[start, end]` window, or `None`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_diarization.py`:

```python
from nextext.core.diarization import build_speaker_segments

_TURNS = [
    {"start": 0.0, "end": 1.0, "speaker": "Speaker 1"},
    {"start": 1.0, "end": 2.0, "speaker": "Speaker 2"},
]


def test_build_keeps_single_speaker_segment_verbatim() -> None:
    """A segment whose words share one speaker is emitted with exact text preserved."""
    segments = [{"start": 0.0, "end": 0.9, "text": "hello world"}]
    words = [
        {"word": "hello", "start": 0.0, "end": 0.4},
        {"word": "world", "start": 0.4, "end": 0.8},
    ]

    result = build_speaker_segments(segments, words, _TURNS)

    assert result == [{"start": 0.0, "end": 0.9, "text": "hello world", "speaker": "Speaker 1"}]


def test_build_splits_mixed_speaker_segment_at_word() -> None:
    """A segment spanning a speaker change splits at the exact word."""
    segments = [{"start": 0.0, "end": 2.0, "text": "hi there"}]
    words = [
        {"word": "hi", "start": 0.0, "end": 0.4},     # midpoint 0.2 -> Speaker 1
        {"word": "there", "start": 1.2, "end": 1.8},  # midpoint 1.5 -> Speaker 2
    ]

    result = build_speaker_segments(segments, words, _TURNS)

    assert result == [
        {"start": 0.0, "end": 0.4, "text": "hi", "speaker": "Speaker 1"},
        {"start": 1.2, "end": 1.8, "text": "there", "speaker": "Speaker 2"},
    ]


def test_build_falls_back_to_segment_level_without_words() -> None:
    """With no word timestamps, assignment is segment-level max overlap."""
    segments = [{"start": 0.0, "end": 0.9, "text": "hello"}]

    result = build_speaker_segments(segments, [], _TURNS)

    assert result == [{"start": 0.0, "end": 0.9, "text": "hello", "speaker": "Speaker 1"}]
    # Input is not mutated (a copy is returned).
    assert "speaker" not in segments[0]


def test_build_unlabeled_word_inherits_neighbouring_run() -> None:
    """A word overlapping no turn does not force a split; it joins the current run."""
    segments = [{"start": 0.0, "end": 3.0, "text": "a b c"}]
    words = [
        {"word": "a", "start": 0.0, "end": 0.4},    # Speaker 1
        {"word": "b", "start": 2.2, "end": 2.4},    # overlaps no turn -> None
        {"word": "c", "start": 2.5, "end": 2.8},    # overlaps no turn -> None
    ]
    turns = [{"start": 0.0, "end": 1.0, "speaker": "Speaker 1"}]

    result = build_speaker_segments(segments, words, turns)

    # Single distinct speaker (Speaker 1) -> verbatim segment, exact text.
    assert result == [{"start": 0.0, "end": 3.0, "text": "a b c", "speaker": "Speaker 1"}]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_diarization.py::test_build_splits_mixed_speaker_segment_at_word -v`
Expected: FAIL with `ImportError: cannot import name 'build_speaker_segments'`.

- [ ] **Step 3: Write the implementation**

In `nextext/core/diarization.py`, add `_speaker_by_overlap`, refactor `assign_speakers_by_overlap` (lines 113-146) to use it, and add `build_speaker_segments`:

```python
def _speaker_by_overlap(
    start: float,
    end: float,
    turns: list[dict[str, Any]],
) -> str | None:
    """Return the speaker with the greatest cumulative overlap of ``[start, end]``.

    Args:
        start (float): Window start in seconds.
        end (float): Window end in seconds.
        turns (list[dict[str, Any]]): Speaker turns with ``start`` / ``end`` /
            ``speaker`` keys.

    Returns:
        str | None: The maximally-overlapping speaker label, or ``None`` when
            the window overlaps no turn.
    """
    durations: dict[str, float] = {}
    for turn in turns:
        overlap_start = max(start, float(turn["start"]))
        overlap_end = min(end, float(turn["end"]))
        if overlap_end > overlap_start:
            speaker = str(turn["speaker"])
            durations[speaker] = durations.get(speaker, 0.0) + (overlap_end - overlap_start)
    if not durations:
        return None
    return max(durations, key=lambda s: durations[s])


def assign_speakers_by_overlap(
    transcription_segments: list[dict[str, Any]],
    diarize_segments: list[dict[str, Any]],
) -> None:
    """Label transcript segments with the maximally-overlapping speaker.

    For each transcription segment, the total temporal overlap against every
    diarization turn is accumulated per speaker, and the speaker with the
    greatest overlap wins. Segments that overlap no turn are left untouched
    (they gain no ``speaker`` key). ``transcription_segments`` is mutated in
    place. This is the segment-level fallback used when word timestamps are
    unavailable.

    Args:
        transcription_segments (list[dict[str, Any]]): Whisper segments with
            float ``start`` / ``end`` keys (seconds). Each gains a ``speaker``
            key when an overlapping turn exists.
        diarize_segments (list[dict[str, Any]]): Speaker turns from
            :func:`diarize_file`, each with ``start`` / ``end`` / ``speaker``.
    """
    for segment in transcription_segments:
        speaker = _speaker_by_overlap(float(segment["start"]), float(segment["end"]), diarize_segments)
        if speaker is not None:
            segment["speaker"] = speaker


def build_speaker_segments(
    segments: list[dict[str, Any]],
    words: list[dict[str, Any]],
    turns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign speakers to transcript segments, splitting mixed-speaker segments by word.

    When ``words`` is available, each word is assigned the maximally-overlapping
    speaker and every Whisper segment is inspected: if all of its words share a
    single speaker the segment is emitted unchanged (with a ``speaker`` key and
    its **exact** text preserved); if its words carry two or more speakers the
    segment is split at each speaker change, one output segment per run of
    same-speaker words. Words overlapping no turn do not force a split — they
    join the surrounding run.

    When ``words`` is empty (an endpoint that returns no word timestamps), it
    falls back to segment-level :func:`assign_speakers_by_overlap` on copies of
    the input.

    Args:
        segments (list[dict[str, Any]]): Whisper segments with float ``start`` /
            ``end`` and ``text`` keys.
        words (list[dict[str, Any]]): Whisper words with float ``start`` /
            ``end`` and ``word`` keys; may be empty.
        turns (list[dict[str, Any]]): Canonicalized speaker turns.

    Returns:
        list[dict[str, Any]]: New segment dicts, speaker-labeled and — where a
            segment spanned a speaker change — split at the word boundary.
    """
    if not words:
        labeled = [dict(segment) for segment in segments]
        assign_speakers_by_overlap(labeled, turns)
        return labeled

    result: list[dict[str, Any]] = []
    for segment in segments:
        seg_start = float(segment["start"])
        seg_end = float(segment["end"])
        seg_words = [w for w in words if seg_start <= (float(w["start"]) + float(w["end"])) / 2 < seg_end]
        labeled_words = [
            (w, _speaker_by_overlap(float(w["start"]), float(w["end"]), turns)) for w in seg_words
        ]
        distinct = {speaker for _, speaker in labeled_words if speaker is not None}

        if len(distinct) <= 1:
            new_segment = dict(segment)
            speaker = next(iter(distinct), None)
            if speaker is None:
                speaker = _speaker_by_overlap(seg_start, seg_end, turns)
            if speaker is not None:
                new_segment["speaker"] = speaker
            result.append(new_segment)
            continue

        run_words: list[dict[str, Any]] = []
        run_speaker: str | None = None
        for word, speaker in labeled_words:
            if run_words and speaker is not None and run_speaker is not None and speaker != run_speaker:
                result.append(_word_run_segment(run_words, run_speaker))
                run_words = []
                run_speaker = None
            run_words.append(word)
            if speaker is not None:
                run_speaker = speaker
        if run_words:
            result.append(_word_run_segment(run_words, run_speaker))
    return result


def _word_run_segment(run_words: list[dict[str, Any]], speaker: str | None) -> dict[str, Any]:
    """Build one output segment from a run of same-speaker words.

    Args:
        run_words (list[dict[str, Any]]): Consecutive Whisper words with
            ``word`` / ``start`` / ``end`` keys.
        speaker (str | None): The run's speaker label, if any.

    Returns:
        dict[str, Any]: A segment with ``start`` / ``end`` / ``text`` and an
            optional ``speaker`` key. Text joins the words with single spaces
            (imperfect for space-less scripts — a documented limitation, and
            only hit for genuinely mixed-speaker segments).
    """
    segment: dict[str, Any] = {
        "start": float(run_words[0]["start"]),
        "end": float(run_words[-1]["end"]),
        "text": " ".join(str(w["word"]).strip() for w in run_words).strip(),
    }
    if speaker is not None:
        segment["speaker"] = speaker
    return segment
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_diarization.py -v`
Expected: PASS (new builder tests + existing `assign_speakers_by_overlap` tests still green after the refactor).

- [ ] **Step 5: Commit**

```bash
git add nextext/core/diarization.py tests/test_diarization.py
git commit -m "$(cat <<'EOF'
feat: word-level speaker assignment that splits mixed-speaker segments

build_speaker_segments assigns each Whisper word its max-overlap speaker and
splits a segment at the exact word where the voice changes; single-speaker
segments keep their exact text. Falls back to segment-level overlap when no
word timestamps are available. assign_speakers_by_overlap refactored onto a
shared _speaker_by_overlap helper.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Transcription — request words, capture them, remove dead diarization code, hide-if-≤1

**Files:**
- Modify: `nextext/core/transcription.py` (request at lines 370-378; capture at 388-398; delete `_assign_speakers` 195-220 and `ExternalWhisperTranscriber.diarization()` 409-435; `transcript_output` drop rule 437-476)
- Test: `tests/test_transcription.py` (imports at line 16; delete diarization tests 203-312; add `words` to fake responses; assert granularities)

**Interfaces:**
- Produces: `transcriber.transcription_result` now carries `{"segments": [...], "words": [...]}` (words empty when the endpoint returns none). `transcript_output()` renders a `speaker` column only when **≥2** distinct speakers are present. `ExternalWhisperTranscriber` no longer defines `diarization()` or `_assign_speakers`; the `n_speakers` field is retained (unused) until Task 5.

- [ ] **Step 1: Update the transcription test to assert word granularity + words capture, and delete the dead-code tests**

In `tests/test_transcription.py`:

Remove `_assign_speakers` from the import block (line 16) so it reads:

```python
from nextext.core.transcription import (
    ExternalWhisperTranscriber,
    _ends_with_punctuation,
    _merge_transcriptions_by_sentence,
    _seconds_to_time,
)
```

Delete the three `_assign_speakers` unit tests (lines 166-200) **and** the five `ExternalWhisperTranscriber.diarization()` tests (lines 203-312: `test_external_diarization_skips_for_single_speaker`, `_skips_when_no_segments`, `_assigns_speakers`, `_requires_transcription_first`, `_failure_propagates`).

Add a request-shape test near the other transcription-request tests (e.g. after the block ending line 723). It reuses the existing fake-client pattern:

```python
def test_transcription_requests_word_granularity_and_captures_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The transcribe call asks for segment+word granularity and stores words.

    Args:
        monkeypatch (pytest.MonkeyPatch): Patches the client, VAD guard, and
            the normalization seam.
    """
    seg = SimpleNamespace(start=0.0, end=1.0, text="hi", no_speech_prob=0.0)
    word = SimpleNamespace(word="hi", start=0.0, end=0.4)
    fake_response = SimpleNamespace(segments=[seg], words=[word], language="en")
    fake_client = MagicMock()
    fake_client.audio.transcriptions.create.return_value = fake_response

    transcriber = ExternalWhisperTranscriber.__new__(ExternalWhisperTranscriber)
    transcriber.file_path = transcription.Path(__file__)
    transcriber.src_lang = "en"
    transcriber.task = "transcribe"
    transcriber._model_id = "whisper-1"
    transcriber._client = None
    transcriber.transcription_result = None

    monkeypatch.setattr(type(transcriber), "_get_client", property(lambda self: fake_client))
    monkeypatch.setattr(transcription, "has_speech", lambda _path: True)
    monkeypatch.setattr(transcription, "normalize_for_transcription", _passthrough_normalize)

    transcriber.transcription()

    _, kwargs = fake_client.audio.transcriptions.create.call_args
    assert kwargs["timestamp_granularities"] == ["segment", "word"]
    assert transcriber.transcription_result["words"] == [{"word": "hi", "start": 0.0, "end": 0.4}]
```

Also update the existing fake responses that build `SimpleNamespace(segments=[...], language=...)` (lines 379, 424, 463, 625, 664, 696) to include `words=[]`, e.g. `SimpleNamespace(segments=[seg], words=[], language="es")`, so `getattr(response, "words", None)` resolves cleanly.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_transcription.py::test_transcription_requests_word_granularity_and_captures_words -v`
Expected: FAIL — `KeyError: 'words'` (result has no `words` yet) / granularity assertion mismatch.

- [ ] **Step 3: Update `transcription.py` — request words, capture them, delete dead code, fix drop rule**

(a) In `transcription()`, change the transcribe kwargs (lines 370-375) to request both granularities:

```python
                        kwargs: dict[str, Any] = {
                            "model": self._model_id,
                            "file": f,
                            "response_format": "verbose_json",
                            "timestamp_granularities": ["segment", "word"],
                        }
```

(b) Replace the result assembly (lines 388-398) to capture words:

```python
        raw_segments = [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "no_speech_prob": float(getattr(seg, "no_speech_prob", 0.0) or 0.0),
            }
            for seg in response.segments
        ]
        segments = _filter_no_speech_segments(raw_segments)
        words = [
            {"word": str(getattr(w, "word", "")), "start": float(getattr(w, "start", 0.0)), "end": float(getattr(w, "end", 0.0))}
            for w in (getattr(response, "words", None) or [])
        ]
        self.transcription_result = {"segments": segments, "words": words}
```

(c) Delete the module-level `_assign_speakers` function (lines 195-220) and the `ExternalWhisperTranscriber.diarization()` method (lines 409-435) entirely — the pipeline does diarization via `nextext/core/diarization.py`.

(d) Change `transcript_output()` (lines 437-476) so the speaker column appears only with ≥2 distinct speakers, removing the `n_speakers`-based drop:

```python
    def transcript_output(self) -> pd.DataFrame:
        """Get the external transcription result as a DataFrame.

        A ``speaker`` column is included only when the segments carry two or
        more distinct speaker labels; a single detected speaker (or none)
        yields a clean, speaker-free transcript.

        Returns:
            pd.DataFrame: A DataFrame containing the transcription results,
            including a speaker column when ≥2 distinct speakers were labeled.

        Raises:
            ValueError: If transcription has not been run yet.
        """
        if self.transcription_result is None or "segments" not in self.transcription_result:
            raise ValueError("Transcription result is not available. Run transcription first.")

        segments = self.transcription_result["segments"]
        speakers_present = {str(item["speaker"]) for item in segments if item.get("speaker")}
        has_speaker = len(speakers_present) >= 2

        rows = []
        for item in segments:
            row = [
                _seconds_to_time(item["start"]),
                _seconds_to_time(item["end"]),
            ]
            if has_speaker:
                row.append(item.get("speaker", "Unknown"))
            row.append(item["text"])
            rows.append(row)

        columns: list[str] = [self.start_column, self.end_column]
        if has_speaker:
            columns.append(self.speaker_column)
        columns.append(self.text_column)

        df = pd.DataFrame(rows, columns=pd.Index(columns))
        return _merge_transcriptions_by_sentence(
            df,
            self.start_column,
            self.end_column,
            self.speaker_column,
            self.text_column,
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_transcription.py -v`
Expected: PASS (new request test green; deleted-code tests gone; remaining tests unaffected).

- [ ] **Step 5: Commit**

```bash
git add nextext/core/transcription.py tests/test_transcription.py
git commit -m "$(cat <<'EOF'
refactor: request word timestamps, capture words, drop duplicated diarization code

Ask Whisper for segment+word granularity and store the words alongside the
segments. Remove ExternalWhisperTranscriber.diarization() and _assign_speakers
(dead — the pipeline diarizes via nextext.core.diarization). transcript_output
now shows the speaker column only when ≥2 distinct speakers are present, so a
single detected speaker yields a clean transcript.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Backend contract switch — `diarize` boolean end-to-end (schema, pipeline, jobs, CLI)

**Files:**
- Modify: `nextext/api/schemas.py:33` (`JobOptions`)
- Modify: `nextext/pipeline.py:10,20-73` (`transcription_pipeline`)
- Modify: `nextext/core/transcription.py` (`ExternalWhisperTranscriber.__init__` — remove `n_speakers`)
- Modify: `nextext/api/jobs.py:311,355-359`
- Modify: `nextext/cli.py:90-97,299-305`
- Test: `tests/test_pipeline.py:110-162`; the option-dict fixtures carrying `"speakers": 1,` (`tests/test_cli.py:43`, `tests/test_api/test_jobs.py:56,87,137,170`, and any others the sweep in Step 6 finds); schema test in `tests/test_api/test_job_options_keyframes.py`; CLI test in `tests/test_cli.py`.

**Interfaces:**
- Consumes: `canonicalize_speaker_labels`, `build_speaker_segments`, `diarize_file` (Tasks 2-3), and `transcription_result["words"]` (Task 4).
- Produces: `JobOptions.diarize: bool = True` (no `speakers` field). `transcription_pipeline(file_path: Path, src_lang: str, diarize: bool) -> tuple[pd.DataFrame, str]`. `ExternalWhisperTranscriber.__init__` no longer accepts `n_speakers`.

- [ ] **Step 1: Update the pipeline test for the new signature + word-level path**

Replace the diarization assertions in `tests/test_pipeline.py` (the `DummyTranscriber`-based test, lines ~110-162). The dummy must expose `transcription_result` with `segments` + `words`, and the test asserts no bounds are sent and that `build_speaker_segments` is called:

```python
    class DummyTranscriber:
        instance: "DummyTranscriber"

        def __init__(self, **params: Any) -> None:
            DummyTranscriber.instance = self
            self.params = params
            self.transcription_called = False
            self.src_lang = "fr"
            self.transcription_result: dict[str, Any] = {
                "segments": [{"start": 0.0, "end": 1.0, "text": "bonjour"}],
                "words": [{"word": "bonjour", "start": 0.0, "end": 1.0}],
            }

        def transcription(self) -> None:
            self.transcription_called = True

        def transcript_output(self) -> pd.DataFrame:
            return pd.DataFrame({"text": ["bonjour"]})

    diarize_calls: dict[str, Any] = {}

    def fake_diarize_file(file_path: Path, **kwargs: Any) -> list[dict[str, Any]]:
        diarize_calls.update(kwargs)
        diarize_calls["called"] = True
        return [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]

    build_calls: list[Any] = []

    def fake_build(segments: list[dict[str, Any]], words: list[dict[str, Any]], turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
        build_calls.append((segments, words, turns))
        return segments

    monkeypatch.setattr(pipeline, "ExternalWhisperTranscriber", DummyTranscriber)
    monkeypatch.setattr(pipeline, "diarize_file", fake_diarize_file)
    monkeypatch.setattr(pipeline, "canonicalize_speaker_labels", lambda turns: turns)
    monkeypatch.setattr(pipeline, "build_speaker_segments", fake_build)

    df, detected_lang = pipeline.transcription_pipeline(
        file_path=Path("/tmp/audio.wav"),
        src_lang="auto",
        diarize=True,
    )

    instance = DummyTranscriber.instance
    assert "n_speakers" not in instance.params  # no speaker-count is threaded any more
    assert instance.params["model_id"] == "test-model"
    assert instance.transcription_called is True
    assert diarize_calls.get("called") is True
    assert "max_speakers" not in diarize_calls and "num_speakers" not in diarize_calls  # auto-detect
    assert len(build_calls) == 1
    assert list(df["text"]) == ["bonjour"]
    assert detected_lang == "fr"
```

Update the single-speaker/bypass test below it (originally `test_transcription_pipeline_falls_back_to_original_language`, lines 165+) to call with `diarize=False` and assert `diarize_calls.get("called")` is falsy (diarization skipped). Keep its language-fallback assertions.

- [ ] **Step 2: Run the pipeline test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -k transcription_pipeline -v`
Expected: FAIL — `TypeError: transcription_pipeline() got an unexpected keyword argument 'diarize'` (still `n_speakers`).

- [ ] **Step 3: Update `pipeline.py`**

Change the import (line 10) and `transcription_pipeline` (lines 20-73):

```python
from nextext.core.diarization import build_speaker_segments, canonicalize_speaker_labels, diarize_file
```

```python
def transcription_pipeline(
    file_path: Path,
    src_lang: str,
    diarize: bool,
) -> tuple[pd.DataFrame, str]:
    """Transcribe the audio file via the external Whisper API, optionally diarized.

    The audio always goes to an OpenAI-compatible ``/v1/audio/transcriptions``
    endpoint resolved by :func:`nextext.utils.env_cfg.load_whisper_env`;
    Nextext ships no local Whisper. Whisper always transcribes in the source
    language — translation to a target language is handled separately by
    :func:`translation_pipeline`.

    When ``diarize`` is true and the transcript is non-empty, the audio is sent
    to the ``/diarize`` service with **no** speaker bounds (pyannote estimates
    the count). The returned turns are relabeled to contiguous ``Speaker N`` by
    first appearance and aligned onto the transcript at the word level, so a
    Whisper segment spanning a speaker change is split at the exact word. It
    degrades to segment-level alignment when the endpoint returns no words, and
    to an unlabelled transcript when ``DIARIZE_API_BASE`` is unset or the
    service is unreachable. Diarization is skipped for ``diarize=False`` and for
    empty transcripts.

    Args:
        file_path (Path): Path to the audio file.
        src_lang (str): Source language code.
        diarize (bool): Whether to run speaker diarization.

    Returns:
        tuple[pd.DataFrame, str]: The transcript DataFrame and the
            resolved source language code.
    """
    config = load_whisper_env()
    transcriber = ExternalWhisperTranscriber(
        file_path=file_path,
        src_lang=src_lang,
        model_id=config.model,
    )
    transcriber.transcription()

    result = transcriber.transcription_result or {}
    segments: list[dict[str, Any]] = result.get("segments", [])
    words: list[dict[str, Any]] = result.get("words", [])
    if diarize and segments:
        turns = canonicalize_speaker_labels(diarize_file(file_path))
        if turns:
            transcriber.transcription_result["segments"] = build_speaker_segments(segments, words, turns)

    df = transcriber.transcript_output()
    updated_src_lang = transcriber.src_lang or src_lang
    return df, updated_src_lang
```

- [ ] **Step 4: Remove `n_speakers` from `ExternalWhisperTranscriber.__init__`**

In `nextext/core/transcription.py`, delete the `n_speakers` parameter (line 250), its assignment (`self.n_speakers = n_speakers`, line 275), the `n_speakers` Args/Attributes docstring lines, and the `diarization()` mention in the class docstring's Methods list. The constructor keeps `file_path`, `trg_lang`, `src_lang`, `model_id`, `task`, and the column-name params.

- [ ] **Step 5: Update `schemas.py`, `jobs.py`, `cli.py`**

`nextext/api/schemas.py` — replace line 33:

```python
    diarize: bool = True
```

(remove `speakers: int = Field(default=1, ge=1, le=10)`).

`nextext/api/jobs.py` — line 311 in `file_opts`:

```python
        "diarize": opts.diarize,
```

and the `transcription_pipeline` call (lines 355-359):

```python
    df, updated_src_lang = transcription_pipeline(
        file_path=state.file_path,
        src_lang=file_opts["src_lang"] or "",
        diarize=file_opts["diarize"],
    )
```

`nextext/cli.py` — replace the `--speakers` argument (lines 90-97):

```python
    parser.add_argument(
        "--diarize",
        dest="diarize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detect and label speakers (default: on). Use --no-diarize to skip.",
    )
```

Ensure `import argparse` is present at the top of `cli.py` (it is — it defines the parser). Update the pipeline call (lines 301-305):

```python
        transcript_df, updated_src_lang = transcription_pipeline(
            file_path=args.file_path,
            src_lang=args.src_lang,
            diarize=args.diarize,
        )
```

- [ ] **Step 6: Sweep the option-dict fixtures, then add schema + CLI tests**

Every test that builds a job-options dict carries `"speakers": 1,` and will now fail against `extra="forbid"`. Rewrite them all in one pass:

Run: `grep -rl '"speakers": 1,' tests/ | xargs sed -i '' 's/"speakers": 1,/"diarize": True,/g'`
Then verify none remain: `grep -rn '"speakers"' tests/`
Expected: no matches. (macOS `sed` needs the empty `-i ''`; the known hits are `tests/test_cli.py:43` and `tests/test_api/test_jobs.py:56,87,137,170`.)

Add the schema assertion to `tests/test_api/test_job_options_keyframes.py` (which already exercises `JobOptions`):

```python
def test_job_options_diarize_defaults_on_and_rejects_removed_speakers() -> None:
    """diarize defaults to True; the removed speakers field is rejected."""
    import pytest
    from pydantic import ValidationError
    from nextext.api.schemas import JobOptions

    assert JobOptions().diarize is True
    with pytest.raises(ValidationError):
        JobOptions(speakers=2)
```

Add the CLI test to `tests/test_cli.py` using the module's actual factory, `parse_arguments(args_list)`:

```python
def test_cli_diarize_defaults_on_and_can_be_disabled() -> None:
    """--diarize defaults True; --no-diarize turns it off."""
    from nextext.cli import parse_arguments

    assert parse_arguments(["-f", "x.wav"]).diarize is True
    assert parse_arguments(["-f", "x.wav", "--no-diarize"]).diarize is False
```

- [ ] **Step 7: Run the affected suites to verify they pass**

Run: `uv run pytest tests/test_pipeline.py tests/test_transcription.py tests/test_diarization.py -v` and the schema/CLI tests you touched.
Expected: PASS.

- [ ] **Step 8: Run the full backend suite**

Run: `uv run pytest`
Expected: PASS. Fix any remaining reference to `speakers`/`n_speakers` the grep in Step 9 surfaces.

- [ ] **Step 9: Grep for stragglers**

Run: `grep -rn "n_speakers\|opts.speakers\|\"speakers\"\|--speakers" nextext/ tests/`
Expected: no matches (all migrated). Fix any that remain, re-run `uv run pytest`.

- [ ] **Step 10: Commit**

```bash
git add nextext/ tests/
git commit -m "$(cat <<'EOF'
feat: always-on auto-detecting diarization via a bypassable diarize flag

Replace the per-job max-speakers count (speakers: int, 1-10) with diarize:
bool (default on). When on, call /diarize with no bounds so pyannote
auto-detects the count, canonicalize the labels, and assign speakers at the
word level. --no-diarize (CLI) / diarize=false (API) bypasses. Removes the
now-unused n_speakers plumbing from the transcriber and pipeline.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Frontend — "Detect speakers" checkbox

**Files:**
- Modify: `frontend/src/api/types.ts:17`
- Modify: `frontend/src/components/upload/UploadForm.tsx:23,33-41,79-82`
- Modify (fixtures): `frontend/src/api/jobs.test.ts:11`, `frontend/src/components/results/ResultPanel.test.tsx:27`
- Test: `frontend/src/components/upload/UploadForm.test.tsx`

**Interfaces:**
- Produces: `JobOptions.diarize: boolean` (replaces `speakers: number`); `UploadForm` submits `diarize` from a checkbox defaulting to checked.

- [ ] **Step 1: Write the failing test**

Append to `frontend/src/components/upload/UploadForm.test.tsx`:

```tsx
describe('UploadForm diarize toggle', () => {
  it('submits diarize=true by default and false when unchecked', () => {
    const onRun = vi.fn()
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const { container } = render(
      <QueryClientProvider client={qc}>
        <UploadForm pending={false} onRun={onRun} />
      </QueryClientProvider>,
    )
    addFiles(container, [audio('a.mp3')])

    const toggle = screen.getByLabelText('Detect speakers') as HTMLInputElement
    expect(toggle.checked).toBe(true)

    fireEvent.click(screen.getByRole('button', { name: /Run/ }))
    expect(onRun.mock.calls[0][1]).toMatchObject({ diarize: true })

    fireEvent.click(toggle)
    fireEvent.click(screen.getByRole('button', { name: /Run/ }))
    expect(onRun.mock.calls[1][1]).toMatchObject({ diarize: false })
  })
})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd frontend && pnpm exec vitest run src/components/upload/UploadForm.test.tsx`
Expected: FAIL — `Unable to find a label with the text of: Detect speakers`.

- [ ] **Step 3: Update the type**

`frontend/src/api/types.ts` line 17 — in `JobOptions`, replace `speakers: number` with:

```ts
  diarize: boolean
```

- [ ] **Step 4: Update `UploadForm.tsx`**

Replace the state hook (line 23):

```tsx
  const [diarize, setDiarize] = useState<boolean>(true)
```

Replace the `speakers` key in the `onRun` options object (lines 33-41) with `diarize`:

```tsx
    onRun(files, {
      src_lang: srcLang || null,
      trg_lang: effectiveTrgLang,
      task,
      diarize,
      words,
      summarization,
      hate_speech: hateSpeech,
    })
```

Remove the "Max speakers" number input `<label>` (lines 79-82) and add a "Detect speakers" checkbox to the checkbox row (lines 102-106), as the first control:

```tsx
      <div className="flex gap-4 text-sm">
        <label className="flex items-center gap-2"><input type="checkbox" checked={diarize} onChange={(e) => setDiarize(e.target.checked)} /> Detect speakers</label>
        <label className="flex items-center gap-2"><input type="checkbox" checked={words} onChange={(e) => setWords(e.target.checked)} /> Word analysis</label>
        <label className="flex items-center gap-2"><input type="checkbox" checked={summarization} onChange={(e) => setSummarization(e.target.checked)} /> Summary</label>
        <label className="flex items-center gap-2"><input type="checkbox" checked={hateSpeech} onChange={(e) => setHateSpeech(e.target.checked)} /> Hate speech</label>
      </div>
```

(`getByLabelText('Detect speakers')` resolves because the `<input>` is nested inside its `<label>`.)

- [ ] **Step 5: Update the two fixtures**

`frontend/src/api/jobs.test.ts` line 11 — in `OPTS`, replace `speakers: 1,` with `diarize: true,`.
`frontend/src/components/results/ResultPanel.test.tsx` line 27 — in `options`, replace `speakers: 1,` with `diarize: true,`.

- [ ] **Step 6: Run the frontend suite to verify it passes**

Run: `cd frontend && pnpm test`
Expected: PASS (new toggle test + updated fixtures; typecheck clean — no `speakers` remains).

- [ ] **Step 7: Verify no `speakers` references remain in the frontend**

Run: `cd frontend && grep -rn "speakers" src/`
Expected: no matches. Fix any straggler, re-run `pnpm test`.

- [ ] **Step 8: Commit**

```bash
git add frontend/src
git commit -m "$(cat <<'EOF'
feat(frontend): replace max-speakers input with a Detect speakers toggle

diarize: boolean (default on) replaces speakers: number end-to-end in the
SPA; the upload form submits a checkbox instead of a speaker count.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Docs

**Files:**
- Modify: `CLAUDE.md` (the diarization/pipeline-flow + env sections)
- Verify: `docs/superpowers/specs/2026-07-10-diarization-always-on-wordlevel-design.md` (already committed; no change expected)

**Interfaces:** none (documentation only).

- [ ] **Step 1: Update `CLAUDE.md`**

In the pipeline-flow section (the "Transcription (always-on)" bullet and the diarization mentions), replace the "optional speaker diarization … when `max speakers > 1`" wording with: diarization is **on by default and auto-detecting**, runs with no speaker bounds, is bypassable per job (`diarize=false` / CLI `--no-diarize`), relabels speakers to contiguous `Speaker N` by first appearance, aligns at the word level (segment-level fallback when the endpoint returns no words), and hides the speaker column when ≤1 speaker is detected.

In the `DIARIZE_API_BASE` env description, remove "Diarization runs only when `max speakers > 1`." and replace with "Diarization runs by default for every job (auto-detecting the speaker count) unless the job sets `diarize=false`." Leave `DIARIZE_TIMEOUT` unchanged. Note that Nextext requests word timestamps (`timestamp_granularities=["segment","word"]`) and degrades gracefully when the ASR endpoint returns none.

- [ ] **Step 2: Verify docstring/lint gate**

Run: `uv run pre-commit run --all-files`
Expected: PASS (ruff + pyrefly + docstring checks across the touched Python files).

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs: describe always-on auto-detecting word-level diarization

Update CLAUDE.md's pipeline-flow and DIARIZE_API_BASE sections for the
diarize boolean (default on, bypassable), auto-detection, canonical
Speaker N labels, word-level alignment with segment-level fallback, and
hide-if-single-speaker.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Final verification (before opening PRs)

- [ ] **Backend + frontend gate**

Run: `uv run pytest && cd frontend && pnpm test && cd .. && uv run pre-commit run --all-files && make verify`
Expected: all green. (`make verify` runs pre-commit + the frontend `pnpm lint` + `pnpm build`.)

- [ ] **Two PRs**
  - `../vllm-service` branch `feature/asr-word-timestamps` (Task 1) — its own PR.
  - Nextext branch `feature/diarization-always-on-wordlevel` (Tasks 2-7) — its own PR; note in the description that the dialogue-precision win requires the vllm-service PR deployed, and that Nextext degrades to segment-level until then.

---

## Self-Review

**Spec coverage:**
- Always-on auto-detect (no bounds) → Task 5 (pipeline calls `diarize_file` with no bounds). ✓
- Bypassable, default on → Task 5 (`diarize: bool = True`, CLI `--no-diarize`), Task 6 (checkbox). ✓
- Canonical `Speaker N` by first appearance → Task 2. ✓
- Word-level segment-anchored splitting + segment-level fallback → Task 3, wired in Task 5. ✓
- ASR word timestamps (backward-compatible) → Task 1. ✓
- Hide column when ≤1 distinct speaker → Task 4 (`transcript_output`). ✓
- Remove duplicated overlap logic (`_assign_speakers`, `.diarization()`) → Task 4. ✓
- Schema/jobs/CLI/frontend contract switch → Tasks 5-6. ✓
- Skip on empty transcript / fail-soft → preserved in Task 5's `if diarize and segments` + `if turns`. ✓
- Docs → Task 7. ✓
- Tests across diarization/pipeline/transcription/schema/CLI/frontend/ASR-smoke → each task. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code; commands have expected output. ✓

**Type consistency:** `build_speaker_segments(segments, words, turns)`, `canonicalize_speaker_labels(turns)`, `_speaker_by_overlap(start, end, turns)`, `transcription_pipeline(file_path, src_lang, diarize)`, `JobOptions.diarize: bool`, `JobOptions`/types `diarize: boolean` — names and signatures match across Tasks 2-6. `transcription_result` shape `{"segments","words"}` produced in Task 4 and consumed in Task 5. ✓
