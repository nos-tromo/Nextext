# Sentence Restoration for Low-Punctuation Transcripts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For punctuation-poor transcripts (e.g. Arabic), recover real sentence boundaries from Whisper word timestamps so each transcript row is one sentence — granular *and* a coherent translation unit — instead of a whole-speaker-turn blob.

**Architecture:** A new stateless agent (`nextext/core/sentence_segmentation.py`) asks `TEXT_MODEL` (via the existing `InferencePipeline`) to classify sentence boundaries over the contiguous word stream, returning `index:code` pairs (never text) so words/timestamps stay model-untouched. It emits one segment per sentence with a restored terminal mark (`.`/`؟`/`!`), respecting speaker runs, then feeds the existing merge/translate path unchanged. It runs whenever a transcript's terminal-punctuation density is below a threshold (language-agnostic), gated by an env knob (default on), and fails soft to today's behavior.

**Tech Stack:** Python 3.12, pandas, `openai` SDK (via `InferencePipeline`), loguru, pytest + monkeypatch. No new dependencies.

## Global Constraints

- Python 3.12 target; **Google-style docstrings on every new/modified function**; explicit type hints; distinct variable names across branches (pyrefly).
- **No-torch invariant** (`tests/test_no_torch.py`): the new agent is pure stdlib + existing deps (HTTP via `InferencePipeline`); import nothing that pulls torch/transformers.
- All model inference is external HTTP through `InferencePipeline` / `TEXT_MODEL`; ship no local models.
- **Fail-soft** like NER / diarization / summary — never crash a job; degrade to today's behavior and `logger.warning`.
- Env config lives only in `nextext/utils/env_cfg.py` dataclasses; follow the `load_diarize_vad_gate_env` pattern (`_parse_tristate_bool` for on/off, validated-numeric-with-warn-and-fallback).
- **DRY:** reuse `_speaker_by_overlap` (`diarization.py`), `_ends_with_punctuation` (`transcription.py`), and the existing `_merge_transcriptions_by_sentence` — do not reimplement.
- Verify before done: `uv run pytest` (full suite) and `uv run pre-commit run --all-files` (ruff + pyrefly). `make verify` is the pre-push gate.
- **Arabic characters in source** (e.g. `؟`, `۔` in code or test assertions) trip ruff `RUF001` (ambiguous unicode). Append `# noqa: RUF001` on each such line, exactly as `nextext/core/transcription.py:112` already does. Run `uv run ruff check` after adding Arabic literals to catch any missed line.
- Commits: small, topical, conventional prefixes (`feat:`/`test:`/`docs:`).

## File Structure

- Create `nextext/core/sentence_segmentation.py` — the agent: `terminal_punctuation_ratio` (gate primitive), `restore_sentence_segments` (public), `_segment_run` / `_speaker_runs` / `_build_sentence` / `_parse_boundaries` (private), `_TYPE_TO_MARK` + constants.
- Create `nextext/utils/prompts/en/sentence_segment.txt` — the segmentation prompt.
- Modify `nextext/utils/env_cfg.py` — add `SentenceRestoreConfig` + `load_sentence_restore_env` + default constant.
- Modify `nextext/pipeline.py` — wire restoration into `transcription_pipeline`; add imports.
- Create `tests/test_sentence_segmentation.py` — agent unit tests.
- Modify `tests/test_env_cfg.py` — loader tests.
- Modify `tests/test_pipeline.py` — restore wiring tests; pin restore off in the existing diarization test.
- Modify `tests/test_transcription.py` — confirm restored marks split into one row each.
- Modify docs: `.env.example`, `CLAUDE.md`, `AGENTS.md`.

---

### Task 1: Env config — `load_sentence_restore_env`

**Files:**
- Modify: `nextext/utils/env_cfg.py` (add constant, dataclass, loader — place next to `DiarizeVadGateConfig` / `load_diarize_vad_gate_env`)
- Modify: `.env.example` (document the two new vars)
- Test: `tests/test_env_cfg.py`

**Interfaces:**
- Produces: `SentenceRestoreConfig(enabled: bool, min_punct_ratio: float)` and `load_sentence_restore_env() -> SentenceRestoreConfig`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_env_cfg.py` (add `load_sentence_restore_env` to the existing `from nextext.utils.env_cfg import (...)` block):

```python
def test_load_sentence_restore_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset vars → enabled with the 0.01 default ratio."""
    monkeypatch.delenv("NEXTEXT_SENTENCE_RESTORE", raising=False)
    monkeypatch.delenv("SENTENCE_RESTORE_MIN_PUNCT_RATIO", raising=False)
    cfg = load_sentence_restore_env()
    assert cfg.enabled is True
    assert cfg.min_punct_ratio == 0.01


def test_load_sentence_restore_env_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit falsy token disables restoration."""
    monkeypatch.setenv("NEXTEXT_SENTENCE_RESTORE", "off")
    assert load_sentence_restore_env().enabled is False


def test_load_sentence_restore_env_custom_ratio(monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid in-range ratio is honoured."""
    monkeypatch.delenv("NEXTEXT_SENTENCE_RESTORE", raising=False)
    monkeypatch.setenv("SENTENCE_RESTORE_MIN_PUNCT_RATIO", "0.03")
    assert load_sentence_restore_env().min_punct_ratio == 0.03


def test_load_sentence_restore_env_invalid_ratio_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Out-of-range / non-numeric ratios warn and fall back to the default."""
    monkeypatch.delenv("NEXTEXT_SENTENCE_RESTORE", raising=False)
    monkeypatch.setenv("SENTENCE_RESTORE_MIN_PUNCT_RATIO", "5")
    assert load_sentence_restore_env().min_punct_ratio == 0.01
    monkeypatch.setenv("SENTENCE_RESTORE_MIN_PUNCT_RATIO", "abc")
    assert load_sentence_restore_env().min_punct_ratio == 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_env_cfg.py -k sentence_restore -v`
Expected: FAIL — `ImportError: cannot import name 'load_sentence_restore_env'`.

- [ ] **Step 3: Implement the loader**

In `nextext/utils/env_cfg.py`, add the default constant near the other `DEFAULT_*` values:

```python
DEFAULT_SENTENCE_RESTORE_MIN_PUNCT_RATIO: float = 0.01
```

Add the dataclass next to `DiarizeVadGateConfig`:

```python
@dataclass(frozen=True)
class SentenceRestoreConfig:
    """Dataclass for LLM sentence restoration on low-punctuation transcripts.

    When enabled and a transcript's terminal-punctuation density is below
    ``min_punct_ratio``, each contiguous speaker run is re-segmented into
    sentences by ``TEXT_MODEL`` so downstream rows are whole sentences rather
    than whole-speaker-turn blobs.

    Attributes:
        enabled: Whether restoration may run (``NEXTEXT_SENTENCE_RESTORE``,
            default ``True``).
        min_punct_ratio: Terminal-punctuation-per-word threshold below which a
            transcript is treated as low-punctuation
            (``SENTENCE_RESTORE_MIN_PUNCT_RATIO``, default 0.01).
    """

    enabled: bool
    min_punct_ratio: float
```

Add the loader next to `load_diarize_vad_gate_env`:

```python
def load_sentence_restore_env() -> SentenceRestoreConfig:
    """Loads the sentence-restoration configuration from environment variables.

    Returns:
        SentenceRestoreConfig: the resolved settings.
        - enabled (bool): ``NEXTEXT_SENTENCE_RESTORE`` (default ``True``; only an
          explicit falsy token — ``0``/``false``/``no``/``off`` — disables it;
          unrecognised values warn and keep the default).
        - min_punct_ratio (float): ``SENTENCE_RESTORE_MIN_PUNCT_RATIO``; values
          outside ``(0, 1)`` or non-numeric warn and fall back to
          :data:`DEFAULT_SENTENCE_RESTORE_MIN_PUNCT_RATIO`.
    """
    parsed = _parse_tristate_bool("NEXTEXT_SENTENCE_RESTORE")
    enabled = True if parsed is None else parsed

    min_punct_ratio = DEFAULT_SENTENCE_RESTORE_MIN_PUNCT_RATIO
    raw_ratio = os.getenv("SENTENCE_RESTORE_MIN_PUNCT_RATIO", "").strip()
    if raw_ratio:
        try:
            value = float(raw_ratio)
            if not 0.0 < value < 1.0:
                raise ValueError
            min_punct_ratio = value
        except ValueError:
            logger.warning(
                "Invalid SENTENCE_RESTORE_MIN_PUNCT_RATIO '{}'. Falling back to {}.",
                raw_ratio,
                DEFAULT_SENTENCE_RESTORE_MIN_PUNCT_RATIO,
            )

    return SentenceRestoreConfig(enabled=enabled, min_punct_ratio=min_punct_ratio)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_env_cfg.py -k sentence_restore -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Document the env vars in `.env.example`**

Add, next to the `NEXTEXT_DIARIZE_VAD_GATE` block:

```bash
# --- Sentence restoration (low-punctuation transcripts) ---
# When a transcript's terminal-punctuation density is below the threshold
# (e.g. Whisper Arabic output), TEXT_MODEL re-segments it into whole sentences
# so each row is one sentence (granular + coherent for translation). Default on.
NEXTEXT_SENTENCE_RESTORE=on
# Terminal marks per word below which a transcript counts as low-punctuation.
# Punctuated EN/DE sit ~0.05-0.07; unpunctuated Arabic ~0. Default 0.01.
SENTENCE_RESTORE_MIN_PUNCT_RATIO=0.01
```

- [ ] **Step 6: Commit**

```bash
git add nextext/utils/env_cfg.py tests/test_env_cfg.py .env.example
git commit -m "feat(config): NEXTEXT_SENTENCE_RESTORE + SENTENCE_RESTORE_MIN_PUNCT_RATIO"
```

---

### Task 2: Gate primitive — `terminal_punctuation_ratio`

**Files:**
- Create: `nextext/core/sentence_segmentation.py`
- Test: `tests/test_sentence_segmentation.py`

**Interfaces:**
- Produces: `terminal_punctuation_ratio(text: str) -> float` — count of terminal marks (`. ! ? ؟ ۔ …`) ÷ word count; `0.0` for empty/whitespace text.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sentence_segmentation.py`:

```python
"""Tests for the sentence-restoration agent."""

from typing import Any

from nextext.core.sentence_segmentation import terminal_punctuation_ratio


def test_terminal_punctuation_ratio_high_for_punctuated_english() -> None:
    """Well-punctuated prose scores well above the 0.01 gate."""
    assert terminal_punctuation_ratio("Hello there. How are you? Fine!") > 0.05


def test_terminal_punctuation_ratio_zero_for_unpunctuated_arabic() -> None:
    """Unpunctuated Arabic scores 0.0."""
    text = "وصل وزير الخارجية إلى تل أبيب لإجراء محادثات مع المسؤولين"
    assert terminal_punctuation_ratio(text) == 0.0


def test_terminal_punctuation_ratio_empty_is_zero() -> None:
    """Whitespace-only text yields 0.0, not a division error."""
    assert terminal_punctuation_ratio("   ") == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sentence_segmentation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nextext.core.sentence_segmentation'`.

- [ ] **Step 3: Create the module with the gate primitive**

Create `nextext/core/sentence_segmentation.py`:

```python
"""Sentence-restoration agent: recover sentence boundaries for low-punctuation transcripts.

Whisper transcribes some scripts (notably Arabic) with essentially no terminal
punctuation, so ``_merge_transcriptions_by_sentence`` cannot find sentence
boundaries and rows grow to whole speaker turns. This agent asks ``TEXT_MODEL``
(via :class:`nextext.core.openai_cfg.InferencePipeline`) to classify sentence
boundaries over the contiguous word stream, returning ``index:code`` pairs
(never text), and rebuilds the segment list so each segment is one sentence with
a restored terminal mark. It is fail-soft: any failure degrades to emitting the
run as a single segment.
"""

from typing import Any

from loguru import logger

_TERMINAL_MARKS: tuple[str, ...] = (".", "!", "?", "؟", "۔", "…")  # noqa: RUF001 - Arabic marks


def terminal_punctuation_ratio(text: str) -> float:
    """Report the terminal-punctuation density of a transcript.

    Density is the count of sentence-terminal marks (``. ! ? ؟ ۔ …``) divided by
    the whitespace-delimited word count — a language-agnostic proxy for "does
    this text carry sentence boundaries". Used to gate restoration.

    Args:
        text (str): The transcript text to measure.

    Returns:
        float: Marks-per-word ratio; ``0.0`` for empty or whitespace-only text.
    """
    words = text.split()
    if not words:
        return 0.0
    marks = sum(text.count(mark) for mark in _TERMINAL_MARKS)
    return marks / len(words)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sentence_segmentation.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add nextext/core/sentence_segmentation.py tests/test_sentence_segmentation.py
git commit -m "feat(sentence-restore): terminal_punctuation_ratio gate primitive"
```

---

### Task 3: Boundary segmentation — `_segment_run` + prompt

**Files:**
- Modify: `nextext/core/sentence_segmentation.py`
- Create: `nextext/utils/prompts/en/sentence_segment.txt`
- Test: `tests/test_sentence_segmentation.py`

**Interfaces:**
- Consumes: `InferencePipeline.load_prompt("sentence_segment")` + `.call_model(prompt, include_system_prompt, temperature, num_predict)`.
- Produces: `_segment_run(run_words, inference_pipeline) -> list[tuple[int, str]]` — ascending `(inclusive_end_index, mark)` pairs covering all words, last index always `len(run_words) - 1`. `_TYPE_TO_MARK = {"S": ".", "Q": "؟", "E": "!"}`; module constants `_SEGMENT_WORD_BUDGET = 400`, `_SEGMENT_MAX_TOKENS = 256`, `_DEFAULT_MARK = "."`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_sentence_segmentation.py` (extend the header imports to `from typing import Any, override`, `import pytest`, and the two below). The fake **subclasses `InferencePipeline` with `@override` and the full `call_model` signature** — the repo's `ruff` `D`/`ANN` rules and `pyrefly` apply to test files, and the existing `_RecordingPipeline` establishes this pattern:

```python
import pytest
from typing import override

from nextext.core import sentence_segmentation
from nextext.core.openai_cfg import InferencePipeline
from nextext.core.sentence_segmentation import _segment_run


class _FakePipeline(InferencePipeline):
    """InferencePipeline double returning queued ``index:code`` replies per call."""

    def __init__(self, replies: list[str]) -> None:
        """Store the queued replies.

        Args:
            replies (list[str]): One reply per ``call_model`` invocation; an
                empty queue returns ``""``.
        """
        self.replies = list(replies)
        self.prompts: list[str] = []

    @override
    def load_prompt(self, keyword: str = "system") -> str:
        """Return a passthrough ``{tokens}`` template.

        Args:
            keyword (str): Prompt keyword; expected to be ``"sentence_segment"``.

        Returns:
            str: A template whose only placeholder is ``{tokens}``.
        """
        assert keyword == "sentence_segment"
        return "{tokens}"

    @override
    def call_model(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.1,
        seed: int = 42,
        stop: list[str] | None = None,
        num_predict: int | None = None,
        top_p: float | None = None,
        system_prompt: str | None = None,
        include_system_prompt: bool = True,
        think: bool | None = None,
    ) -> str:
        """Record the prompt and pop the next canned reply.

        Args:
            prompt (str): The rendered prompt.
            model (str | None): Unused test-double argument.
            temperature (float): Unused test-double argument.
            seed (int): Unused test-double argument.
            stop (list[str] | None): Unused test-double argument.
            num_predict (int | None): Unused test-double argument.
            top_p (float | None): Unused test-double argument.
            system_prompt (str | None): Unused test-double argument.
            include_system_prompt (bool): Unused test-double argument.
            think (bool | None): Unused test-double argument.

        Returns:
            str: The next queued reply, or ``""`` when exhausted.
        """
        del model, temperature, seed, stop, num_predict, top_p, system_prompt, include_system_prompt, think
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else ""


def _words(labels: str) -> list[dict[str, Any]]:
    """Build word dicts from single-char labels, one second apart.

    Args:
        labels (str): Characters, one per word.

    Returns:
        list[dict[str, Any]]: Word dicts with ``word``/``start``/``end``.
    """
    return [{"word": ch, "start": float(i), "end": float(i) + 0.5} for i, ch in enumerate(labels)]


def test_segment_run_parses_index_code_and_forces_final_boundary() -> None:
    """Model boundaries are parsed with marks; a final boundary is guaranteed."""
    run = _words("abcdef")  # indices 0..5
    result = _segment_run(run, _FakePipeline(["2:S, 5:Q"]))
    assert result == [(2, "."), (5, "؟")]


def test_segment_run_defaults_unknown_code_to_period() -> None:
    """Unknown / malformed codes fall back to a period."""
    run = _words("abcd")
    result = _segment_run(run, _FakePipeline(["1:Z, 3:X"]))
    assert result == [(1, "."), (3, ".")]


def test_segment_run_failsoft_on_error_is_single_sentence() -> None:
    """A raising call degrades to one boundary at the run end (period)."""

    class _Boom(_FakePipeline):
        """Fake whose ``call_model`` always raises."""

        @override
        def call_model(
            self,
            prompt: str,
            model: str | None = None,
            temperature: float = 0.1,
            seed: int = 42,
            stop: list[str] | None = None,
            num_predict: int | None = None,
            top_p: float | None = None,
            system_prompt: str | None = None,
            include_system_prompt: bool = True,
            think: bool | None = None,
        ) -> str:
            """Raise to simulate a provider outage.

            Args:
                prompt (str): Ignored.
                model (str | None): Ignored.
                temperature (float): Ignored.
                seed (int): Ignored.
                stop (list[str] | None): Ignored.
                num_predict (int | None): Ignored.
                top_p (float | None): Ignored.
                system_prompt (str | None): Ignored.
                include_system_prompt (bool): Ignored.
                think (bool | None): Ignored.

            Raises:
                RuntimeError: Always.
            """
            raise RuntimeError("provider down")

    run = _words("abc")
    assert _segment_run(run, _Boom([])) == [(2, ".")]


def test_segment_run_chunks_and_offsets_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run longer than the budget is windowed; indices are offset per window."""
    monkeypatch.setattr(sentence_segmentation, "_SEGMENT_WORD_BUDGET", 3)
    run = _words("abcdef")  # window0=0..2, window1=3..5
    # window0 reply "1:S" -> (1,'.') + forced end (2,'.'); window1 "1:Q" -> (4,'؟') + forced (5,'.')
    result = _segment_run(run, _FakePipeline(["1:S", "1:Q"]))
    assert result == [(1, "."), (2, "."), (4, "؟"), (5, ".")]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sentence_segmentation.py -k segment_run -v`
Expected: FAIL — `ImportError: cannot import name '_segment_run'`.

- [ ] **Step 3: Create the prompt template**

Create `nextext/utils/prompts/en/sentence_segment.txt`:

```
You segment an unpunctuated speech transcript into sentences.

You are given a numbered list of word tokens, one per line as "<index>\t<word>",
numbered starting at 0. Decide where each sentence ends and its type.

Output ONLY a comma-separated, ascending list of "<index>:<code>" pairs — one
per sentence — where <index> is the index of the sentence's LAST token and
<code> is one of:
- S for a statement
- Q for a question
- E for an exclamation

Rules:
- Output nothing but the pairs, e.g. "4:S, 9:Q, 15:S".
- Never output the words themselves.
- Never add, remove, reorder, or edit any token.
- The last token's index must appear as the final pair.

Tokens:
{tokens}
```

- [ ] **Step 4: Implement `_segment_run` and helpers**

First add these imports to the **top-of-file import block** (imports must stay at the top — ruff `E402`/`I001`; `Any` and `logger` are first used here, so they belong in this task, and ruff `I` will sort them stdlib → third-party → first-party):

```python
from typing import Any

from loguru import logger

from nextext.core.openai_cfg import InferencePipeline
```

Then append the constants and functions to the module body:

```python
_TYPE_TO_MARK: dict[str, str] = {"S": ".", "Q": "؟", "E": "!"}  # noqa: RUF001 - Arabic mark
_DEFAULT_MARK: str = "."
_SEGMENT_WORD_BUDGET: int = 400
_SEGMENT_MAX_TOKENS: int = 256


def _parse_boundaries(reply: str, window_len: int) -> list[tuple[int, str]]:
    """Parse a model reply of ``index:code`` pairs into sanitized boundaries.

    Keeps only pairs whose index is within ``[0, window_len)``; maps the type
    code via :data:`_TYPE_TO_MARK` (unknown/absent → ``.``); dedupes by index
    (first wins) and sorts ascending.

    Args:
        reply (str): The raw model reply, e.g. ``"4:S, 9:Q"``.
        window_len (int): Number of tokens in the window (index upper bound).

    Returns:
        list[tuple[int, str]]: Ascending ``(index, mark)`` pairs; possibly empty.
    """
    marks_by_index: dict[int, str] = {}
    for item in reply.split(","):
        token = item.strip()
        if not token or ":" not in token:
            continue
        index_text, _, code = token.partition(":")
        try:
            index = int(index_text.strip())
        except ValueError:
            continue
        if not 0 <= index < window_len:
            continue
        marks_by_index.setdefault(index, _TYPE_TO_MARK.get(code.strip().upper(), _DEFAULT_MARK))
    return sorted(marks_by_index.items())


def _segment_run(run_words: list[dict[str, Any]], inference_pipeline: InferencePipeline) -> list[tuple[int, str]]:
    """Classify sentence boundaries for one contiguous run of words.

    The run is windowed at :data:`_SEGMENT_WORD_BUDGET`; each window is sent to
    the model as a numbered token list and the reply parsed into boundaries. A
    boundary is forced at every window end (chunk edge / fail-soft), guaranteeing
    the final word terminates a sentence. On any error or empty parse the window
    degrades to a single sentence.

    Args:
        run_words (list[dict[str, Any]]): Words with ``word``/``start``/``end``.
        inference_pipeline (InferencePipeline): Shared inference client.

    Returns:
        list[tuple[int, str]]: Ascending ``(inclusive_end_index, mark)`` pairs
            spanning the run; the last index is ``len(run_words) - 1``.
    """
    marks_by_index: dict[int, str] = {}
    total = len(run_words)
    for start in range(0, total, _SEGMENT_WORD_BUDGET):
        window = run_words[start : start + _SEGMENT_WORD_BUDGET]
        window_len = len(window)
        tokens = "\n".join(f"{i}\t{str(word['word']).strip()}" for i, word in enumerate(window))
        try:
            prompt = inference_pipeline.load_prompt("sentence_segment").format(tokens=tokens)
            reply = inference_pipeline.call_model(
                prompt=prompt,
                include_system_prompt=False,
                temperature=0.0,
                num_predict=_SEGMENT_MAX_TOKENS,
            )
            local = _parse_boundaries(reply, window_len)
        except Exception as exc:  # fail-soft: any provider/parse failure → one sentence
            logger.warning("Sentence segmentation failed for a run window; treating it as one sentence: {}", exc)
            local = []
        for index, mark in local:
            marks_by_index.setdefault(start + index, mark)
        marks_by_index.setdefault(start + window_len - 1, _DEFAULT_MARK)
    return sorted(marks_by_index.items())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_sentence_segmentation.py -k segment_run -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add nextext/core/sentence_segmentation.py nextext/utils/prompts/en/sentence_segment.txt tests/test_sentence_segmentation.py
git commit -m "feat(sentence-restore): _segment_run boundary classification + prompt"
```

---

### Task 4: Public entry — `restore_sentence_segments`

**Files:**
- Modify: `nextext/core/sentence_segmentation.py`
- Test: `tests/test_sentence_segmentation.py`

**Interfaces:**
- Consumes: `_segment_run` (Task 3); `_speaker_by_overlap` (`diarization.py`); `_ends_with_punctuation` (`transcription.py`).
- Produces: `restore_sentence_segments(words, turns, inference_pipeline, *, default_mark=".") -> list[dict]` — sentence-level segments `{start, end, text, [speaker]}`; `[]` when `words` is empty.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_sentence_segmentation.py`:

```python
from nextext.core.sentence_segmentation import restore_sentence_segments


def test_restore_splits_run_into_sentences_with_marks() -> None:
    """Undiarized run → sentences with word-derived times and restored marks."""
    words = _words("abcdef")
    segments = restore_sentence_segments(words, None, _FakePipeline(["2:S, 5:Q"]))
    assert len(segments) == 2
    assert segments[0]["text"] == "a b c."
    assert segments[0]["start"] == 0.0 and segments[0]["end"] == 2.5
    assert segments[1]["text"] == "d e f؟"
    assert "speaker" not in segments[0]


def test_restore_returns_empty_without_words() -> None:
    """No word timestamps → empty result (caller keeps existing segments)."""
    assert restore_sentence_segments([], None, _FakePipeline([])) == []


def test_restore_does_not_double_punctuate() -> None:
    """A sentence already ending in punctuation gets no extra mark."""
    words = [{"word": "hi.", "start": 0.0, "end": 0.5}]
    segments = restore_sentence_segments(words, None, _FakePipeline(["0:S"]))
    assert segments[0]["text"] == "hi."


def test_restore_inherits_speaker_and_splits_on_change() -> None:
    """Words are partitioned into contiguous speaker runs before segmenting."""
    words = [
        {"word": "a", "start": 0.0, "end": 1.0},
        {"word": "b", "start": 1.0, "end": 2.0},
        {"word": "c", "start": 6.0, "end": 7.0},
        {"word": "d", "start": 7.0, "end": 8.0},
    ]
    turns = [
        {"start": 0.0, "end": 5.0, "speaker": "Speaker 1"},
        {"start": 5.0, "end": 10.0, "speaker": "Speaker 2"},
    ]
    # Two runs, each segmented with one canned reply ("1:S" → local end index 1).
    segments = restore_sentence_segments(words, turns, _FakePipeline(["1:S", "1:S"]))
    assert [seg["speaker"] for seg in segments] == ["Speaker 1", "Speaker 2"]
    assert segments[0]["text"] == "a b."
    assert segments[1]["text"] == "c d."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sentence_segmentation.py -k restore -v`
Expected: FAIL — `ImportError: cannot import name 'restore_sentence_segments'`.

- [ ] **Step 3: Implement the public entry + run/sentence helpers**

Append to `nextext/core/sentence_segmentation.py` (add the two reused imports at the top of the file's import block):

```python
from nextext.core.diarization import _speaker_by_overlap
from nextext.core.transcription import _ends_with_punctuation
```

```python
def _speaker_runs(
    words: list[dict[str, Any]],
    turns: list[dict[str, Any]] | None,
) -> list[tuple[str | None, list[dict[str, Any]]]]:
    """Partition words into contiguous same-speaker runs.

    Mirrors :func:`nextext.core.diarization.build_speaker_segments`: a word
    overlapping no turn does not force a split (it joins the surrounding run);
    only a change between two labelled speakers starts a new run. With no turns,
    the whole word list is one unlabelled run.

    Args:
        words (list[dict[str, Any]]): Words with ``start``/``end`` keys.
        turns (list[dict[str, Any]] | None): Canonicalized speaker turns, or None.

    Returns:
        list[tuple[str | None, list[dict[str, Any]]]]: ``(speaker, words)`` runs
            in order.
    """
    if not turns:
        return [(None, list(words))]
    runs: list[tuple[str | None, list[dict[str, Any]]]] = []
    run_words: list[dict[str, Any]] = []
    run_speaker: str | None = None
    for word in words:
        speaker = _speaker_by_overlap(float(word["start"]), float(word["end"]), turns)
        if run_words and speaker is not None and run_speaker is not None and speaker != run_speaker:
            runs.append((run_speaker, run_words))
            run_words = []
            run_speaker = None
        run_words.append(word)
        if speaker is not None:
            run_speaker = speaker
    if run_words:
        runs.append((run_speaker, run_words))
    return runs


def _build_sentence(
    sentence_words: list[dict[str, Any]],
    speaker: str | None,
    mark: str,
) -> dict[str, Any]:
    """Build one sentence segment from a run of words.

    Args:
        sentence_words (list[dict[str, Any]]): Consecutive words with
            ``word``/``start``/``end`` keys (non-empty).
        speaker (str | None): The sentence's speaker label, if any.
        mark (str): Terminal mark to append when the text lacks one.

    Returns:
        dict[str, Any]: A segment with ``start``/``end``/``text`` and an optional
            ``speaker`` key.
    """
    text = " ".join(str(word["word"]).strip() for word in sentence_words).strip()
    if text and mark and not _ends_with_punctuation(text):
        text = f"{text}{mark}"
    segment: dict[str, Any] = {
        "start": float(sentence_words[0]["start"]),
        "end": float(sentence_words[-1]["end"]),
        "text": text,
    }
    if speaker is not None:
        segment["speaker"] = speaker
    return segment


def restore_sentence_segments(
    words: list[dict[str, Any]],
    turns: list[dict[str, Any]] | None,
    inference_pipeline: InferencePipeline,
    *,
    default_mark: str = ".",
) -> list[dict[str, Any]]:
    """Re-segment a transcript into one segment per sentence.

    Each contiguous speaker run (the whole transcript when ``turns`` is None) is
    classified into sentences by :func:`_segment_run`, and one segment is emitted
    per sentence: ``start``/``end`` from the sentence's first/last word, text
    joined from the words with a restored terminal mark, and the run's speaker.
    Fail-soft: with no words it returns ``[]`` (the caller keeps its segments),
    and any model failure degrades a run to a single segment.

    Args:
        words (list[dict[str, Any]]): Whisper words with ``word``/``start``/``end``.
        turns (list[dict[str, Any]] | None): Canonicalized speaker turns, or None
            when undiarized.
        inference_pipeline (InferencePipeline): Shared inference client.
        default_mark (str): Fallback terminal mark for an unclassified boundary.

    Returns:
        list[dict[str, Any]]: Sentence-level segments with ``start``/``end``/
            ``text`` and an optional ``speaker`` key; ``[]`` when ``words`` empty.
    """
    if not words:
        return []
    result: list[dict[str, Any]] = []
    for speaker, run_words in _speaker_runs(words, turns):
        if not run_words:
            continue
        previous = 0
        for end_index, mark in _segment_run(run_words, inference_pipeline):
            sentence_words = run_words[previous : end_index + 1]
            if sentence_words:
                result.append(_build_sentence(sentence_words, speaker, mark or default_mark))
            previous = end_index + 1
    return result
```

- [ ] **Step 4: Run the full agent test file**

Run: `uv run pytest tests/test_sentence_segmentation.py -v`
Expected: PASS (all tests — ratio, `_segment_run`, restore).

- [ ] **Step 5: Commit**

```bash
git add nextext/core/sentence_segmentation.py tests/test_sentence_segmentation.py
git commit -m "feat(sentence-restore): restore_sentence_segments public entry"
```

---

### Task 5: Wire restoration into the pipeline

**Files:**
- Modify: `nextext/pipeline.py:69-83` (imports + `transcription_pipeline` branch)
- Modify: `tests/test_pipeline.py` (pin restore off in the existing diarization test; add wiring tests)
- Modify: `tests/test_transcription.py` (confirm restored marks split into rows)

**Interfaces:**
- Consumes: `load_sentence_restore_env` (Task 1), `terminal_punctuation_ratio` + `restore_sentence_segments` (Tasks 2/4), existing `InferencePipeline`, `build_speaker_segments`.

- [ ] **Step 1: Add imports**

In `nextext/pipeline.py`, add to the `nextext.core` imports:

```python
from nextext.core.sentence_segmentation import restore_sentence_segments, terminal_punctuation_ratio
```

and extend the `env_cfg` import to include `load_sentence_restore_env`:

```python
from nextext.utils.env_cfg import (
    load_diarize_vad_gate_env,
    load_sentence_restore_env,
    load_summary_env,
    load_whisper_env,
)
```

(`InferencePipeline` is already imported.)

- [ ] **Step 2: Replace the diarization block in `transcription_pipeline`**

Replace lines 72-81 (the `if diarize and segments ...: build_speaker_segments` block) with hoisted `turns` + the restore/diarize branch:

```python
    turns: list[dict[str, Any]] = []
    if diarize and segments and transcriber.transcription_result is not None:
        turns = diarize_file(file_path)
        gate = load_diarize_vad_gate_env()
        if turns and gate.enabled:
            vad_intervals = speech_segments(file_path, threshold=gate.threshold, pad_ms=gate.pad_ms)
            if vad_intervals is not None:
                turns = gate_turns_by_vad(turns, vad_intervals)
        turns = canonicalize_speaker_labels(turns)

    if segments and transcriber.transcription_result is not None:
        restore_cfg = load_sentence_restore_env()
        transcript_text = " ".join(str(seg.get("text", "")) for seg in segments)
        low_punctuation = terminal_punctuation_ratio(transcript_text) < restore_cfg.min_punct_ratio
        if restore_cfg.enabled and words and low_punctuation:
            restored = restore_sentence_segments(words, turns or None, InferencePipeline())
            if restored:
                transcriber.transcription_result["segments"] = restored
        elif diarize and turns:
            transcriber.transcription_result["segments"] = build_speaker_segments(segments, words, turns)
```

Rationale: restoration supersedes `build_speaker_segments` on the low-punctuation path (it re-derives the same speakers from `turns`); the diarized well-punctuated path is the `elif`; a diarized low-punctuation job with no word timestamps (`words` empty) falls through to the `elif` and still gets speaker labels. `InferencePipeline()` is constructed only when restoration runs, so pure-transcription jobs never need `TEXT_MODEL`.

- [ ] **Step 3: Pin restore OFF in the existing diarization test**

The existing `test_transcription_pipeline_invokes_transcriber_and_diarizes` uses unpunctuated text (`"bonjour"`) with words present, which would now trigger restoration and bypass `build_speaker_segments`. Keep its intent by disabling restore. Add to its monkeypatches (alongside the others near line 147), and import `SentenceRestoreConfig` at the top of `tests/test_pipeline.py`:

```python
from nextext.utils.env_cfg import SentenceRestoreConfig

monkeypatch.setattr(
    pipeline, "load_sentence_restore_env", lambda: SentenceRestoreConfig(enabled=False, min_punct_ratio=0.01)
)
```

- [ ] **Step 4: Run it to verify the existing test still passes**

Run: `uv run pytest tests/test_pipeline.py::test_transcription_pipeline_invokes_transcriber_and_diarizes -v`
Expected: PASS (restore disabled → `build_speaker_segments` still called once).

- [ ] **Step 5: Write the new wiring tests (failing)**

Add to `tests/test_pipeline.py` (reuse the module's existing `WhisperClientConfig` / `DiarizeVadGateConfig` imports):

```python
class _RestorableTranscriber:
    """Transcriber stand-in with unpunctuated text and word timestamps."""

    def __init__(self, **params: Any) -> None:
        """Seed an unpunctuated one-segment result with words.

        Args:
            **params (Any): Ignored construction params.
        """
        self.src_lang = "ar"
        self.transcription_result: dict[str, Any] = {
            "segments": [{"start": 0.0, "end": 6.0, "text": "a b c d e f"}],
            "words": [{"word": ch, "start": float(i), "end": float(i) + 0.5} for i, ch in enumerate("abcdef")],
        }

    def transcription(self) -> None:
        """No-op stand-in for the Whisper call."""

    def transcript_output(self) -> pd.DataFrame:
        """Return a one-row transcript frame.

        Returns:
            pd.DataFrame: A dummy transcript.
        """
        return pd.DataFrame({"text": ["x"]})


def _install_restorable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wire a restorable transcriber and a no-network InferencePipeline.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setattr(
        pipeline, "load_whisper_env", lambda: WhisperClientConfig(api_base="http://a/v1", api_key="k", model="m")
    )
    monkeypatch.setattr(pipeline, "ExternalWhisperTranscriber", _RestorableTranscriber)
    monkeypatch.setattr(pipeline, "InferencePipeline", lambda: object())


def test_transcription_pipeline_restores_when_low_punctuation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undiarized low-punctuation transcript → restoration runs with turns=None."""
    _install_restorable(monkeypatch)
    monkeypatch.setattr(
        pipeline, "load_sentence_restore_env", lambda: SentenceRestoreConfig(enabled=True, min_punct_ratio=0.01)
    )
    recorded: dict[str, Any] = {}

    def fake_restore(words: list[dict[str, Any]], turns: Any, inference_pipeline: Any) -> list[dict[str, Any]]:
        """Record the call and return one canned sentence segment.

        Args:
            words (list[dict[str, Any]]): Words forwarded by the pipeline.
            turns (Any): Speaker turns (or None) forwarded by the pipeline.
            inference_pipeline (Any): Inference client forwarded by the pipeline.

        Returns:
            list[dict[str, Any]]: A single restored segment.
        """
        recorded["called"] = True
        recorded["turns"] = turns
        return [{"start": 0.0, "end": 6.0, "text": "a b c d e f."}]

    monkeypatch.setattr(pipeline, "restore_sentence_segments", fake_restore)

    pipeline.transcription_pipeline(file_path=Path("/tmp/a.wav"), src_lang="ar", diarize=False)

    assert recorded["called"] is True
    assert recorded["turns"] is None


def test_transcription_pipeline_skips_restore_when_well_punctuated(monkeypatch: pytest.MonkeyPatch) -> None:
    """A punctuated transcript is left alone (ratio above threshold)."""
    _install_restorable(monkeypatch)
    monkeypatch.setattr(
        pipeline, "load_sentence_restore_env", lambda: SentenceRestoreConfig(enabled=True, min_punct_ratio=0.01)
    )

    class _Punctuated(_RestorableTranscriber):
        """Restorable transcriber whose transcript is already punctuated."""

        def __init__(self, **params: Any) -> None:
            """Seed a punctuated segment (words inherited from the base).

            Args:
                **params (Any): Ignored construction params.
            """
            super().__init__(**params)
            self.transcription_result["segments"] = [{"start": 0.0, "end": 6.0, "text": "a. b. c. d. e. f."}]

    monkeypatch.setattr(pipeline, "ExternalWhisperTranscriber", _Punctuated)

    def boom(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        """Fail if restoration is invoked on punctuated text.

        Args:
            *args (Any): Ignored.
            **kwargs (Any): Ignored.

        Returns:
            list[dict[str, Any]]: Never returns.
        """
        pytest.fail("restore_sentence_segments should not run on punctuated text")

    monkeypatch.setattr(pipeline, "restore_sentence_segments", boom)

    pipeline.transcription_pipeline(file_path=Path("/tmp/a.wav"), src_lang="ar", diarize=False)


def test_transcription_pipeline_restore_supersedes_build_when_diarized(monkeypatch: pytest.MonkeyPatch) -> None:
    """Diarized low-punctuation path uses restore (with turns), not build_speaker_segments."""
    _install_restorable(monkeypatch)
    monkeypatch.setattr(
        pipeline, "load_sentence_restore_env", lambda: SentenceRestoreConfig(enabled=True, min_punct_ratio=0.01)
    )
    monkeypatch.setattr(pipeline, "diarize_file", lambda fp: [{"start": 0.0, "end": 6.0, "speaker": "SPEAKER_00"}])
    monkeypatch.setattr(pipeline, "canonicalize_speaker_labels", lambda turns: turns)
    monkeypatch.setattr(
        pipeline, "load_diarize_vad_gate_env", lambda: DiarizeVadGateConfig(enabled=False, threshold=0.4, pad_ms=100)
    )

    def no_build(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        """Fail if build_speaker_segments runs when restoration supersedes it.

        Args:
            *args (Any): Ignored.
            **kwargs (Any): Ignored.

        Returns:
            list[dict[str, Any]]: Never returns.
        """
        pytest.fail("build_speaker_segments should not run when restoration supersedes it")

    monkeypatch.setattr(pipeline, "build_speaker_segments", no_build)
    recorded: dict[str, Any] = {}

    def fake_restore(words: list[dict[str, Any]], turns: Any, inference_pipeline: Any) -> list[dict[str, Any]]:
        """Record the turns passed to restoration and return a canned segment.

        Args:
            words (list[dict[str, Any]]): Words forwarded by the pipeline.
            turns (Any): Canonicalized speaker turns forwarded by the pipeline.
            inference_pipeline (Any): Inference client forwarded by the pipeline.

        Returns:
            list[dict[str, Any]]: A single speaker-labeled restored segment.
        """
        recorded["turns"] = turns
        return [{"start": 0.0, "end": 6.0, "text": "a b c d e f.", "speaker": "SPEAKER_00"}]

    monkeypatch.setattr(pipeline, "restore_sentence_segments", fake_restore)

    pipeline.transcription_pipeline(file_path=Path("/tmp/a.wav"), src_lang="ar", diarize=True)

    assert recorded["turns"] == [{"start": 0.0, "end": 6.0, "speaker": "SPEAKER_00"}]
```

- [ ] **Step 6: Run the new wiring tests**

Run: `uv run pytest tests/test_pipeline.py -k "restore" -v`
Expected: PASS (3 tests).

- [ ] **Step 7: Confirm restored marks split into rows (transcription merge)**

Add to `tests/test_transcription.py`:

```python
def test_merge_splits_restored_sentences_into_rows() -> None:
    """Restored terminal marks make _merge_transcriptions_by_sentence emit one row per sentence."""
    data = pd.DataFrame(
        {
            "start": ["0:00:00", "0:00:03"],
            "end": ["0:00:03", "0:00:06"],
            "text": ["جملة أولى.", "جملة ثانية؟"],  # noqa: RUF001 - Arabic question mark
        }
    )
    merged = _merge_transcriptions_by_sentence(data)
    assert len(merged) == 2
    assert list(merged["text"]) == ["جملة أولى.", "جملة ثانية؟"]  # noqa: RUF001
```

- [ ] **Step 8: Run it**

Run: `uv run pytest tests/test_transcription.py::test_merge_splits_restored_sentences_into_rows -v`
Expected: PASS.

- [ ] **Step 9: Run the full suite**

Run: `uv run pytest`
Expected: PASS (all tests, including the untouched VAD-gating tests which use `words: []`).

- [ ] **Step 10: Commit**

```bash
git add nextext/pipeline.py tests/test_pipeline.py tests/test_transcription.py
git commit -m "feat(pipeline): restore sentence boundaries for low-punctuation transcripts"
```

---

### Task 6: Documentation + final verification

**Files:**
- Modify: `CLAUDE.md`, `AGENTS.md`

- [ ] **Step 1: Update `CLAUDE.md`**

Add to the pipeline flow (step 1, "Transcription") a clause noting sentence restoration, add a module bullet, and add the Environment entries. Module bullet, under "Key modules":

```markdown
- `nextext/core/sentence_segmentation.py` — sentence-restoration agent: for
  low-punctuation transcripts (e.g. Arabic), re-segments the word stream into
  one segment per sentence via `TEXT_MODEL` (`restore_sentence_segments`), which
  returns `index:code` boundaries (never text) and appends the classified
  terminal mark (`.`/`؟`/`!`). Gated on `terminal_punctuation_ratio`; fail-soft.
```

Environment entries (next to the VAD-gate vars):

```markdown
- `NEXTEXT_SENTENCE_RESTORE` / `SENTENCE_RESTORE_MIN_PUNCT_RATIO` (backend + CLI) —
  Sentence restoration for punctuation-poor transcripts. When on (default) and a
  transcript's terminal-punctuation density (marks ÷ words) is below
  `SENTENCE_RESTORE_MIN_PUNCT_RATIO` (default `0.01`), each contiguous speaker
  run is re-segmented into whole sentences by `TEXT_MODEL`, so rows are one
  sentence each (granular and a coherent translation unit) instead of a
  whole-speaker-turn blob. The model returns `index:code` boundaries — never
  text — so words/timestamps stay untouched; questions get `؟`, exclamations
  `!`, else `.`. Fail-soft: a model outage degrades to today's behavior. Resolved
  by `load_sentence_restore_env`. Set `NEXTEXT_SENTENCE_RESTORE=off` to disable.
```

- [ ] **Step 2: Update `AGENTS.md`**

Add a section for the new agent following the existing agent-documentation format, with its I/O contract:

```markdown
### Sentence-segmentation agent (`nextext/core/sentence_segmentation.py`)

- **Input:** Whisper words (`{word, start, end}`), canonicalized speaker turns
  (or `None`), and an `InferencePipeline`.
- **Output:** sentence-level segments `{start, end, text, [speaker]}` — one per
  sentence, with a restored terminal mark (`.`/`؟`/`!`).
- **Contract:** the LLM returns `index:code` pairs (`S`/`Q`/`E`) over a numbered
  token list — never text — so words and timestamps are model-untouched. Runs
  only when `terminal_punctuation_ratio(text) < SENTENCE_RESTORE_MIN_PUNCT_RATIO`
  and `NEXTEXT_SENTENCE_RESTORE` is on. Fail-soft: no words → `[]`; any model
  failure → the run is emitted as a single segment.
```

- [ ] **Step 3: Full verification**

Run: `uv run pytest`
Expected: PASS (entire suite).

Run: `uv run pre-commit run --all-files`
Expected: PASS (ruff check + ruff format + pyrefly). Fix any lint/type findings before committing.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md AGENTS.md
git commit -m "docs: document sentence restoration agent + env vars"
```

---

## Self-Review Notes

- **Spec coverage:** gate primitive (Task 2) ✓; hallucination-proof `index:code` contract (Task 3) ✓; speaker-run partitioning + mark insertion via `_ends_with_punctuation` (Task 4) ✓; pipeline gate + supersede-`build_speaker_segments` + lazy `InferencePipeline` (Task 5) ✓; env knob default-on + ratio validation (Task 1) ✓; error-handling matrix covered by fail-soft tests (Tasks 3–4) ✓; docs (Task 6) ✓.
- **Type consistency:** `_segment_run -> list[tuple[int, str]]` consumed by `restore_sentence_segments`; `terminal_punctuation_ratio(str) -> float` and `load_sentence_restore_env() -> SentenceRestoreConfig(enabled, min_punct_ratio)` used verbatim in the pipeline branch.
- **Regression guard:** the existing VAD-gating pipeline tests use `words: []`, so restoration is skipped and `build_speaker_segments` still runs; the one diarization test with words (`test_transcription_pipeline_invokes_transcriber_and_diarizes`) is explicitly pinned to restore-off in Task 5 Step 3.
```
