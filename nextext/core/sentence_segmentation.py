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

import re
from typing import Any

from loguru import logger

from nextext.core.diarization import _speaker_by_overlap
from nextext.core.openai_cfg import InferencePipeline
from nextext.core.transcription import _ends_with_punctuation

_TERMINAL_MARKS: tuple[str, ...] = (".", "!", "?", "؟", "۔", "…")  # noqa: RUF001 - Arabic marks


def terminal_punctuation_ratio(text: str) -> float:
    """Report the terminal-punctuation density of a transcript.

    Density is the count of sentence-terminal marks (the ASCII ``. ! ?`` plus
    their Arabic and ellipsis equivalents, as listed in ``_TERMINAL_MARKS``)
    divided by the whitespace-delimited word count — a language-agnostic proxy
    for whether the text carries sentence boundaries. Used to gate restoration.

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


_TYPE_TO_MARK: dict[str, str] = {"S": ".", "Q": "؟", "E": "!"}
_DEFAULT_MARK: str = "."
_SEGMENT_WORD_BUDGET: int = 400
# Must comfortably exceed the number of "index:code" pairs a _SEGMENT_WORD_BUDGET-word window can produce.
_SEGMENT_MAX_TOKENS: int = 512


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
    for index_text, code in re.findall(r"(-?\d+)\s*:\s*([A-Za-z])", reply):
        index = int(index_text)
        if not 0 <= index < window_len:
            continue
        marks_by_index.setdefault(index, _TYPE_TO_MARK.get(code.upper(), _DEFAULT_MARK))
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
            local: list[tuple[int, str]] = []
        for index, mark in local:
            marks_by_index.setdefault(start + index, mark)
        marks_by_index.setdefault(start + window_len - 1, _DEFAULT_MARK)
    return sorted(marks_by_index.items())


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
                result.append(_build_sentence(sentence_words, speaker, mark))
            previous = end_index + 1
    return result
