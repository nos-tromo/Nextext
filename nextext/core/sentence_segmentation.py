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

from nextext.core.openai_cfg import InferencePipeline

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
            local: list[tuple[int, str]] = []
        for index, mark in local:
            marks_by_index.setdefault(start + index, mark)
        marks_by_index.setdefault(start + window_len - 1, _DEFAULT_MARK)
    return sorted(marks_by_index.items())
