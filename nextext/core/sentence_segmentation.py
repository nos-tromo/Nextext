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
