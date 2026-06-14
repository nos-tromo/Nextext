"""Tests for ``nextext.core.words.WordCounter`` edge cases."""

from collections import Counter

from nextext.core.words import WordCounter


def test_create_wordcloud_returns_none_when_word_counts_empty() -> None:
    """Empty ``word_counts`` must yield ``None`` rather than crashing.

    The underlying ``wordcloud`` library raises ``ValueError`` when given an
    empty text payload. ``create_wordcloud`` should detect this case (e.g. a
    very short transcript whose tokens were all filtered as stopwords) and
    return ``None`` so the calling pipeline can skip rendering gracefully.
    """
    wc = WordCounter.__new__(WordCounter)
    wc.language = "de"
    wc.font_path = None  # type: ignore[assignment]
    wc.word_counts = Counter()

    assert wc.create_wordcloud() is None
