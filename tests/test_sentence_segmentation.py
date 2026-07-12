"""Tests for the sentence-restoration agent."""

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
