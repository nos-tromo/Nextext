"""Tests for the sentence-restoration agent."""

from typing import Any, override

import pytest

from nextext.core import sentence_segmentation
from nextext.core.openai_cfg import InferencePipeline
from nextext.core.sentence_segmentation import _segment_run, restore_sentence_segments, terminal_punctuation_ratio


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
